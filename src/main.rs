use std::borrow::Cow;
use std::time::{Duration, Instant};

use cgmath::{InnerSpace, Matrix3, SquareMatrix, Vector2, Vector3};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{Event, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Theme, Window, WindowBuilder};

use crate::color_map::ColorMap;

mod color_map;
mod flow;

const DEFAULT_SCALE: u32 = 4;
const BG_COLOR_MAPS: [&ColorMap; 3] = [&color_map::GRAY, &color_map::INFERNO, &color_map::VIRIDIS];
const LINE_COLOR_MAPS: [&ColorMap; 3] = [&color_map::INFERNO, &color_map::VIRIDIS, &color_map::RED];
const ARROW_COLOR_MAPS: [&ColorMap; 3] =
    [&color_map::INFERNO, &color_map::VIRIDIS, &color_map::RED];

const SPEED_STEP_SIZE: f32 = 0.125;
const MIN_SPEED: f32 = 0.125;
const MAX_SPEED: f32 = 4.0;
const PAN_STEP_SIZE: f32 = 0.02;
const MAX_PAN_DIST: f32 = 2.0;
const ZOOM_FACTOR: f32 = 1.1;
const MIN_ZOOM: f32 = 0.125;
const MAX_ZOOM: f32 = 100.0;

const NUM_SPAWN_LINES: usize = 2 * flow::Y_CELLS;

const MIN_ARROW_STEP: f32 = 0.25;
const MAX_ARROW_STEP: f32 = 32.0;

const VELOCITY_TEXTURE_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: flow::X_CELLS as u32,
    height: flow::Y_CELLS as u32,
    depth_or_array_layers: 1,
};
const COLOR_MAP_TEXTURE_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: color_map::SIZE as u32,
    height: 1,
    depth_or_array_layers: 1,
};

fn main() {
    if let Err(e) = pollster::block_on(run()) {
        eprintln!("{e}");
    }
}

async fn run() -> anyhow::Result<()> {
    env_logger::init();

    let flow_field = flow::Field::read("flow.raw".as_ref())?;

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    const DEFAULT_CONTENT_SIZE: PhysicalSize<u32> = PhysicalSize {
        width: DEFAULT_SCALE * flow::X_CELLS as u32,
        height: DEFAULT_SCALE * flow::Y_CELLS as u32,
    };
    _ = window.request_inner_size(DEFAULT_CONTENT_SIZE);
    window.set_title("flowvis");
    window.set_theme(Some(Theme::Dark));

    let mut state = State::new(window, flow_field).await;

    event_loop.run(move |event, window_target| match event {
        Event::WindowEvent { window_id, event } if window_id == state.window().id() => {
            if !state.input(&event) {
                match event {
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure if the surface is lost
                            Err(wgpu::SurfaceError::Lost) => {
                                state.resize(state.size);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                window_target.exit();
                            }
                            Err(e) => eprintln!("{e}"),
                        }
                    }
                    WindowEvent::CloseRequested => {
                        window_target.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        // TODO: does this need handling
                    }
                    _ => {}
                }
            }
        }
        Event::AboutToWait => {
            state.window().request_redraw();
        }
        _ => {}
    })?;

    Ok(())
}

struct State {
    window: Window,
    surface: wgpu::Surface<'static>, // is this ok ???
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    bg_pipeline: BgPipeline,
    line_pipeline: FieldLinePipeline,
    arrow_pipeline: ArrowPipeline,
    text_pipeline: TextPipeline,

    mouse: Mouse,
    keyboard: Keyboard,
    transform: Transform,
    transform_buffer: wgpu::Buffer,
    max_velocity_buffer: wgpu::Buffer,

    playback: PlaybackState,
    bg: BgState,
    line: LineState,
    arrow: ArrowState,

    flow_field: flow::Field,
}

struct PlaybackState {
    play: bool,
    speed: f32,
    current_frame: usize,
    uploaded_frame: usize,
    last_frame_uploaded: Instant,
}

impl PlaybackState {
    fn prev_frame(&mut self) {
        let frame = &mut self.current_frame;
        *frame = frame.checked_sub(1).unwrap_or(flow::T_CELLS - 1);
    }

    fn next_frame(&mut self) {
        let frame = &mut self.current_frame;
        *frame = (*frame + 1) % flow::T_CELLS;
    }

    fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(MIN_SPEED, MAX_SPEED)
    }

    fn slower(&mut self) {
        self.set_speed(self.speed - SPEED_STEP_SIZE);
    }

    fn faster(&mut self) {
        self.set_speed(self.speed + SPEED_STEP_SIZE);
    }
}

struct BgState {
    visible: bool,
    filter: bool,
    current_color_map: usize,
    uploaded_color_map: usize,
}

struct LineState {
    visible: bool,
    method: LineMethod,
    origins: Vec<flow::Pos2>,
    /// draw a line at the cursor position
    interactive: bool,
    /// recompute stream lines
    invalidated: bool,
    current_color_map: usize,
    uploaded_color_map: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LineMethod {
    /// Euler
    Euler,
    /// Runge-Kutta method of order 2
    Rk2,
    /// Runge-Kutta method of order 4
    Rk4,
}

impl LineMethod {
    fn cycle(&mut self) {
        use LineMethod::*;
        *self = match self {
            Euler => Rk2,
            Rk2 => Rk4,
            Rk4 => Euler,
        }
    }

    fn cycle_rev(&mut self) {
        use LineMethod::*;
        *self = match self {
            Euler => Rk4,
            Rk2 => Euler,
            Rk4 => Rk2,
        }
    }
}

struct ArrowState {
    visible: bool,
    step_size: f32,
    invalidated: bool,
    current_color_map: usize,
    uploaded_color_map: usize,
}

impl ArrowState {
    fn set_step_size(&mut self, step_size: f32) {
        self.step_size = (step_size).clamp(MIN_ARROW_STEP, MAX_ARROW_STEP);
    }

    fn larger_step_size(&mut self) {
        self.set_step_size(self.step_size * 2.0);
    }

    fn smaller_step_size(&mut self) {
        self.set_step_size(self.step_size * 0.5);
    }
}

#[derive(Default)]
struct Mouse {
    /// Normalized mouse position, interval: [-1, 1]
    pos: Option<Vector2<f32>>,
    // left_down: bool,
    // right_down: bool,
    middle_down: bool,
}

#[derive(Default)]
struct Keyboard {
    l_ctrl_down: bool,
    r_ctrl_down: bool,
    l_shift_down: bool,
    r_shift_down: bool,
}

impl Keyboard {
    pub fn ctrl_down(&self) -> bool {
        self.l_ctrl_down || self.r_ctrl_down
    }

    pub fn shift_down(&self) -> bool {
        self.l_shift_down || self.r_shift_down
    }
}

struct Transform {
    offset: Vector2<f32>,
    zoom: f32,
    rotation: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            offset: Vector2::new(0.0, 0.0),
            zoom: 1.0,
            rotation: 0.0,
        }
    }
}

impl Transform {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Pan by the normalized delta with pre-multiplied zoom.
    pub fn pan_by_premultiplied(&mut self, delta: Vector2<f32>) {
        let Vector2 { x, y } = self.offset + delta;
        self.offset = Vector2::new(
            x.clamp(-flow::X_SCALE * MAX_PAN_DIST, flow::X_SCALE * MAX_PAN_DIST),
            y.clamp(-flow::Y_SCALE * MAX_PAN_DIST, flow::Y_SCALE * MAX_PAN_DIST),
        );
    }

    /// Pan by the normalized delta.
    pub fn pan_by(&mut self, delta: Vector2<f32>) {
        self.pan_by_premultiplied(2.0 * delta / self.zoom);
    }

    fn zoom_internal(&mut self, new: f32) {
        self.zoom = new.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    pub fn zoom_discrete(&mut self, steps: i8) {
        let new = self.zoom * ZOOM_FACTOR.powi(steps as i32);
        self.zoom_internal(new);
    }

    pub fn zoom_smooth(&mut self, factor: f32) {
        let new = self.zoom * factor;
        self.zoom_internal(new);
    }
}

impl Transform {
    fn build_matrix(&self, aspect: f32) -> Matrix3<f32> {
        let Self {
            offset,
            zoom,
            rotation,
        } = *self;

        #[rustfmt::skip]
        let offset_mat = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            offset.x, offset.y, 1.0,
        );
        #[rustfmt::skip]
        let zoom_mat = Matrix3::new(
            zoom, 0.0, 0.0,
            0.0, zoom, 0.0,
            0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let rot_mat = Matrix3::new(
            rotation.cos(), rotation.sin(), 0.0,
            -rotation.sin(), rotation.cos(), 0.0,
            0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let aspect_mat = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, aspect,  0.0,
            0.0, 0.0, 1.0,
        );

        zoom_mat * offset_mat * aspect_mat * rot_mat
    }
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TransformUniform([[f32; 4]; 3]);

impl From<Matrix3<f32>> for TransformUniform {
    fn from(mat: Matrix3<f32>) -> Self {
        // each column of the `matrix3x3<f32>` has an alignment of 16 bytes
        Self([
            [mat.x.x, mat.x.y, mat.x.z, 0.0],
            [mat.y.x, mat.y.y, mat.y.z, 0.0],
            [mat.z.x, mat.z.y, mat.z.z, 0.0],
        ])
    }
}

impl State {
    async fn new(window: Window, flow_field: flow::Field) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // SAFETY: the surface needs to live as long as the window that created it.
        // The State struct owns the window, so this should be safe
        let surface = unsafe {
            let target = wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap();
            instance.create_surface_unsafe(target).unwrap()
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::FLOAT32_FILTERABLE,
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| {
                dbg!("oh no, anyway...");
                surface_caps.formats[0]
            });
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: Vec::new(),
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let transform = Transform::default();
        let aspect = config.width as f32 / config.height as f32;
        let transform_uniform: TransformUniform = transform.build_matrix(aspect).into();
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("transform_buffer"),
            contents: bytemuck::cast_slice(&[transform_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let max_velocity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("max_velocity_buffer"),
            contents: bytemuck::cast_slice(&[flow_field.max_velocity]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let playback = PlaybackState {
            play: true,
            speed: 1.0,
            current_frame: 0,
            uploaded_frame: 0,
            last_frame_uploaded: Instant::now(),
        };

        let bg = BgState {
            visible: true,
            filter: true,
            current_color_map: 0,
            uploaded_color_map: 0,
        };
        let bg_pipeline = create_bg_pipeline(
            &device,
            &queue,
            &config,
            &transform_buffer,
            &max_velocity_buffer,
            &flow_field,
            &bg,
            playback.current_frame,
        );

        let mut line = LineState {
            visible: false,
            method: LineMethod::Rk4,
            origins: Vec::new(),
            interactive: false,
            invalidated: true,
            current_color_map: 0,
            uploaded_color_map: 0,
        };
        spawn_lines1(&mut line);
        let line_pipeline = create_line_pipeline(
            &device,
            &queue,
            &config,
            &transform_buffer,
            &max_velocity_buffer,
            &flow_field,
            &line,
            playback.current_frame,
        );

        let arrow = ArrowState {
            visible: false,
            step_size: 2.0,
            invalidated: false,
            current_color_map: 0,
            uploaded_color_map: 0,
        };
        let arrow_pipeline = create_arrow_pipeline(
            &device,
            &queue,
            &config,
            &transform_buffer,
            &max_velocity_buffer,
            &flow_field,
            &arrow,
            playback.current_frame,
        );

        let mut text_pipeline = {
            let mut font_system = glyphon::FontSystem::new();
            let cache = glyphon::SwashCache::new();
            let mut atlas = glyphon::TextAtlas::new(&device, &queue, surface_format);
            let renderer = glyphon::TextRenderer::new(
                &mut atlas,
                &device,
                wgpu::MultisampleState::default(),
                None,
            );
            let buffer = glyphon::Buffer::new(
                &mut font_system,
                glyphon::Metrics {
                    font_size: 13.0,
                    line_height: 16.0,
                },
            );
            TextPipeline {
                font_system,
                cache,
                atlas,
                renderer,
                buffer,
            }
        };
        update_text(
            &config,
            &mut text_pipeline,
            &transform,
            &playback,
            &bg,
            &line,
            &arrow,
        );

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,

            bg_pipeline,
            line_pipeline,
            arrow_pipeline,
            text_pipeline,

            mouse: Mouse::default(),
            keyboard: Keyboard::default(),
            transform,
            transform_buffer,
            max_velocity_buffer,

            playback,
            bg,
            line,
            arrow,

            flow_field,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        state: key_state,
                        ..
                    },
                ..
            } => match keycode {
                KeyCode::ControlLeft => {
                    self.keyboard.l_ctrl_down = key_state.is_pressed();
                    true
                }
                KeyCode::ControlRight => {
                    self.keyboard.r_ctrl_down = key_state.is_pressed();
                    true
                }
                KeyCode::ShiftLeft => {
                    self.keyboard.l_shift_down = key_state.is_pressed();
                    true
                }
                KeyCode::ShiftRight => {
                    self.keyboard.r_shift_down = key_state.is_pressed();
                    true
                }

                // transform
                KeyCode::Minus if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.transform.zoom_discrete(-1);
                    true
                }
                KeyCode::Equal if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.transform.zoom_discrete(1);
                    true
                }
                KeyCode::Backspace if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.transform.reset();
                    true
                }

                // playback
                KeyCode::Space if key_state.is_pressed() => {
                    self.playback.play = !self.playback.play;
                    true
                }
                KeyCode::Comma if key_state.is_pressed() => {
                    self.playback.prev_frame();
                    true
                }
                KeyCode::Period if key_state.is_pressed() => {
                    self.playback.next_frame();
                    true
                }
                KeyCode::BracketLeft if key_state.is_pressed() => {
                    self.playback.slower();
                    true
                }
                KeyCode::BracketRight if key_state.is_pressed() => {
                    self.playback.faster();
                    true
                }

                // visibility and color-maps
                KeyCode::Digit1 if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.bg.current_color_map =
                        (self.bg.current_color_map + 1) % BG_COLOR_MAPS.len();
                    true
                }
                KeyCode::Digit1 if key_state.is_pressed() => {
                    self.bg.visible = !self.bg.visible;
                    true
                }
                KeyCode::Digit2 if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.line.current_color_map =
                        (self.line.current_color_map + 1) % LINE_COLOR_MAPS.len();
                    true
                }
                KeyCode::Digit2 if key_state.is_pressed() => {
                    self.line.visible = !self.line.visible;
                    true
                }
                KeyCode::Digit3 if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.arrow.current_color_map =
                        (self.arrow.current_color_map + 1) % ARROW_COLOR_MAPS.len();
                    true
                }
                KeyCode::Digit3 if key_state.is_pressed() => {
                    self.arrow.visible = !self.arrow.visible;
                    true
                }

                // bg
                KeyCode::KeyF if key_state.is_pressed() => {
                    self.bg.filter = !self.bg.filter;
                    self.bg_pipeline = create_bg_pipeline(
                        &self.device,
                        &self.queue,
                        &self.config,
                        &self.transform_buffer,
                        &self.max_velocity_buffer,
                        &self.flow_field,
                        &self.bg,
                        self.playback.current_frame,
                    );
                    true
                }

                // line
                KeyCode::KeyI if key_state.is_pressed() => {
                    self.line.interactive = !self.line.interactive;
                    self.line.invalidated = true;
                    true
                }
                KeyCode::KeyM if key_state.is_pressed() && self.keyboard.shift_down() => {
                    self.line.method.cycle_rev();
                    self.line.invalidated = true;
                    true
                }
                KeyCode::KeyM if key_state.is_pressed() => {
                    self.line.method.cycle();
                    self.line.invalidated = true;
                    true
                }
                KeyCode::KeyL if key_state.is_pressed() && self.keyboard.shift_down() => {
                    spawn_lines2(&mut self.line);
                    self.line.invalidated = true;
                    true
                }
                KeyCode::KeyL if key_state.is_pressed() => {
                    spawn_lines1(&mut self.line);
                    self.line.invalidated = true;
                    true
                }
                KeyCode::Delete if key_state.is_pressed() => {
                    self.line.origins.clear();
                    self.line.invalidated = true;
                    true
                }

                // arrow
                KeyCode::KeyA if key_state.is_pressed() && self.keyboard.shift_down() => {
                    self.arrow.larger_step_size();
                    self.arrow.invalidated = true;
                    true
                }
                KeyCode::KeyA if key_state.is_pressed() => {
                    self.arrow.smaller_step_size();
                    self.arrow.invalidated = true;
                    true
                }
                _ => false,
            },
            WindowEvent::CursorMoved { position, .. } => {
                // TODO: maybe highlight hovered lines?
                let aspect = self.config.width as f32 / self.config.height as f32;
                let transform_mat = self.transform.build_matrix(aspect);
                let homogeneous = Vector3::new(
                    2.0 * (position.x as f32 / (self.size.width - 1) as f32) - 1.0,
                    // flip y pos
                    -2.0 * (position.y as f32 / (self.size.height - 1) as f32) + 1.0,
                    1.0,
                );
                let inverse_mat = transform_mat.invert().expect("should always be invertable");
                let new_pos = inverse_mat * homogeneous;
                let new_pos = Vector2::new(new_pos.x, new_pos.y);

                match self.mouse.pos {
                    Some(old_pos) if self.mouse.middle_down => {
                        let delta = new_pos - old_pos;
                        self.transform.pan_by_premultiplied(delta);
                    }
                    _ => {
                        self.mouse.pos = Some(new_pos);
                    }
                }

                if self.line.interactive {
                    self.line.invalidated = true;
                }

                true
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse.pos = None;
                true
            }
            // Zoom
            WindowEvent::MouseWheel { delta, .. } if self.keyboard.ctrl_down() => match delta {
                winit::event::MouseScrollDelta::LineDelta(_x, y) => {
                    self.transform.zoom_discrete(*y as i8);
                    true
                }
                winit::event::MouseScrollDelta::PixelDelta(delta) => {
                    let normalized = delta.y as f32 / self.config.height as f32;
                    let abs = normalized.abs();
                    let factor = if normalized < 0.0 {
                        1.0 / (1.0 + 2.0 * abs)
                    } else {
                        1.0 + 2.0 * abs
                    };
                    self.transform.zoom_smooth(factor);
                    true
                }
            },
            // pan
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) if self.keyboard.shift_down() => {
                    self.transform.pan_by(Vector2::new(PAN_STEP_SIZE * y, 0.0));
                    true
                }
                winit::event::MouseScrollDelta::LineDelta(x, y) => {
                    self.transform
                        .pan_by(Vector2::new(PAN_STEP_SIZE * x, PAN_STEP_SIZE * y));
                    true
                }
                winit::event::MouseScrollDelta::PixelDelta(delta) if self.keyboard.shift_down() => {
                    let normalized = Vector2::new(
                        // flip y delta
                        -delta.y as f32 / self.config.height as f32,
                        0.0,
                    );
                    self.transform.pan_by(normalized);
                    true
                }
                winit::event::MouseScrollDelta::PixelDelta(delta) => {
                    let normalized = Vector2::new(
                        delta.x as f32 / self.config.width as f32,
                        // flip y delta
                        -delta.y as f32 / self.config.height as f32,
                    );
                    self.transform.pan_by(normalized);
                    true
                }
            },
            // TODO: handle touchpad events
            // WindowEvent::TouchpadMagnify { device_id, delta, phase } => (),
            // WindowEvent::TouchpadRotate { device_id, delta, phase } => (),
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => match button {
                MouseButton::Left if button_state.is_pressed() => {
                    if let Some(flow_pos) = self.mouse.pos.and_then(normalized_to_flow_pos) {
                        self.line.origins.push(flow_pos);
                        self.line.invalidated = true;
                    }
                    true
                }
                MouseButton::Right if button_state.is_pressed() => {
                    if let Some(pos) = self.mouse.pos {
                        let mut closest = None;

                        let last_origin_index = self.line_pipeline.first_vertices.last();
                        let vertcies = match (self.line.interactive, last_origin_index) {
                            (true, Some(&oi)) => &self.line_pipeline.vertices[..oi],
                            _ => &self.line_pipeline.vertices,
                        };
                        for (vi, v) in vertcies.iter().enumerate() {
                            let p: Vector2<f32> = unsafe { std::mem::transmute(v.position) };
                            let mag = (pos - p).magnitude();
                            if mag < 0.002 {
                                match closest {
                                    Some((other_mag, _)) => {
                                        if mag < other_mag {
                                            closest = Some((mag, vi));
                                        }
                                    }
                                    None => {
                                        closest = Some((mag, vi));
                                    }
                                }
                            }
                        }

                        if let Some((_, vi)) = closest {
                            let mut oi = self.line_pipeline.first_vertices.len() - 1;
                            for (i, fi) in self.line_pipeline.first_vertices.iter().enumerate() {
                                if vi < *fi {
                                    oi = i - 1;
                                    break;
                                }
                            }
                            self.line.origins.remove(oi);
                            self.line.invalidated = true;
                        }
                    }
                    true
                }
                MouseButton::Middle => {
                    self.mouse.middle_down = button_state.is_pressed();
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update(&mut self) {
        let playback = &mut self.playback;
        let bg = &mut self.bg;
        let line = &mut self.line;
        let arrow = &mut self.arrow;

        let aspect = self.config.width as f32 / self.config.height as f32;
        let transform_uniform: TransformUniform = self.transform.build_matrix(aspect).into();
        self.queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&[transform_uniform]),
        );

        let now = Instant::now();
        let desired_delta = Duration::from_secs_f32(flow::T_STEP / playback.speed);
        let actual_delta = now.duration_since(playback.last_frame_uploaded);
        if playback.play && actual_delta >= desired_delta {
            playback.next_frame();
        }
        if playback.current_frame != playback.uploaded_frame {
            if bg.visible {
                write_frame_to_texture(
                    &self.queue,
                    &self.bg_pipeline.velocity_texture,
                    self.flow_field.frame(playback.current_frame),
                );
            }
            playback.last_frame_uploaded = now;
            playback.uploaded_frame = playback.current_frame;
            line.invalidated = true;
            arrow.invalidated = true;
            log::debug!("frame_delta = {actual_delta:?}");
        }
        if bg.current_color_map != bg.uploaded_color_map {
            write_color_map_to_texture(
                &self.queue,
                &self.bg_pipeline.color_map_texture,
                BG_COLOR_MAPS[bg.current_color_map],
            );
            bg.uploaded_color_map = bg.current_color_map;
        }

        // line
        if line.current_color_map != line.uploaded_color_map {
            write_color_map_to_texture(
                &self.queue,
                &self.line_pipeline.color_map_texture,
                LINE_COLOR_MAPS[line.current_color_map],
            );
            line.uploaded_color_map = line.current_color_map;
        }
        if line.visible && line.invalidated {
            let pl = &mut self.line_pipeline;
            let start = Instant::now();
            compute_lines(
                &mut pl.first_vertices,
                &mut pl.vertices,
                &mut pl.indices,
                line,
                &self.flow_field,
                playback.current_frame,
                self.mouse.pos,
            );
            let end = Instant::now();
            log::debug!("line computation = {:?}", end - start);
            line.invalidated = false;

            log::debug!("line vertices = {}", pl.vertices.len());
            let start = Instant::now();
            update_buffer(
                &self.device,
                &self.queue,
                &mut pl.vertex_buffer,
                &pl.vertices,
                wgpu::BufferUsages::VERTEX,
            );
            update_buffer(
                &self.device,
                &self.queue,
                &mut pl.index_buffer,
                &pl.indices,
                wgpu::BufferUsages::INDEX,
            );
            let end = Instant::now();
            log::debug!("upload buffers = {:?}", end - start);
        }

        // arrow
        if arrow.current_color_map != arrow.uploaded_color_map {
            write_color_map_to_texture(
                &self.queue,
                &self.arrow_pipeline.color_map_texture,
                ARROW_COLOR_MAPS[arrow.current_color_map],
            );
            arrow.uploaded_color_map = arrow.current_color_map;
        }
        if arrow.visible && arrow.invalidated {
            let pl = &mut self.arrow_pipeline;
            self.queue.write_buffer(
                &pl.step_size_buffer,
                0,
                bytemuck::cast_slice(&[arrow.step_size]),
            );
            update_arrows(
                &mut pl.vertices,
                &mut pl.indices,
                &self.flow_field,
                playback.current_frame,
                arrow.step_size,
            );
            arrow.invalidated = false;

            log::debug!("arrow vertices = {}", pl.vertices.len());
            update_buffer(
                &self.device,
                &self.queue,
                &mut pl.vertex_buffer,
                &pl.vertices,
                wgpu::BufferUsages::VERTEX,
            );
            update_buffer(
                &self.device,
                &self.queue,
                &mut pl.index_buffer,
                &pl.indices,
                wgpu::BufferUsages::INDEX,
            );
        }

        update_text(
            &self.config,
            &mut self.text_pipeline,
            &self.transform,
            &self.playback,
            &self.bg,
            &self.line,
            &self.arrow,
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        {
            self.text_pipeline
                .renderer
                .prepare(
                    &self.device,
                    &self.queue,
                    &mut self.text_pipeline.font_system,
                    &mut self.text_pipeline.atlas,
                    glyphon::Resolution {
                        width: self.config.width,
                        height: self.config.height,
                    },
                    [glyphon::TextArea {
                        buffer: &self.text_pipeline.buffer,
                        left: 0.0,
                        top: 0.0,
                        scale: 1.0,
                        bounds: glyphon::TextBounds {
                            left: 0,
                            top: 0,
                            right: self.config.width as i32,
                            bottom: self.config.height as i32,
                        },
                        default_color: glyphon::Color::rgb(0xFF, 0x40, 0x80),
                    }],
                    &mut self.text_pipeline.cache,
                )
                .unwrap();

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if self.bg.visible {
                let pl = &self.bg_pipeline;
                render_pass.set_pipeline(&pl.render_pipeline);
                render_pass.set_bind_group(0, &pl.bind_group, &[]);
                render_pass.set_vertex_buffer(0, pl.vertex_buffer.slice(..));
                render_pass.set_index_buffer(pl.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..pl.num_indices, 0, 0..1);
            }

            if self.line.visible {
                let pl = &self.line_pipeline;
                render_pass.set_pipeline(&pl.render_pipeline);
                render_pass.set_bind_group(0, &pl.bind_group, &[]);
                render_pass.set_vertex_buffer(0, pl.vertex_buffer.slice(..));
                render_pass.set_index_buffer(pl.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..2 * pl.indices.len() as u32, 0, 0..1);
            }

            if self.arrow.visible {
                let pl = &self.arrow_pipeline;
                render_pass.set_pipeline(&pl.render_pipeline);
                render_pass.set_bind_group(0, &pl.bind_group, &[]);
                render_pass.set_vertex_buffer(0, pl.vertex_buffer.slice(..));
                render_pass.set_index_buffer(pl.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..3 * pl.indices.len() as u32, 0, 0..1);
            }

            self.text_pipeline
                .renderer
                .render(&self.text_pipeline.atlas, &mut render_pass)
                .unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct BgPipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    velocity_texture: wgpu::Texture,
    color_map_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

struct FieldLinePipeline {
    render_pipeline: wgpu::RenderPipeline,
    first_vertices: Vec<usize>,
    vertices: Vec<ScalarVertex>,
    indices: Vec<[u32; 2]>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    color_map_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

struct ArrowPipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertices: Vec<ArrowVertex>,
    indices: Vec<[u32; 3]>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    color_map_texture: wgpu::Texture,
    step_size_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct TextPipeline {
    font_system: glyphon::FontSystem,
    cache: glyphon::SwashCache,
    atlas: glyphon::TextAtlas,
    renderer: glyphon::TextRenderer,
    buffer: glyphon::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarVertex {
    position: [f32; 2],
    scalar: f32,
}

impl ScalarVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ScalarVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ArrowVertex {
    position: [f32; 2],
    velocity: [f32; 2],
}

impl ArrowVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ArrowVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextureVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

impl TextureVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextureVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

fn create_bg_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    transform_buffer: &wgpu::Buffer,
    max_velocity_buffer: &wgpu::Buffer,
    flow_field: &flow::Field,
    bg: &BgState,
    current_frame: usize,
) -> BgPipeline {
    let velocity_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("velocity_texture"),
        size: VELOCITY_TEXTURE_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_frame_to_texture(queue, &velocity_texture, flow_field.frame(current_frame));
    let velocity_texture_view =
        velocity_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let velocity_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: match bg.filter {
            true => wgpu::FilterMode::Linear,
            false => wgpu::FilterMode::Nearest,
        },
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let color_map_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bg_color_map_texture"),
        size: COLOR_MAP_TEXTURE_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_color_map_to_texture(
        queue,
        &color_map_texture,
        BG_COLOR_MAPS[bg.current_color_map],
    );
    let color_map_texture_view =
        color_map_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let color_map_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bg_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float {
                        filterable: bg.filter,
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // should match filterable field of the texture
                ty: wgpu::BindingType::Sampler(match bg.filter {
                    true => wgpu::SamplerBindingType::Filtering,
                    false => wgpu::SamplerBindingType::NonFiltering,
                }),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D1,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // should match filterable field of the texture
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&velocity_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&velocity_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&color_map_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&color_map_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: transform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: max_velocity_buffer.as_entire_binding(),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bg_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("texture_shader.wgsl"))),
    });
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bg_render_pipline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bg_render_pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[TextureVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    #[rustfmt::skip]
    const BG_VERTICES: &[TextureVertex] = &[
        TextureVertex { position: [-flow::X_SCALE, -flow::Y_SCALE], tex_coords: [0.0, 0.0] },
        TextureVertex { position: [ flow::X_SCALE, -flow::Y_SCALE], tex_coords: [1.0, 0.0] },
        TextureVertex { position: [ flow::X_SCALE,  flow::Y_SCALE], tex_coords: [1.0, 1.0] },
        TextureVertex { position: [-flow::X_SCALE,  flow::Y_SCALE], tex_coords: [0.0, 1.0] },
    ];
    #[rustfmt::skip]
    const BG_INDICES: &[u16] = &[
        0, 1, 3,
        1, 2, 3,
    ];
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bg_vertex_buffer"),
        contents: bytemuck::cast_slice(BG_VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bg_index_buffer"),
        contents: bytemuck::cast_slice(BG_INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });
    let num_indices = BG_INDICES.len() as u32;

    BgPipeline {
        render_pipeline,
        vertex_buffer,
        index_buffer,
        num_indices,
        velocity_texture,
        color_map_texture,
        bind_group,
    }
}

fn write_frame_to_texture(queue: &wgpu::Queue, texture: &wgpu::Texture, frame: flow::Frame) {
    type PixelType = f32;
    let velocities = frame.iter().map(|v| v.norm()).collect::<Vec<PixelType>>();

    let pixel_size = std::mem::size_of::<PixelType>() as u32;
    let bytes_per_row = pixel_size * VELOCITY_TEXTURE_SIZE.width;
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&velocities),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(VELOCITY_TEXTURE_SIZE.height),
        },
        VELOCITY_TEXTURE_SIZE,
    );
}

fn write_color_map_to_texture(queue: &wgpu::Queue, texture: &wgpu::Texture, map: &ColorMap) {
    type PixelType = [u8; 4];
    let pixel_size = std::mem::size_of::<PixelType>() as u32;
    let bytes_per_row = pixel_size * color_map::SIZE as u32;
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(map.as_slice()),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: None,
        },
        COLOR_MAP_TEXTURE_SIZE,
    );
}

fn create_line_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    transform_buffer: &wgpu::Buffer,
    max_velocity_buffer: &wgpu::Buffer,
    flow_field: &flow::Field,
    line: &LineState,
    current_frame: usize,
) -> FieldLinePipeline {
    let mut first_vertices = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    compute_lines(
        &mut first_vertices,
        &mut vertices,
        &mut indices,
        line,
        flow_field,
        current_frame,
        None,
    );

    let color_map_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("line_color_map_texture"),
        size: COLOR_MAP_TEXTURE_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    write_color_map_to_texture(
        queue,
        &color_map_texture,
        LINE_COLOR_MAPS[line.current_color_map],
    );
    let color_map_texture_view =
        color_map_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let color_map_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("line_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D1,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // should match filterable field of the texture
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("line_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_map_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&color_map_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: transform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: max_velocity_buffer.as_entire_binding(),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("line_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("line_shader.wgsl"))),
    });
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipline_layout_2"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("line_render_pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[ScalarVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("line_vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("line_index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });

    FieldLinePipeline {
        render_pipeline,
        first_vertices,
        vertices,
        indices,
        vertex_buffer,
        index_buffer,
        color_map_texture,
        bind_group,
    }
}

fn create_arrow_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    transform_buffer: &wgpu::Buffer,
    max_velocity_buffer: &wgpu::Buffer,
    flow_field: &flow::Field,
    arrow: &ArrowState,
    current_frame: usize,
) -> ArrowPipeline {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    update_arrows(
        &mut vertices,
        &mut indices,
        flow_field,
        current_frame,
        arrow.step_size,
    );

    let color_map_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("arrow_color_map_texture"),
        size: COLOR_MAP_TEXTURE_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    write_color_map_to_texture(queue, &color_map_texture, &color_map::INFERNO);
    let color_map_texture_view =
        color_map_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let color_map_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let step_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("arrow_step_size_buffer"),
        contents: bytemuck::cast_slice(&[arrow.step_size]),
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("arrow_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D1,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // should match filterable field of the texture
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("arrow_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_map_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&color_map_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: transform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: max_velocity_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: step_size_buffer.as_entire_binding(),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("arrow_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("arrow_shader.wgsl"))),
    });
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("arrow_render_pipline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("arrow_line_render_pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[ArrowVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("arrow_vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("arrow_index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });

    ArrowPipeline {
        render_pipeline,
        vertices,
        indices,
        vertex_buffer,
        index_buffer,
        color_map_texture,
        step_size_buffer,
        bind_group,
    }
}

fn update_text(
    config: &wgpu::SurfaceConfiguration,
    pl: &mut TextPipeline,
    transform: &Transform,
    playback: &PlaybackState,
    bg: &BgState,
    line: &LineState,
    arrow: &ArrowState,
) {
    use std::fmt::Write as _;

    fn on_off_str(val: bool) -> &'static str {
        match val {
            true => "on",
            false => "off",
        }
    }
    fn write_or_pad(str: &mut String, display: bool, args: std::fmt::Arguments) {
        let start = str.len();
        str.write_fmt(args).ok();
        if !display {
            // SAFETY: we know the start boundary is valid, so we can safely fill the remainder
            // with spaces
            let bytes = unsafe { str[start..].as_bytes_mut() };
            bytes.fill(b' ');
        }
    }

    let frame = playback.current_frame;
    let speed = playback.speed;
    let zoom = transform.zoom;
    let mut text = format!("frame = {frame:4}  speed = {speed:.3}x  zoom = {zoom:.2}x");

    let bg_on = on_off_str(bg.visible);
    let filter = on_off_str(bg.filter);
    write!(&mut text, " | bg = {bg_on:3}").ok();
    #[rustfmt::skip]
    write_or_pad(&mut text, bg.visible, format_args!("  filter bg = {filter:3}"));

    let line_on = on_off_str(line.visible);
    let method = match line.method {
        LineMethod::Euler => "Euler",
        LineMethod::Rk2 => "RK2",
        LineMethod::Rk4 => "RK4",
    };
    let interactive = on_off_str(line.interactive);
    write!(&mut text, " | stream lines = {line_on:3}").ok();
    #[rustfmt::skip]
    write_or_pad(&mut text, line.visible, format_args!("  line method = {method:5}"));
    #[rustfmt::skip]
    write_or_pad(&mut text, line.visible, format_args!("  interactive line = {interactive:3}"));

    let arrow_on = on_off_str(arrow.visible);
    let arrow_step_size = arrow.step_size;
    write!(&mut text, " | arrows = {arrow_on:3}").ok();
    #[rustfmt::skip]
    write_or_pad(&mut text, arrow.visible, format_args!("  arrow step size = {arrow_step_size:.2}"));

    pl.buffer.set_size(
        &mut pl.font_system,
        config.width as f32,
        config.height as f32,
    );
    pl.buffer.set_text(
        &mut pl.font_system,
        &text,
        glyphon::Attrs::new().family(glyphon::Family::Monospace),
        glyphon::Shaping::Advanced,
    );
    pl.buffer.shape_until_scroll(&mut pl.font_system);
}

/// Either reuse the buffer if there is enough space, or destroy it and create a new one.
fn update_buffer<T: bytemuck::NoUninit>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut wgpu::Buffer,
    values: &[T],
    usage: wgpu::BufferUsages,
) {
    let raw = bytemuck::cast_slice(values);
    if buffer.size() < raw.len() as u64 {
        buffer.destroy();
        *buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line_index_buffer"),
            contents: raw,
            usage: usage | wgpu::BufferUsages::COPY_DST,
        });
    } else {
        queue.write_buffer(buffer, 0, raw);
    }
}

fn spawn_lines1(line: &mut LineState) {
    line.origins.clear();
    line.origins
        .extend((0..NUM_SPAWN_LINES).map(|i| flow::Pos2 {
            x: 0.0,
            y: (flow::Y_CELLS - 1) as f32 * i as f32 / (NUM_SPAWN_LINES - 1) as f32,
        }));
}

fn spawn_lines2(line: &mut LineState) {
    line.origins.clear();
    line.origins.extend((0..NUM_SPAWN_LINES).map(|i| {
        let i = 2.0 * (i as f32 / (NUM_SPAWN_LINES - 1) as f32) - 1.0;
        let y = i.signum() * i.abs().powf(1.5);
        let pos = Vector2::new(-flow::X_SCALE, flow::Y_SCALE * y);
        normalized_to_flow_pos(pos).unwrap()
    }));
}

fn compute_lines(
    first_vertices: &mut Vec<usize>,
    vertices: &mut Vec<ScalarVertex>,
    indices: &mut Vec<[u32; 2]>,
    line: &LineState,
    flow_field: &flow::Field,
    current_frame: usize,
    mouse_pos: Option<Vector2<f32>>,
) {
    first_vertices.clear();
    vertices.clear();
    indices.clear();
    for p in line.origins.iter() {
        compute_stream_line(
            first_vertices,
            vertices,
            indices,
            &flow_field,
            current_frame,
            *p,
            line.method,
        );
    }
    if line.interactive {
        if let Some(flow_pos) = mouse_pos.and_then(normalized_to_flow_pos) {
            compute_stream_line(
                first_vertices,
                vertices,
                indices,
                &flow_field,
                current_frame,
                flow_pos,
                line.method,
            );
        }
    }
}

fn compute_stream_line(
    first_vertices: &mut Vec<usize>,
    vertices: &mut Vec<ScalarVertex>,
    indices: &mut Vec<[u32; 2]>,
    flow_field: &flow::Field,
    current_frame: usize,
    start_pos: flow::Pos2,
    method: LineMethod,
) {
    let frame = flow_field.frame(current_frame);
    let first_vertex = vertices.len();
    let mut pos = start_pos;
    let mut v0 = bilinear_lookup(frame, pos);

    const MIN_VELOCITY: f32 = 0.01;
    const MAX_NUM_STEPS: usize = 10_000;
    const STEP_SIZE: f32 = 0.5;

    vertices.push(ScalarVertex {
        position: flow_pos_to_wgpu_coord(start_pos),
        scalar: v0.norm(),
    });

    loop {
        // RK4 method
        let k1 = v0 * STEP_SIZE;
        match method {
            LineMethod::Euler => {
                pos += k1;
            }
            LineMethod::Rk2 => {
                let p1 = pos + k1;
                if !flow::in_bounds(p1) {
                    break;
                }
                let k2 = bilinear_lookup(frame, p1) * STEP_SIZE;
                pos += (k1 + k2) * 0.5;
            }
            LineMethod::Rk4 => {
                let p1 = pos + k1 * 0.5;
                if !flow::in_bounds(p1) {
                    break;
                }
                let k2 = bilinear_lookup(frame, p1) * STEP_SIZE;
                let p2 = pos + k2 * 0.5;
                if !flow::in_bounds(p2) {
                    break;
                }
                let k3 = bilinear_lookup(frame, p2) * STEP_SIZE;
                let p3 = pos + k3;
                if !flow::in_bounds(p3) {
                    break;
                }
                let k4 = bilinear_lookup(frame, p3) * STEP_SIZE;
                let p4 = pos + k4;
                if !flow::in_bounds(p4) {
                    break;
                }

                pos += (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
            }
        }
        if !flow::in_bounds(pos) {
            break;
        }

        let v1 = bilinear_lookup(frame, pos);

        vertices.push(ScalarVertex {
            position: flow_pos_to_wgpu_coord(pos),
            scalar: v1.norm(),
        });

        if v1.norm() < MIN_VELOCITY {
            break;
        }
        if vertices.len() - first_vertex > MAX_NUM_STEPS {
            break;
        }

        v0 = v1;
    }

    if first_vertex < vertices.len() {
        first_vertices.push(first_vertex as usize);
        indices.extend((first_vertex..vertices.len() - 1).map(|i| [i as u32, i as u32 + 1]));
    } else {
        vertices.pop();
    }
}

fn update_arrows(
    vertices: &mut Vec<ArrowVertex>,
    indices: &mut Vec<[u32; 3]>,
    flow_field: &flow::Field,
    current_frame: usize,
    arrow_step: f32,
) {
    vertices.clear();
    indices.clear();

    let frame = flow_field.frame(current_frame);
    let num_x_arrows = (flow::X_CELLS as f32 - 1.0) / arrow_step;
    let num_y_arrows = (flow::Y_CELLS as f32 - 1.0) / arrow_step;

    for y in 0..num_y_arrows.floor() as u32 {
        for x in 0..num_x_arrows.floor() as u32 {
            let flow_pos = flow::Pos2 {
                x: arrow_step as f32 * x as f32 + 0.5 * (1.0 + num_x_arrows.fract()) * arrow_step,
                y: arrow_step as f32 * y as f32 + 0.5 * (1.0 + num_y_arrows.fract()) * arrow_step,
            };
            let velocity = bilinear_lookup(frame, flow_pos);
            let vertex = ArrowVertex {
                position: flow_pos_to_wgpu_coord(flow_pos),
                velocity: [velocity.x, velocity.y],
            };

            let i = vertices.len() as u32;
            vertices.extend([vertex; 7]);
            indices.extend([
                [i + 0, i + 1, i + 2],
                [i + 3, i + 5, i + 4],
                [i + 4, i + 5, i + 6],
            ]);
        }
    }
}

fn bilinear_lookup(frame: flow::Frame, pos: flow::Pos2) -> flow::Vec2 {
    let aa = frame.get(pos.x as u32, pos.y as u32);
    let ab = frame.get(pos.x as u32, pos.y.ceil() as u32);
    let ba = frame.get(pos.x.ceil() as u32, pos.y as u32);
    let bb = frame.get(pos.x.ceil() as u32, pos.y.ceil() as u32);

    let u_x = pos.x.fract();
    let u_y = pos.y.fract();
    let a = lerp(u_y, aa, ab);
    let b = lerp(u_y, ba, bb);

    lerp(u_x, a, b)
}

#[inline(always)]
fn lerp(u: f32, a: flow::Vec2, b: flow::Vec2) -> flow::Vec2 {
    a * (1.0 - u) + b * u
}

fn normalized_to_flow_pos(pos: Vector2<f32>) -> Option<flow::Pos2> {
    let in_x = (-flow::X_SCALE..=flow::X_SCALE).contains(&pos.x);
    let in_y = (-flow::Y_SCALE..=flow::Y_SCALE).contains(&pos.y);

    (in_x && in_y).then(|| flow::Pos2 {
        x: 0.5 / flow::X_SCALE * (pos.x + flow::X_SCALE) * (flow::X_CELLS - 1) as f32,
        y: 0.5 / flow::Y_SCALE * (pos.y + flow::Y_SCALE) * (flow::Y_CELLS - 1) as f32,
    })
}

fn flow_pos_to_wgpu_coord(pos: flow::Pos2) -> [f32; 2] {
    [
        2.0 * flow::X_SCALE * (pos.x / (flow::X_CELLS - 1) as f32 - 0.5),
        2.0 * flow::Y_SCALE * (pos.y / (flow::Y_CELLS - 1) as f32 - 0.5),
    ]
}
