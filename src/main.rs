use std::borrow::Cow;
use std::time::{Duration, Instant};

use cgmath::{Matrix4, SquareMatrix, Vector2, Vector4};
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

    mouse: Mouse,
    keyboard: Keyboard,
    transform: Transform,
    transform_buffer: wgpu::Buffer,

    playback: PlaybackState,
    bg: BgState,
    line: LineState,
    arrow: ArrowState,

    flow_field: flow::Field,
}

struct PlaybackState {
    play: bool,
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
}

struct BgState {
    filter: bool,
    current_color_map: usize,
    uploaded_color_map: usize,
}

struct LineState {
    origins: Vec<flow::Pos2>,
    /// draw a line at the cursor position
    interactive: bool,
    /// recompute stream lines
    invalidated: bool,
    current_color_map: usize,
    uploaded_color_map: usize,
}

struct ArrowState {
    current_color_map: usize,
    uploaded_color_map: usize,
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

    pub fn pan_by(&mut self, delta: Vector2<f32>) {
        let Vector2 { x, y } = self.offset + delta;
        self.offset = Vector2::new(x.clamp(-2.0, 2.0), y.clamp(-2.0, 2.0));
    }

    pub fn zoom_by(&mut self, steps: f32) {
        self.zoom = (self.zoom + 0.125 * steps).clamp(0.125, 20.0);
    }
}

impl Transform {
    fn build_matrix(&self, aspect: f32) -> Matrix4<f32> {
        let Self {
            offset,
            zoom,
            rotation,
        } = *self;

        #[rustfmt::skip]
        let offset_mat = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            offset.x, offset.y, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let zoom_mat = Matrix4::new(
            zoom, 0.0, 0.0, 0.0,
            0.0, zoom, 0.0, 0.0,
            0.0, 0.0, zoom, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let rot_mat = Matrix4::new(
            rotation.cos(), rotation.sin(), 0.0, 0.0,
            -rotation.sin(), rotation.cos(), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let aspect_mat = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, aspect,  0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        zoom_mat * offset_mat * aspect_mat * rot_mat
    }
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TranformUniform([[f32; 4]; 4]);

impl From<Matrix4<f32>> for TranformUniform {
    fn from(value: Matrix4<f32>) -> Self {
        // SAFETY: Matrix4<f32> is layed out the same
        unsafe { std::mem::transmute(value) }
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
            .filter(|f| f.is_srgb())
            .next()
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
        let transform_uniform: TranformUniform = transform.build_matrix(aspect).into();
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("transform_buffer"),
            contents: bytemuck::cast_slice(&[transform_uniform]),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
        });

        let playback = PlaybackState {
            play: true,
            current_frame: 0,
            uploaded_frame: 0,
            last_frame_uploaded: Instant::now(),
        };

        let bg = BgState {
            filter: true,
            current_color_map: 0,
            uploaded_color_map: 0,
        };
        let bg_pipeline = create_bg_pipeline(
            &device,
            &queue,
            &config,
            &flow_field,
            &transform_buffer,
            &bg,
            playback.current_frame,
        );

        let line = LineState {
            origins: Vec::new(),
            interactive: false,
            invalidated: false,
            current_color_map: 0,
            uploaded_color_map: 0,
        };
        let line_pipeline = create_line_pipeline(&device, &queue, &config, &transform_buffer);

        let arrow = ArrowState {
            current_color_map: 0,
            uploaded_color_map: 0,
        };
        let arrow_pipeline = create_arrow_pipeline(&device, &queue, &config, &transform_buffer);

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

            mouse: Mouse::default(),
            keyboard: Keyboard::default(),
            transform,
            transform_buffer,

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
                KeyCode::KeyC if key_state.is_pressed() => {
                    self.bg.current_color_map =
                        (self.bg.current_color_map + 1) % BG_COLOR_MAPS.len();
                    true
                }
                KeyCode::KeyF if key_state.is_pressed() => {
                    self.bg.filter = !self.bg.filter;
                    self.bg_pipeline = create_bg_pipeline(
                        &self.device,
                        &self.queue,
                        &self.config,
                        &self.flow_field,
                        &self.transform_buffer,
                        &self.bg,
                        self.playback.current_frame,
                    );
                    true
                }
                KeyCode::KeyI if key_state.is_pressed() => {
                    self.line.interactive = !self.line.interactive;
                    self.line.invalidated = true;
                    true
                }
                KeyCode::KeyL if key_state.is_pressed() && self.keyboard.shift_down() => {
                    self.line.origins.clear();
                    let num = 4 * flow::Y_CELLS;
                    self.line
                        .origins
                        .extend((0..num).map(|i| flow::Pos2 {
                            x: 0.0,
                            y: (flow::Y_CELLS - 1) as f32 * i as f32 / (num - 1) as f32,
                        }));
                    self.line.invalidated = true;
                    true
                }
                KeyCode::KeyL if key_state.is_pressed() => {
                    self.line.origins.clear();
                    let num = 4 * flow::Y_CELLS;
                    let new_origins = (0..num).map(|i| {
                        let i = 2.0 * (i as f32 / (num - 1) as f32) - 1.0;
                        let y = i.signum() * i.abs().powf(1.5);
                        let pos = Vector2::new(-1.0, 0.125 * y);
                        normalized_to_flow_pos(pos).unwrap()
                    });
                    self.line.origins.extend(new_origins);
                    self.line.invalidated = true;
                    true
                }
                KeyCode::Delete if key_state.is_pressed() => {
                    self.line.origins.clear();
                    self.line.invalidated = true;
                    true
                }

                KeyCode::Minus if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.transform.zoom_by(-1.0);
                    true
                }
                KeyCode::Equal if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.transform.zoom_by(1.0);
                    true
                }
                KeyCode::Backspace if key_state.is_pressed() && self.keyboard.ctrl_down() => {
                    self.transform.reset();
                    true
                }
                _ => false,
            },
            WindowEvent::CursorMoved { position, .. } => {
                let aspect = self.config.width as f32 / self.config.height as f32;
                let transform_mat = self.transform.build_matrix(aspect);
                let homogeneous = Vector4::new(
                    2.0 * (position.x as f32 / (self.size.width - 1) as f32) - 1.0,
                    // flip y pos
                    -2.0 * (position.y as f32 / (self.size.height - 1) as f32) + 1.0,
                    0.0,
                    1.0,
                );
                let inverse_mat = transform_mat.invert().expect("should always be invertable");
                let new_pos = inverse_mat * homogeneous;
                let new_pos = Vector2::new(new_pos.x, new_pos.y);

                let in_bounds =
                    (-1.0..=1.0).contains(&new_pos.x) && (-1.0..=1.0).contains(&new_pos.y);
                if self.mouse.middle_down {
                    if let Some(old_pos) = self.mouse.pos {
                        let delta = new_pos - old_pos;
                        self.transform.pan_by(delta);
                    }
                } else {
                    self.mouse.pos = in_bounds.then_some(new_pos);
                }
                self.line.invalidated = true;

                true
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse.pos = None;
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_x, y) => {
                        self.transform.zoom_by(*y);
                        true
                    }
                    winit::event::MouseScrollDelta::PixelDelta(_) => {
                        // TODO
                        true
                    }
                }
            }
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
                    if let Some(flow_pos) = self.mouse.pos.and_then(normalized_to_flow_pos) {
                        self.line.origins.retain(|&o| (o - flow_pos).norm() >= 0.5);
                        self.line.invalidated = true;
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
        let transform_uniform: TranformUniform = self.transform.build_matrix(aspect).into();
        self.queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&[transform_uniform]),
        );

        let now = Instant::now();
        let desired_delta = Duration::from_secs_f32(flow::T_STEP);
        let actual_delta = now.duration_since(playback.last_frame_uploaded);
        if playback.play && actual_delta >= desired_delta {
            playback.next_frame();
        }
        if playback.current_frame != playback.uploaded_frame {
            write_frame_to_texture(
                &self.queue,
                &self.bg_pipeline.velocity_texture,
                self.flow_field.frame(playback.current_frame),
                self.flow_field.max_velocity,
            );
            playback.last_frame_uploaded = now;
            playback.uploaded_frame = playback.current_frame;
            line.invalidated = true;
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
        if line.invalidated {
            let pl = &mut self.line_pipeline;
            pl.vertices.clear();
            pl.indices.clear();
            for p in line.origins.iter() {
                compute_stream_line(
                    &mut pl.vertices,
                    &mut pl.indices,
                    &self.flow_field,
                    playback.current_frame,
                    *p,
                );
            }
            if line.interactive {
                if let Some(flow_pos) = self.mouse.pos.and_then(normalized_to_flow_pos) {
                    compute_stream_line(
                        &mut pl.vertices,
                        &mut pl.indices,
                        &self.flow_field,
                        playback.current_frame,
                        flow_pos,
                    );
                }
            }
            log::debug!("line vertices = {}", pl.vertices.len());
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
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            {
                let bg = &self.bg_pipeline;
                render_pass.set_pipeline(&bg.render_pipeline);
                render_pass.set_bind_group(0, &bg.bind_group, &[]);
                render_pass.set_vertex_buffer(0, bg.vertex_buffer.slice(..));
                render_pass.set_index_buffer(bg.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..bg.num_indices, 0, 0..1);
            }

            {
                let pl = &self.line_pipeline;
                render_pass.set_pipeline(&pl.render_pipeline);
                render_pass.set_bind_group(0, &pl.bind_group, &[]);
                render_pass.set_vertex_buffer(0, pl.vertex_buffer.slice(..));
                render_pass.set_index_buffer(pl.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..2 * pl.indices.len() as u32, 0, 0..1);
            }
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
    vertices: Vec<ScalarVertex>,
    indices: Vec<[u32; 2]>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    color_map_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

struct ArrowPipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertices: Vec<ScalarVertex>,
    indices: Vec<[u32; 3]>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarVertex {
    position: [f32; 3],
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
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextureVertex {
    position: [f32; 3],
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
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
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
    flow_field: &flow::Field,
    transform_buffer: &wgpu::Buffer,
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
    write_frame_to_texture(
        queue,
        &velocity_texture,
        flow_field.frame(current_frame),
        flow_field.max_velocity,
    );
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
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    #[rustfmt::skip]
    const BG_VERTICES: &[TextureVertex] = &[
        TextureVertex { position: [-1.0, -0.125, 0.0], tex_coords: [0.0, 1.0] },
        TextureVertex { position: [ 1.0, -0.125, 0.0], tex_coords: [1.0, 1.0] },
        TextureVertex { position: [ 1.0,  0.125, 0.0], tex_coords: [1.0, 0.0] },
        TextureVertex { position: [-1.0,  0.125, 0.0], tex_coords: [0.0, 0.0] },
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

fn write_frame_to_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    frame: flow::Frame,
    max_velocity: f32,
) {
    type PixelType = f32;
    let velocities = frame
        .iter()
        .map(|v| v.norm() / max_velocity)
        .collect::<Vec<PixelType>>();

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
) -> FieldLinePipeline {
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
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("line_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("color_shader.wgsl"))),
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
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    #[rustfmt::skip]
    let vertices = Vec::new();
    let indices = Vec::new();

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
) -> ArrowPipeline {
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
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("line_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("color_shader.wgsl"))),
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
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    #[rustfmt::skip]
    let vertices = Vec::new();
    let indices = Vec::new();

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
        bind_group,
    }
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

fn compute_stream_line(
    vertices: &mut Vec<ScalarVertex>,
    indices: &mut Vec<[u32; 2]>,
    flow_field: &flow::Field,
    current_frame: usize,
    start_pos: flow::Pos2,
) {
    fn flow_pos_to_wgpu_coord(pos: flow::Pos2) -> [f32; 3] {
        [
            2.0 * (pos.x / (flow::X_CELLS - 1) as f32) - 1.0,
            0.25 * (pos.y / (flow::Y_CELLS - 1) as f32) - 0.125,
            0.0,
        ]
    }

    let frame = flow_field.frame(current_frame);
    let first_index = vertices.len();
    let mut empty = true;
    let mut pos = start_pos;

    const MIN_VELOCITY: f32 = 0.01;
    const MAX_NUM_STEPS: usize = 100_000;
    const STEP_SIZE: f32 = 0.5;
    loop {
        let velocity = bilinear_lookup(&frame, pos);
        if velocity.norm() < MIN_VELOCITY {
            break;
        }
        if vertices.len() - first_index > MAX_NUM_STEPS {
            break;
        }

        // TODO: runge-kutta method
        pos += velocity * STEP_SIZE;
        if pos.x < 0.0
            || pos.y < 0.0
            || pos.x > (flow::X_CELLS - 1) as f32
            || pos.y > (flow::Y_CELLS - 1) as f32
        {
            break;
        }

        let color = velocity.norm() / flow_field.max_velocity;
        if empty {
            vertices.push(ScalarVertex {
                position: flow_pos_to_wgpu_coord(start_pos),
                scalar: color,
            });
            empty = false;
        }
        vertices.push(ScalarVertex {
            position: flow_pos_to_wgpu_coord(pos),
            scalar: color,
        });
    }

    if first_index < vertices.len() {
        indices.extend((first_index..vertices.len() - 1).map(|i| [i as u32, i as u32 + 1]));
    }
}

fn bilinear_lookup(frame: &flow::Frame, pos: flow::Pos2) -> flow::Vec2 {
    let aa = frame.get(pos.x.floor() as usize, pos.y.floor() as usize);
    let ab = frame.get(pos.x.floor() as usize, pos.y.ceil() as usize);
    let ba = frame.get(pos.x.ceil() as usize, pos.y.floor() as usize);
    let bb = frame.get(pos.x.ceil() as usize, pos.y.ceil() as usize);

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
    (-0.125..=0.125).contains(&pos.y).then(|| flow::Pos2 {
        x: 0.5 * (pos.x + 1.0) * (flow::X_CELLS - 1) as f32,
        y: 4.0 * (pos.y + 0.125) * (flow::Y_CELLS - 1) as f32,
    })
}
