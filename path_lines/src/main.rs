use std::path::Path;

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

const X_CELLS: usize = 400;
const X_START: f32 = -0.5;
const X_END: f32 = 7.5;
const X_STEP: f32 = (X_END - X_START) / X_CELLS as f32;
const Y_CELLS: usize = 50;
const Y_START: f32 = -0.5;
const Y_END: f32 = 0.5;
const Y_STEP: f32 = (Y_END - Y_START) / Y_CELLS as f32;
const T_CELLS: usize = 1001;
const T_START: f32 = 15.0;
const T_END: f32 = 23.0;
const T_STEP: f32 = (T_END - T_START) / T_CELLS as f32;
const NUM_ELEMS: usize = X_CELLS * Y_CELLS * T_CELLS;

struct FlowField {
    data: Vec<Vec2>,
}

impl FlowField {
    #[cfg(target_endian = "little")]
    fn read(path: &Path) -> anyhow::Result<FlowField> {
        const ELEM_SIZE: usize = std::mem::size_of::<Vec2>();
        let mut raw = std::fs::read(path)?;
        let num_elems = raw.len() / ELEM_SIZE;
        if num_elems != NUM_ELEMS {
            anyhow::bail!("Expected ");
        }

        raw.truncate(num_elems * ELEM_SIZE);

        // SAFETY: the target machine is little endian and the containers length is a multiple of the
        // struct size
        let data = unsafe { std::mem::transmute(raw) };
        Ok(FlowField { data })
    }

    fn get(&self, t: usize, (x, y): (usize, usize)) -> Vec2 {
        self.data[t * Y_CELLS * X_CELLS + y * X_CELLS + x]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    color: wgpu::Color,
    flow_field: FlowField,
}

impl State {
    async fn new(window: Window, flow_field: FlowField) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // SAFETY: the surface needs to live as long as the window that created it.
        // The State struct owns the window, so this should be safe
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

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
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
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
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            color: wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            },
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
            WindowEvent::CursorMoved { position, .. } => {
                self.color.r = position.x / self.size.width as f64;
                self.color.b = position.y / self.size.height as f64;
                true
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        // TODO
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    if let Err(e) = pollster::block_on(run()) {
        eprintln!("{e}");
    }
}

async fn run() -> anyhow::Result<()> {
    env_logger::init();

    let flow_field = FlowField::read("../flow.raw".as_ref())?;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let scale = 4;
    window.set_inner_size(PhysicalSize::new(
        scale * X_CELLS as u32,
        scale * Y_CELLS as u32,
    ));
    window.set_resizable(false);

    let mut state = State::new(window, flow_field).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { window_id, event } if window_id == state.window().id() => {
            if !state.input(&event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(*new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) => {
            if window_id == state.window().id() {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure if the surface is lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        *control_flow = ControlFlow::ExitWithCode(1);
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
        }
        Event::MainEventsCleared => {
            state.window().request_redraw();
        }
        _ => {}
    });
}
