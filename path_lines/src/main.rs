use std::borrow::Cow;
use std::path::Path;

use wgpu::util::DeviceExt;
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

#[rustfmt::skip]
const PENTAGON_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [0.5, 0.0, 0.5] }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [0.5, 0.0, 0.5] }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [0.5, 0.0, 0.5] }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.5, 0.0, 0.5] }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [0.5, 0.0, 0.5] }, // E
];

#[rustfmt::skip]
const PENTAGON_INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

#[rustfmt::skip]
const SQUARE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.5, 0.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.5, 0.0] },
    Vertex { position: [-0.5, 0.5, 0.0], color: [0.0, 0.5, 0.0] },
    Vertex { position: [0.5, 0.5, 0.0], color: [0.0, 0.0, 0.5] },
];

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

struct State {
    window: Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    current_pipline: CurrentPipeline,
    pipeline1: Pipeline1,
    pipeline2: Pipeline2,

    color: wgpu::Color,
    flow_field: FlowField,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CurrentPipeline {
    P1 = 1,
    P2 = 2,
}

struct Pipeline1 {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

struct Pipeline2 {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    /// [
    ///     wgpu::VertexAttribute {
    ///         offset: 0,
    ///         shader_location: 0,
    ///         format: wgpu::VertexFormat::Float32x3,
    ///     },
    ///     wgpu::VertexAttribute {
    ///         offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
    ///         shader_location: 1,
    ///         format: wgpu::VertexFormat::Float32x3,
    ///     },
    /// ],
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
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

        let pipeline1 = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipline Layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline 1"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc()],
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

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer 1"),
                contents: bytemuck::cast_slice(PENTAGON_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let num_indices = PENTAGON_INDICES.len() as u32;
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer 1"),
                contents: bytemuck::cast_slice(PENTAGON_INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });

            Pipeline1 {
                render_pipeline,
                vertex_buffer,
                index_buffer,
                num_indices,
            }
        };

        let pipeline2 = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipline Layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline 2"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc()],
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
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
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

            let num_vertices = SQUARE_VERTICES.len() as u32;
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer 2"),
                contents: bytemuck::cast_slice(SQUARE_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

            Pipeline2 {
                render_pipeline,
                vertex_buffer,
                num_vertices,
            }
        };

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,

            current_pipline: CurrentPipeline::P1,
            pipeline1,
            pipeline2,

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
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        scancode,
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Space),
                        ..
                    },
                ..
            } => {
                use CurrentPipeline::*;
                self.current_pipline = match self.current_pipline {
                    P1 => P2,
                    P2 => P1,
                };
                true
            }
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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

            match self.current_pipline {
                CurrentPipeline::P1 => {
                    let p1 = &self.pipeline1;
                    render_pass.set_pipeline(&p1.render_pipeline);
                    render_pass.set_vertex_buffer(0, p1.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(p1.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..p1.num_indices, 0, 0..1);
                }
                CurrentPipeline::P2 => {
                    let p2 = &self.pipeline2;
                    render_pass.set_pipeline(&p2.render_pipeline);
                    render_pass.set_vertex_buffer(0, p2.vertex_buffer.slice(..));
                    render_pass.draw(0..p2.num_vertices, 0..1);
                }
            }
        }

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
