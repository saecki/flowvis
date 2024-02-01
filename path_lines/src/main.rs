use std::borrow::Cow;
use std::path::Path;
use std::time::{Duration, Instant};

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
const FRAME_SIZE: usize = X_CELLS * Y_CELLS;
const TOTAL_ELEMS: usize = FRAME_SIZE * T_CELLS;

const VELOCITY_TEXTURE_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: X_CELLS as u32,
    height: Y_CELLS as u32,
    depth_or_array_layers: 1,
};

struct FlowField {
    max_velocity: f32,
    data: Vec<Vec2>,
}

impl FlowField {
    #[cfg(target_endian = "little")]
    fn read(path: &Path) -> anyhow::Result<FlowField> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;
        let mut data = vec![Vec2::ZERO; TOTAL_ELEMS];
        let raw = bytemuck::cast_slice_mut(data.as_mut_slice());
        file.read_exact(raw)?;

        let max_velocity = data
            .iter()
            .copied()
            .map(|v| v.norm())
            .max_by(f32::total_cmp)
            .unwrap();
        Ok(FlowField { max_velocity, data })
    }

    fn get(&self, t: usize, (x, y): (usize, usize)) -> Vec2 {
        self.data[t * Y_CELLS * X_CELLS + y * X_CELLS + x]
    }

    fn frame(&self, t: usize) -> &[Vec2] {
        &self.data[t * FRAME_SIZE..(t + 1) * FRAME_SIZE]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

struct State {
    window: Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    bg_pipeline: BgPipeline,

    filter: bool,
    play: bool,
    last_frame_uploaded: Instant,
    current_frame: usize,
    uploaded_frame: usize,
    flow_field: FlowField,
}

struct BgPipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    velocity_texture: wgpu::Texture,
    velocity_bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorVertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl ColorVertex {
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
            array_stride: std::mem::size_of::<ColorVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
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
    flow_field: &FlowField,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    filter: bool,
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
    write_frame_to_texture(queue, &velocity_texture, flow_field.frame(0));

    let velocity_texture_view =
        velocity_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let velocity_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: match filter {
            true => wgpu::FilterMode::Linear,
            false => wgpu::FilterMode::Nearest,
        },
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let max_velocity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("max_velocity_buffer"),
        contents: bytemuck::cast_slice(&[flow_field.max_velocity]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: filter },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // should match filterable field above
                    ty: wgpu::BindingType::Sampler(match filter {
                        true => wgpu::SamplerBindingType::Filtering,
                        false => wgpu::SamplerBindingType::NonFiltering,
                    }),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // should match filterable field above
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("max_velocity"),
        });

    let velocity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &texture_bind_group_layout,
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
                resource: max_velocity_buffer.as_entire_binding(),
            },
        ],
        label: Some("velocity_bind_group"),
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("texture_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("texture_shader.wgsl"))),
    });
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bg_render_pipline_layout"),
        bind_group_layouts: &[&texture_bind_group_layout],
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
        TextureVertex { position: [-1.0, -1.0, 0.0], tex_coords: [0.0, 1.0] },
        TextureVertex { position: [ 1.0, -1.0, 0.0], tex_coords: [1.0, 1.0] },
        TextureVertex { position: [ 1.0,  1.0, 0.0], tex_coords: [1.0, 0.0] },
        TextureVertex { position: [-1.0,  1.0, 0.0], tex_coords: [0.0, 0.0] },
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
        velocity_bind_group,
    }
}

fn write_frame_to_texture(queue: &wgpu::Queue, texture: &wgpu::Texture, frame: &[Vec2]) {
    let velocities = frame.iter().map(|v| v.norm()).collect::<Vec<_>>();

    let pixel_size = std::mem::size_of::<f32>() as u32;
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
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
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

        let filter = true;
        let bg_pipeline = create_bg_pipeline(&flow_field, &device, &queue, &config, filter);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,

            filter,
            play: true,
            last_frame_uploaded: Instant::now(),
            current_frame: 0,
            uploaded_frame: 0,
            bg_pipeline,

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
                        virtual_keycode: Some(keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match keycode {
                VirtualKeyCode::Space => {
                    self.play = !self.play;
                }
                VirtualKeyCode::Comma => {
                    self.current_frame = self.current_frame.checked_sub(1).unwrap_or(T_CELLS - 1);
                }
                VirtualKeyCode::Period => {
                    self.current_frame = (self.current_frame + 1) % T_CELLS;
                }
                VirtualKeyCode::F => {
                    self.filter = !self.filter;
                    self.bg_pipeline = create_bg_pipeline(
                        &self.flow_field,
                        &self.device,
                        &self.queue,
                        &self.config,
                        self.filter,
                    );
                }
                _ => (),
            },
            _ => (),
        }
        false
    }

    fn update(&mut self) {
        let now = Instant::now();
        let desired_delta = Duration::from_secs_f32(T_STEP);
        let actual_delta = now.duration_since(self.last_frame_uploaded);
        if self.play && actual_delta >= desired_delta {
            self.current_frame = (self.current_frame + 1) % T_CELLS;
        }
        if self.current_frame != self.uploaded_frame {
            write_frame_to_texture(
                &self.queue,
                &self.bg_pipeline.velocity_texture,
                self.flow_field.frame(self.current_frame),
            );
            self.last_frame_uploaded = now;
            self.uploaded_frame = self.current_frame;
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
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
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
                render_pass.set_bind_group(0, &bg.velocity_bind_group, &[]);
                render_pass.set_vertex_buffer(0, bg.vertex_buffer.slice(..));
                render_pass.set_index_buffer(bg.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..bg.num_indices, 0, 0..1);
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
