use std::borrow::Cow;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowBuilder};

use crate::color_map::ColorMap;

mod color_map;
mod flow;

const COLOR_MAPS: [&ColorMap; 3] = [&color_map::GRAY, &color_map::INFERNO, &color_map::VIRIDIS];

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
    const SCALE: u32 = 4;
    const CONTENT_SIZE: PhysicalSize<u32> = PhysicalSize {
        width: SCALE * flow::X_CELLS as u32,
        height: SCALE * flow::Y_CELLS as u32,
    };
    window.set_min_inner_size(Some(CONTENT_SIZE));
    window.set_max_inner_size(Some(CONTENT_SIZE));

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
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
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
    path_line_pipeline: FieldLinePipeline,

    filter: bool,
    play: bool,
    last_frame_uploaded: Instant,
    current_frame: usize,
    uploaded_frame: usize,
    current_color_map: usize,
    uploaded_color_map: usize,
    mouse_pos: flow::Pos2,
    line_origins: Vec<flow::Pos2>,
    lines_invalidated: bool,
    flow_field: flow::Field,
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
                    required_features: wgpu::Features::FLOAT32_FILTERABLE,
                    required_limits: wgpu::Limits::default(),
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

        let current_frame = 0;
        let current_color_map = 0;
        let filter = true;
        let bg_pipeline = create_bg_pipeline(
            &device,
            &queue,
            &config,
            &flow_field,
            current_frame,
            current_color_map,
            filter,
        );

        let path_line_pipeline = create_line_pipeline(&device, &queue, &config);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,

            bg_pipeline,
            path_line_pipeline,

            filter,
            play: true,
            last_frame_uploaded: Instant::now(),
            current_frame,
            uploaded_frame: current_frame,
            current_color_map,
            uploaded_color_map: current_color_map,
            mouse_pos: flow::Pos2 { x: 30.0, y: 25.0 },
            line_origins: Vec::new(),
            lines_invalidated: true,
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match keycode {
                KeyCode::Space => {
                    self.play = !self.play;
                    true
                }
                KeyCode::Comma => {
                    self.prev_frame();
                    true
                }
                KeyCode::Period => {
                    self.next_frame();
                    true
                }
                KeyCode::KeyC => {
                    self.current_color_map = (self.current_color_map + 1) % COLOR_MAPS.len();
                    true
                }
                KeyCode::KeyF => {
                    self.filter = !self.filter;
                    self.bg_pipeline = create_bg_pipeline(
                        &self.device,
                        &self.queue,
                        &self.config,
                        &self.flow_field,
                        self.current_frame,
                        self.current_color_map,
                        self.filter,
                    );
                    true
                }
                _ => false,
            },
            WindowEvent::CursorMoved { position, .. } => {
                let flow_pos = flow::Pos2 {
                    x: (flow::X_CELLS - 1) as f32
                        * (position.x as f32 / (self.size.width - 1) as f32),
                    // flip y pos
                    y: (flow::Y_CELLS - 1) as f32
                        * (1.0 - position.y as f32 / (self.size.height - 1) as f32),
                };

                self.mouse_pos = flow_pos;
                self.lines_invalidated = true;
                true
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => match button {
                MouseButton::Left => {
                    self.line_origins.push(self.mouse_pos);
                    self.lines_invalidated = true;
                    true
                }
                MouseButton::Right => {
                    self.line_origins
                        .retain(|&o| (o - self.mouse_pos).norm() >= 1.0);
                    self.lines_invalidated = true;
                    true
                }
                MouseButton::Middle
                | MouseButton::Back
                | MouseButton::Forward
                | MouseButton::Other(_) => false,
            },
            _ => false,
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let desired_delta = Duration::from_secs_f32(flow::T_STEP);
        let actual_delta = now.duration_since(self.last_frame_uploaded);
        if self.play && actual_delta >= desired_delta {
            self.next_frame();
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
        if self.current_color_map != self.uploaded_color_map {
            write_color_map_to_texture(
                &self.queue,
                &self.bg_pipeline.color_map_texture,
                COLOR_MAPS[self.current_color_map],
            );
            self.uploaded_color_map = self.current_color_map;
        }
        if self.lines_invalidated {
            let pl = &mut self.path_line_pipeline;
            pl.vertices.clear();
            pl.indices.clear();
            for p in self.line_origins.iter() {
                compute_stream_line(
                    &mut pl.vertices,
                    &mut pl.indices,
                    &self.flow_field,
                    self.current_frame,
                    *p,
                );
            }
            compute_stream_line(
                &mut pl.vertices,
                &mut pl.indices,
                &self.flow_field,
                self.current_frame,
                self.mouse_pos,
            );
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

    fn prev_frame(&mut self) {
        self.current_frame = self
            .current_frame
            .checked_sub(1)
            .unwrap_or(flow::T_CELLS - 1);
        self.lines_invalidated = true;
    }

    fn next_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % flow::T_CELLS;
        self.lines_invalidated = true;
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
                render_pass.set_bind_group(0, &bg.bind_group, &[]);
                render_pass.set_vertex_buffer(0, bg.vertex_buffer.slice(..));
                render_pass.set_index_buffer(bg.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..bg.num_indices, 0, 0..1);
            }

            {
                let pl = &self.path_line_pipeline;
                render_pass.set_pipeline(&pl.render_pipeline);
                render_pass.set_bind_group(0, &pl.bind_group, &[]);
                render_pass.set_vertex_buffer(0, pl.vertex_buffer.slice(..));
                render_pass.set_index_buffer(pl.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
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
    vertices: Vec<ColorVertex>,
    indices: Vec<[u16; 2]>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorVertex {
    position: [f32; 3],
    color: f32,
}

impl ColorVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ColorVertex>() as wgpu::BufferAddress,
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
    current_frame: usize,
    current_color_map: usize,
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
    write_frame_to_texture(queue, &velocity_texture, flow_field.frame(current_frame));
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

    let color_map_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color_map_texture_1"),
        size: COLOR_MAP_TEXTURE_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_color_map_to_texture(queue, &color_map_texture, COLOR_MAPS[current_color_map]);
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

    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout_1"),
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
                    // should match filterable field of the texture
                    ty: wgpu::BindingType::Sampler(match filter {
                        true => wgpu::SamplerBindingType::Filtering,
                        false => wgpu::SamplerBindingType::NonFiltering,
                    }),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // should match filterable field of the texture
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group_1"),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&color_map_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&color_map_sampler),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("texture_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("texture_shader.wgsl"))),
    });
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipline_layout_1"),
        bind_group_layouts: &[&texture_bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline_1"),
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
) -> FieldLinePipeline {
    let color_map_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color_map_texture_1"),
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
        label: Some("bind_group_layout_2"),
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
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group_2"),
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
        label: Some("render_pipeline_2"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[ColorVertex::desc()],
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
    vertices: &mut Vec<ColorVertex>,
    indices: &mut Vec<[u16; 2]>,
    flow_field: &flow::Field,
    current_frame: usize,
    start_pos: flow::Pos2,
) {
    fn flow_pos_to_wgpu_coord(pos: flow::Pos2) -> [f32; 3] {
        [
            2.0 * (pos.x / (flow::X_CELLS - 1) as f32) - 1.0,
            2.0 * (pos.y / (flow::Y_CELLS - 1) as f32) - 1.0,
            0.0,
        ]
    }

    let frame = flow_field.frame(current_frame);
    let first_index = vertices.len();
    let mut empty = true;
    let mut pos = start_pos;
    let step_size = 0.5;
    loop {
        let velocity = bilinear_lookup(&frame, pos);
        if velocity.norm() < f32::EPSILON {
            break;
        }

        pos += velocity * step_size;
        if pos.x < 0.0
            || pos.y < 0.0
            || pos.x > (flow::X_CELLS - 1) as f32
            || pos.y > (flow::Y_CELLS - 1) as f32
        {
            break;
        }

        let color = velocity.norm() / flow_field.max_velocity;
        if empty {
            vertices.push(ColorVertex {
                position: flow_pos_to_wgpu_coord(start_pos),
                color,
            });
            empty = false;
        }
        vertices.push(ColorVertex {
            position: flow_pos_to_wgpu_coord(pos),
            color,
        });
    }

    if first_index < vertices.len() {
        indices.extend((first_index..vertices.len() - 1).map(|i| [i as u16, i as u16 + 1]));
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
