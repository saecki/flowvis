struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct FragmentOpts {
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

@group(0) @binding(0)
var velocity_texture: texture_2d<f32>;
@group(0) @binding(1)
var velocity_sampler: sampler;
@group(0) @binding(2)
var<uniform> max_velocity: f32;
@group(0) @binding(3)
var color_map_texture: texture_1d<f32>;
@group(0) @binding(4)
var color_map_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let velocity = textureSample(velocity_texture, velocity_sampler, in.tex_coords).r;
    let normalized = velocity / max_velocity;
    let color = textureSample(color_map_texture, color_map_sampler, normalized);
    return color;
}
