struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) scalar: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) scalar: f32,
}

@group(0) @binding(8)
var<uniform> transform: mat3x3<f32>;

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.scalar = in.scalar;
    let pos = transform * vec3<f32>(in.position, 1.0);
    out.clip_position = vec4<f32>(pos.xy, 0.0, 1.0);
    return out;
}

@group(0) @binding(0)
var color_map_texture: texture_1d<f32>;
@group(0) @binding(1)
var color_map_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(color_map_texture, color_map_sampler, in.scalar);
}
