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
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var<uniform> max_velocity: f32;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let velocity = textureSample(t_diffuse, s_diffuse, in.tex_coords).r;
    let normalized = velocity / max_velocity;
    return srgb_correction(color_map(normalized));
}

const SCALE = vec4<f32>(255.0, 255.0, 255.0, 1.0);
const A = vec4<f32>(180.0, 4.0, 38.0, 1.0) / SCALE;
const B = vec4<f32>(221.0, 221.0, 221.0, 1.0) / SCALE;
const C = vec4<f32>(59.0, 76.0, 192.0, 1.0) / SCALE;
fn color_map(u: f32) -> vec4<f32> {
    if u < 0.5 {
        return mix(A, B, 2.0 * u);
    } else {
        return mix(B, C, 2.0 * u - 1.0);
    }
}

fn srgb_correction(color: vec4<f32>) -> vec4<f32> {
    return pow((color + 0.055) / 1.055, vec4<f32>(2.4, 2.4, 2.4, 1.0));
}
