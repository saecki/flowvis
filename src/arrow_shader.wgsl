struct VertexInput {
    @builtin(vertex_index) sub_idx: u32,
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) scalar: f32,
}

@group(0) @binding(0)
var velocity_texture: texture_2d<f32>;
@group(0) @binding(1)
var velocity_sampler: sampler;

@group(0) @binding(8)
var<uniform> transform: mat3x3<f32>;
@group(0) @binding(9)
var<uniform> max_velocity: f32;
@group(0) @binding(10)
var<uniform> step_size: f32;

const base_scale = 0.001;

fn bilinear_lookup(uv: vec2<f32>, values: vec4<f32>) -> f32 {
    let u: f32 = uv.x;
    let v: f32 = uv.y;

    let c01: f32 = values.x;
    let c11: f32 = values.y;
    let c10: f32 = values.z;
    let c00: f32 = values.w;

    // this seems to be the wrong way around, but it works :D
    let c0: f32 = (1.0 - u) * c10 + u * c00;
    let c1: f32 = (1.0 - u) * c11 + u * c01;

    return (1.0 - v) * c0 + v * c1;
}

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    // manual bilinear interpolation, since textureSample is only supported in fragment shaders
    let size: vec2<u32> = textureDimensions(velocity_texture);
    let max_coords = vec2<f32>(f32(size.x - 1), f32(size.y - 1));
    let uv = fract(in.tex_coords * max_coords);
    let velocity = vec2<f32>(
        bilinear_lookup(uv, textureGather(0, velocity_texture, velocity_sampler, in.tex_coords)),
        bilinear_lookup(uv, textureGather(1, velocity_texture, velocity_sampler, in.tex_coords)),
    );
    let dir = normalize(velocity);
    out.scalar = length(velocity) / max_velocity;

    let rot_mat = mat2x2(
        dir.x, dir.y,
        -dir.y, dir.x,
    );
    var pos: vec2<f32>;
    // The sub-index of this arrow vertex:
    // ```
    //          0
    //         / \
    //        /   \
    //       1-3*4-2
    //         |/|
    //         5-6
    // ```
    // `*` is the vertex `position` initially passed to the shader. Depending on the sub_idx the
    // position if offset and rotated using the `velocity` vector.
    // In total 3 triangles are drawn:
    // - [0, 1, 2]
    // - [3, 5, 4]
    // - [4, 5, 6]
    let sub_idx = in.sub_idx % 7;
    let scale = base_scale * step_size;
    switch sub_idx {
        case 0u: {
            pos = in.position + rot_mat * vec2<f32>(2.0 * scale, 0.0);
        }
        case 1u: {
            pos = in.position + rot_mat * vec2<f32>(0.0, scale);
        }
        case 2u: {
            pos = in.position + rot_mat * vec2<f32>(0.0, -scale);
        }
        case 3u: {
            pos = in.position + rot_mat * vec2<f32>(0.0, 0.5 * scale);
        }
        case 4u: {
            pos = in.position + rot_mat * vec2<f32>(0.0, -0.5 * scale);
        }
        case 5u: {
            pos = in.position + rot_mat * vec2<f32>(-2.0 * scale, 0.5 * scale);
        }
        case 6u, default: {
            pos = in.position + rot_mat * vec2<f32>(-2.0 * scale, -0.5 * scale);
        }
    }

    let t_pos = transform * vec3<f32>(pos, 1.0);
    out.clip_position = vec4<f32>(t_pos.xy, 0.0, 1.0);
    return out;
}

@group(0) @binding(2)
var color_map_texture: texture_1d<f32>;
@group(0) @binding(3)
var color_map_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(color_map_texture, color_map_sampler, in.scalar);
}
