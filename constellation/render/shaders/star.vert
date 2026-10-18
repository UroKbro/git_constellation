#version 330 core
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_vh;
uniform int u_fog;
uniform vec3 u_eye;
uniform vec3 u_fog_center;
uniform float u_fog_radius;
in vec3 in_pos;
in float in_radius;
in vec3 in_color;
in float in_intensity;       // peak HDR, <= 4, log subtree size
in float in_floor_px;        // min billboard half-size (0 for small subtrees)
out vec2 v_uv;
flat out vec3 v_color;
flat out float v_core;       // core radius as a fraction of the billboard

const float GLOW = 3.0;       // glow half-size in star radii
const float CORE_WORLD = 0.8;  // core radius cap, world units
const float CORE_MIN_PX = 2.0;

void main() {
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    vec3 c = (u_view * vec4(in_pos, 1.0)).xyz;
    float px_per_unit = u_proj[1][1] * 0.5 * u_vh / max(-c.z, 1e-4);
    float w = in_intensity / 4.0;
    float half_size = max(in_radius * GLOW * (0.5 + 0.5 * w), in_floor_px / px_per_unit);
    float fog = 1.0;
    if (u_fog == 1) {
        float near = length(u_eye - u_fog_center) - u_fog_radius;
        fog = 1.0 - 0.35 * clamp((length(c) - near) / (2.0 * u_fog_radius), 0.0, 1.0);
    }
    float core = max(min(CORE_WORLD, in_radius), CORE_MIN_PX / px_per_unit);
    v_core = min(core / half_size, 1.0);
    v_uv = corner;
    v_color = in_color * in_intensity * fog;
    gl_Position = u_proj * vec4(c + vec3(corner * half_size, 0.0), 1.0);
}
