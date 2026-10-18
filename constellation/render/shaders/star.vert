#version 330 core
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_vh;
uniform float u_floor_px;
uniform int u_fog;
uniform vec3 u_eye;
uniform vec3 u_fog_center;
uniform float u_fog_radius;
in vec3 in_pos;
in float in_radius;
in vec3 in_color;
out vec2 v_uv;
flat out vec3 v_color;

const float GLOW = 3.0;  // billboard half-size in star radii

void main() {
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    vec3 c = (u_view * vec4(in_pos, 1.0)).xyz;
    float px_per_unit = u_proj[1][1] * 0.5 * u_vh / max(-c.z, 1e-4);
    float half_size = max(in_radius * GLOW, u_floor_px / px_per_unit);
    float fog = 1.0;
    if (u_fog == 1) {
        float near = length(u_eye - u_fog_center) - u_fog_radius;
        fog = 1.0 - 0.35 * clamp((length(c) - near) / (2.0 * u_fog_radius), 0.0, 1.0);
    }
    v_uv = corner;
    v_color = in_color * fog;
    gl_Position = u_proj * vec4(c + vec3(corner * half_size, 0.0), 1.0);
}
