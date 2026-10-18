#version 330 core
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_vh;
uniform float u_now;
uniform int u_fog;
uniform vec3 u_eye;
uniform vec3 u_fog_center;
uniform float u_fog_radius;
in vec3 in_start;
in vec3 in_end;
in vec4 in_color;
in float in_born;
in float in_life;
flat out vec3 v_color;

const float WORLD_SIZE = 0.35;

void main() {
    float t = (u_now - in_born) / max(in_life, 1e-4);
    vec3 c = (u_view * vec4(mix(in_start, in_end, clamp(t, 0.0, 1.0)), 1.0)).xyz;
    float fog = 1.0;
    if (u_fog == 1) {
        float near = length(u_eye - u_fog_center) - u_fog_radius;
        fog = 1.0 - 0.35 * clamp((length(c) - near) / (2.0 * u_fog_radius), 0.0, 1.0);
    }
    float px_per_unit = u_proj[1][1] * 0.5 * u_vh / max(-c.z, 1e-4);
    gl_PointSize = clamp(2.0 * WORLD_SIZE * px_per_unit, 3.0, 14.0);
    float alive = t >= 0.0 && t <= 1.0 ? 1.0 : 0.0;
    float fade = 1.0 - t * t;
    v_color = in_color.rgb * in_color.a * 3.0 * fade * fog;
    gl_Position = alive > 0.5 ? u_proj * vec4(c, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}
