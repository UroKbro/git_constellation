#version 330 core
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_vh;
uniform float u_now;
uniform int u_tail_n;
uniform float u_tail_len;
uniform vec3 u_center;
uniform int u_fog;
uniform vec3 u_eye;
uniform vec3 u_fog_center;
uniform float u_fog_radius;
in vec3 in_pos;
in vec4 in_color;
in vec4 in_color2;
in float in_born;
in float in_life;
flat out vec3 v_color;

const float HEAD_SIZE = 0.8;

const float BRAID_TURNS = 1.5;

void main() {
    int strand = gl_VertexID & 1;                // braid strand: 0 = color, 1 = color2
    int k = gl_VertexID >> 1;                    // 0 = head, then along the tail
    float s = float(k) / float(max(u_tail_n - 1, 1));
    vec3 out_dir = in_pos - u_center;
    out_dir = dot(out_dir, out_dir) > 1e-8 ? normalize(out_dir) : vec3(0.0, 1.0, 0.0);
    vec3 side = cross(out_dir, abs(out_dir.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0));
    side = normalize(side);
    float phase = 6.2831853 * BRAID_TURNS * s + 3.1415927 * float(strand);
    vec3 world = in_pos + out_dir * u_tail_len * s + side * sin(phase) * 0.12 * u_tail_len * s;
    vec3 c = (u_view * vec4(world, 1.0)).xyz;
    float t = (u_now - in_born) / max(in_life, 1e-4);
    float fog = 1.0;
    if (u_fog == 1) {
        float near = length(u_eye - u_fog_center) - u_fog_radius;
        fog = 1.0 - 0.35 * clamp((length(c) - near) / (2.0 * u_fog_radius), 0.0, 1.0);
    }
    float px_per_unit = u_proj[1][1] * 0.5 * u_vh / max(-c.z, 1e-4);
    float size = k == 0 ? 1.0 : mix(0.45, 0.2, s);
    gl_PointSize = clamp(2.0 * HEAD_SIZE * size * px_per_unit, k == 0 ? 6.0 : 2.0, 28.0);
    vec4 col = strand == 1 ? in_color2 : in_color;
    float bright = k == 0 ? 2.0 : 1.8 * (1.0 - s);  // two head points overlap -> ~4
    float fade = 1.0 - clamp(t, 0.0, 1.0);
    v_color = col.rgb * col.a * bright * fade * fog;
    bool alive = t >= 0.0 && t <= 1.0;
    gl_Position = alive ? u_proj * vec4(c, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}
