#version 330 core
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_vh;          // viewport height, drawable px
uniform float u_floor_px;    // minimum on-screen radius
uniform sampler2D u_stars;   // RGBA32F: xyz = star position
uniform int u_n_stars;
in vec3 in_pos;
in float in_radius;
in vec4 in_color;
out vec3 v_ray;              // view-space point on the quad
flat out vec3 v_center;
flat out float v_r;
flat out vec4 v_color;
flat out vec3 v_light;       // view-space light position
flat out int v_has_light;

void main() {
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    vec3 c = (u_view * vec4(in_pos, 1.0)).xyz;
    float depth = max(-c.z, 1e-4);
    float px_per_unit = u_proj[1][1] * 0.5 * u_vh / depth;
    float r = max(in_radius, u_floor_px / px_per_unit);
    float k = 1.0 - min(r * r / dot(c, c), 0.96);
    float half_size = r * min(1.5 / sqrt(k), 8.0);
    vec3 p = c + vec3(corner * half_size, 0.0);

    int best = -1;
    float bd = 1e30;
    for (int i = 0; i < u_n_stars; ++i) {
        vec3 s = texelFetch(u_stars, ivec2(i & 1023, i >> 10), 0).xyz;
        vec3 d = s - in_pos;
        float dd = dot(d, d);
        if (dd < bd) { bd = dd; best = i; }
    }
    v_has_light = best >= 0 && bd > 1e-8 ? 1 : 0;
    vec3 lp = best >= 0 ? texelFetch(u_stars, ivec2(best & 1023, best >> 10), 0).xyz : in_pos;
    v_light = (u_view * vec4(lp, 1.0)).xyz;

    v_ray = p;
    v_center = c;
    v_r = r;
    v_color = in_color;
    gl_Position = u_proj * vec4(p, 1.0);
}
