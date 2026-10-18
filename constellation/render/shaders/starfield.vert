#version 330 core
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_dir;
in float in_bright;
in float in_size;
out float v_bright;

void main() {
    vec4 c = u_proj * vec4(mat3(u_view) * in_dir, 0.0);  // direction at infinity
    gl_Position = vec4(c.xy, c.w * 0.99999, c.w);
    gl_PointSize = in_size;
    v_bright = in_bright;
}
