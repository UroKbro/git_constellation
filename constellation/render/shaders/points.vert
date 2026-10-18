#version 330 core
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_pos;
in vec4 in_color;
in float in_size;
out vec4 v_color;

void main() {
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
    gl_PointSize = max(in_size, 2.0);  // never below 2 drawable px
    v_color = in_color;
}
