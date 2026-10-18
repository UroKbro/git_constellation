#version 330 core
uniform sampler2D u_overlay;
in vec2 v_uv;
out vec4 f_color;

void main() {
    f_color = texture(u_overlay, v_uv);
}
