#version 330 core
uniform vec3 u_tint;
in float v_bright;
out vec4 f_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float a = 1.0 - smoothstep(0.2, 1.0, dot(d, d));
    f_color = vec4(u_tint * v_bright * a, 1.0);
}
