#version 330 core
in float v_bright;
out vec4 f_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float a = 1.0 - smoothstep(0.2, 1.0, dot(d, d));
    f_color = vec4(vec3(0.82, 0.88, 1.0) * v_bright * a, 1.0);
}
