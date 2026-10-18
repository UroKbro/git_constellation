#version 330 core
in vec4 v_color;
out vec4 f_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float a = 1.0 - smoothstep(0.6, 1.0, r2);
    f_color = vec4(v_color.rgb, v_color.a * a);
}
