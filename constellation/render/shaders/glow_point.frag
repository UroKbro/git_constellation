#version 330 core
flat in vec3 v_color;
out vec4 f_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    f_color = vec4(v_color * exp(-r2 * 3.0), 1.0);
}
