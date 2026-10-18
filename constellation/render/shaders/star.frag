#version 330 core
in vec2 v_uv;
flat in vec3 v_color;
out vec4 f_color;

void main() {
    float d = length(v_uv);
    if (d > 1.0) discard;
    float core = 1.0 - smoothstep(0.22, 0.34, d);
    float glow = exp(-d * d * 9.0) * (1.0 - smoothstep(0.7, 1.0, d));
    f_color = vec4(v_color * (core * 5.0 + glow * 1.6), 1.0);
}
