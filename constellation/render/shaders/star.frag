#version 330 core
uniform float u_falloff_k;
in vec2 v_uv;
flat in vec3 v_color;        // colour x intensity (<= 4)
flat in float v_core;
out vec4 f_color;

const float CORE_W = 0.6;    // core share of the peak; glow takes the rest

float falloff(float d) {     // 1 at d = 0, 0 at d >= 1, smooth (mirrors star_falloff)
    float edge = exp(-u_falloff_k);
    return (exp(-u_falloff_k * min(d * d, 1.0)) - edge) / (1.0 - edge);
}

void main() {
    float d = length(v_uv);
    float f = CORE_W * falloff(d / v_core) + (1.0 - CORE_W) * falloff(d);
    f_color = vec4(v_color * f, 1.0);
}
