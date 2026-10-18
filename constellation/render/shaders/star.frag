#version 330 core
uniform float u_falloff_k;
uniform int u_part;          // 0 = core (depth-tested), 1 = glow (no depth test)
in vec2 v_uv;
flat in vec3 v_color;        // colour x intensity (<= 4)
flat in float v_core;
flat in float v_weight;      // subtree weight 0..1
out vec4 f_color;

const float CORE_W = 0.6;    // core share of the peak; glow takes the rest

float falloff(float d) {     // 1 at d = 0, 0 at d >= 1, smooth (mirrors star_falloff)
    float edge = exp(-u_falloff_k);
    return (exp(-u_falloff_k * min(d * d, 1.0)) - edge) / (1.0 - edge);
}

void main() {
    float d = length(v_uv);
    // the glow is scaled by subtree weight again so leaf dirs stay faint, not haze
    float f = u_part == 0 ? CORE_W * falloff(d / v_core)
                          : (1.0 - CORE_W) * v_weight * falloff(d);
    if (f <= 0.0) discard;
    f_color = vec4(v_color * f, 1.0);
}
