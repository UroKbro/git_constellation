#version 330 core
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_strength;
uniform float u_aspect;
uniform vec3 u_nebula_a;
uniform vec3 u_nebula_b;
uniform float u_nebula_amount;
uniform vec2 u_nebula_offset;
in vec2 v_uv;
out vec4 f_color;

// Value-noise fbm for a faint screen-space nebula behind the galaxy.
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = p * 2.03 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

void main() {
    vec3 hdr = texture(u_scene, v_uv).rgb + texture(u_bloom, v_uv).rgb * u_bloom_strength;
    vec2 p = (v_uv - 0.5) * vec2(u_aspect, 1.0) * 2.2 + u_nebula_offset;
    float warp = fbm(p * 0.8 + 4.0);
    float n = fbm(p + 1.5 * vec2(warp, -warp));
    float mixv = smoothstep(0.25, 0.75, fbm(p * 1.4 + 11.0));
    vec3 neb = mix(u_nebula_a, u_nebula_b, mixv) * smoothstep(0.38, 0.95, n) * u_nebula_amount;
    hdr += neb * 0.35;
    vec3 col = 1.0 - exp(-hdr);
    float vig = smoothstep(0.45, 1.15, length((v_uv - 0.5) * vec2(u_aspect, 1.0)));
    col *= 1.0 - 0.35 * vig;
    f_color = vec4(col, 1.0);
}
