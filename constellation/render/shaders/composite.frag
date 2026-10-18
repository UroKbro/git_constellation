#version 330 core
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_strength;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 hdr = texture(u_scene, v_uv).rgb + texture(u_bloom, v_uv).rgb * u_bloom_strength;
    f_color = vec4(1.0 - exp(-hdr), 1.0);
}
