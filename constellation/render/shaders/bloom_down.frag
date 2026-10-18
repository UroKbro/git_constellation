#version 330 core
uniform sampler2D u_scene;
uniform vec2 u_texel;      // 1 / scene size
uniform float u_threshold;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 c = vec3(0.0);
    c += texture(u_scene, v_uv + u_texel * vec2(-1.0, -1.0)).rgb;
    c += texture(u_scene, v_uv + u_texel * vec2( 1.0, -1.0)).rgb;
    c += texture(u_scene, v_uv + u_texel * vec2(-1.0,  1.0)).rgb;
    c += texture(u_scene, v_uv + u_texel * vec2( 1.0,  1.0)).rgb;
    f_color = vec4(max(c * 0.25 - vec3(u_threshold), 0.0), 1.0);
}
