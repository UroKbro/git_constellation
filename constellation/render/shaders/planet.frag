#version 330 core
uniform mat4 u_proj;
uniform int u_fog;
uniform vec3 u_eye;
uniform vec3 u_fog_center;
uniform float u_fog_radius;
in vec3 v_ray;
flat in vec3 v_center;
flat in float v_r;
flat in vec4 v_color;
flat in vec3 v_light;
flat in int v_has_light;
out vec4 f_color;

const float AMBIENT = 0.45;
const float MAX_HDR = 2.0;
const float EMISSIVE = 0.25;

void main() {
    vec3 d = normalize(v_ray);
    float b = dot(d, v_center);
    float h = b * b - dot(v_center, v_center) + v_r * v_r;
    if (h < 0.0) discard;
    float t = max(b - sqrt(h), 1e-4);
    vec3 hit = d * t;
    vec3 n = normalize(hit - v_center);
    vec3 l = v_has_light == 1 ? normalize(v_light - hit) : -d;
    float lambert = max(dot(n, l), 0.0);
    vec3 rgb = v_color.rgb * (AMBIENT + (1.0 - AMBIENT) * lambert)
             + v_color.rgb * v_color.a * EMISSIVE;
    if (u_fog == 1) {
        float near = length(u_eye - u_fog_center) - u_fog_radius;
        float f = clamp((t - near) / (2.0 * u_fog_radius), 0.0, 1.0);
        rgb *= 1.0 - 0.35 * f;
    }
    vec4 clip = u_proj * vec4(hit, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    f_color = vec4(min(rgb, vec3(MAX_HDR)), 1.0);
}
