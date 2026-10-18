"""HDR galaxy renderer: starfield, stars, impostor planets, pulses, comets, bloom, overlay."""
from __future__ import annotations

from importlib import resources

import moderngl
import numpy as np

from constellation.scene import CometBatch, PlanetBatch, PulseBatch, StarBatch

BACKGROUND = (0.02, 0.02, 0.05, 1.0)
STARFIELD_N, STARFIELD_SEED = 2000, 7
PLANET_FLOOR_PX, STAR_FLOOR_PX = 2.0, 4.0
BLOOM_THRESHOLD, BLOOM_STRENGTH, BLOOM_PASSES = 1.0, 1.0, 2
COMET_TAIL_N, COMET_TAIL_LEN = 12, 3.0
STAR_TEX_W = 1024
# Clear the HDR buffer to the value that tonemaps (1 - exp(-x)) back to BACKGROUND.
_HDR_BG = tuple(float(-np.log(1.0 - c)) for c in BACKGROUND[:3])


def load_shader(name: str) -> str:
    return resources.files("constellation.render").joinpath("shaders", name).read_text("utf-8")


def _set(prog: moderngl.Program, name: str, value) -> None:
    """Set a uniform if the compiler kept it."""
    if name in prog:
        prog[name].value = value


def _f4(*cols: np.ndarray) -> bytes:
    n = len(cols[0])
    arrs = [np.asarray(c, "f4") for c in cols]
    return np.hstack([a.reshape(n, a.size // n if n else (a.shape[1] if a.ndim > 1 else 1))
                      for a in arrs]).astype("f4").tobytes()


class _Instances:
    """A growable per-batch buffer + VAO, rewritten only when the batch object changes."""

    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program, fmt: str, *attrs: str):
        self.buf = ctx.buffer(reserve=64, dynamic=True)
        self.vao = ctx.vertex_array(prog, [(self.buf, fmt, *attrs)])
        self.n = 0
        self.batch = None  # held so id() cannot be recycled while compared

    def changed(self, batch) -> bool:
        return batch is not self.batch

    def write(self, batch, data: bytes, n: int) -> None:
        self.batch, self.n = batch, n
        if n:
            if len(data) > self.buf.size:
                self.buf.orphan(len(data))
            self.buf.write(data)

    def release(self) -> None:
        self.vao.release()
        self.buf.release()


class Renderer:
    def __init__(self, ctx: moderngl.Context, size: tuple[int, int],
                 target: moderngl.Framebuffer | None = None):
        self.ctx = ctx
        self.target = ctx.screen if target is None else target
        self.uploads = 0
        self._bloom_scale = 0.5
        self._fbos: list = []
        prog = self._program
        self.p_starfield = prog("starfield.vert", "starfield.frag")
        self.p_planet = prog("planet.vert", "planet.frag")
        self.p_star = prog("star.vert", "star.frag")
        self.p_pulse = prog("pulse.vert", "glow_point.frag")
        self.p_comet = prog("comet.vert", "glow_point.frag")
        self.p_down = prog("fullscreen.vert", "bloom_down.frag")
        self.p_blur = prog("fullscreen.vert", "blur.frag")
        self.p_composite = prog("fullscreen.vert", "composite.frag")
        self.p_overlay = prog("fullscreen.vert", "overlay.frag")

        rng = np.random.default_rng(STARFIELD_SEED)
        d = rng.normal(size=(STARFIELD_N, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        bright = 0.08 + 0.5 * rng.random(STARFIELD_N) ** 3
        sizes = 1.0 + 1.5 * rng.random(STARFIELD_N)
        self._sf_buf = ctx.buffer(_f4(d, bright, sizes))
        self._sf_vao = ctx.vertex_array(
            self.p_starfield, [(self._sf_buf, "3f 1f 1f", "in_dir", "in_bright", "in_size")])

        self.planets = _Instances(ctx, self.p_planet, "3f 1f 4f/i",
                                  "in_pos", "in_radius", "in_color")
        self.stars = _Instances(ctx, self.p_star, "3f 1f 3f/i", "in_pos", "in_radius", "in_color")
        self.pulses = _Instances(ctx, self.p_pulse, "3f 3f 4f 1f 1f",
                                 "in_start", "in_end", "in_color", "in_born", "in_life")
        self.comets = _Instances(ctx, self.p_comet, "3f 4f 4f 1f 1f/i",
                                 "in_pos", "in_color", "in_color2", "in_born", "in_life")
        self._star_tex = ctx.texture((1, 1), 4, dtype="f4")
        self._star_tex.filter = moderngl.NEAREST, moderngl.NEAREST
        self._n_stars = 0
        self._quad = {p: ctx.vertex_array(p, []) for p in
                      (self.p_down, self.p_blur, self.p_composite, self.p_overlay)}
        self._overlay_tex: moderngl.Texture | None = None
        self.resize(size)

    def _program(self, vs: str, fs: str) -> moderngl.Program:
        return self.ctx.program(vertex_shader=load_shader(vs), fragment_shader=load_shader(fs))

    # ---- targets -----------------------------------------------------------
    @property
    def bloom_scale(self) -> float:
        return self._bloom_scale

    def set_bloom_scale(self, s: float) -> None:
        self._bloom_scale = float(s)
        self._build_bloom()

    def resize(self, size: tuple[int, int]) -> None:
        self.size = (max(1, int(size[0])), max(1, int(size[1])))
        for obj in self._fbos[:3]:
            obj.release()
        ctx = self.ctx
        self._scene_tex = ctx.texture(self.size, 4, dtype="f2")
        self._scene_tex.repeat_x = self._scene_tex.repeat_y = False
        self._depth = ctx.depth_renderbuffer(self.size)
        self._scene_fbo = ctx.framebuffer([self._scene_tex], self._depth)
        self._fbos = [self._scene_fbo, self._scene_tex, self._depth]
        self._build_bloom()

    def _build_bloom(self) -> None:
        for obj in getattr(self, "_bloom_objs", []):
            obj.release()
        w, h = self.size
        self.bloom_size = (max(1, round(w * self._bloom_scale)), max(1, round(h * self._bloom_scale)))
        self._bloom_tex, self._bloom_fbo = [], []
        for _ in range(2):
            t = self.ctx.texture(self.bloom_size, 4, dtype="f2")
            t.repeat_x = t.repeat_y = False
            self._bloom_tex.append(t)
            self._bloom_fbo.append(self.ctx.framebuffer([t]))
        self._bloom_objs = self._bloom_fbo + self._bloom_tex

    # ---- uploads -----------------------------------------------------------
    def _upload(self, planets: PlanetBatch, stars: StarBatch, pulses: PulseBatch,
                comets: CometBatch) -> None:
        if self.planets.changed(planets):
            self.planets.write(planets, _f4(planets.pos, planets.radius, planets.color),
                               len(planets.pos))
            self.uploads += 1
        if self.stars.changed(stars):
            n = len(stars.pos)
            self.stars.write(stars, _f4(stars.pos, stars.radius, stars.color), n)
            self._upload_star_tex(stars.pos)
            self.uploads += 1
        if self.pulses.changed(pulses):
            self.pulses.write(pulses, _f4(pulses.start, pulses.end, pulses.color, pulses.born,
                                          pulses.life), len(pulses.start))
            self.uploads += 1
        if self.comets.changed(comets):
            self.comets.write(comets, _f4(comets.pos, comets.color, comets.color2, comets.born,
                                          comets.life), len(comets.pos))
            self.uploads += 1

    def _upload_star_tex(self, pos: np.ndarray) -> None:
        n = self._n_stars = len(pos)
        rows = max(1, -(-n // STAR_TEX_W))
        data = np.zeros((rows * STAR_TEX_W if n > STAR_TEX_W else max(n, 1), 4), "f4")
        data[:n, :3] = pos
        shape = (STAR_TEX_W, rows) if n > STAR_TEX_W else (max(n, 1), 1)
        if self._star_tex.size != shape:
            self._star_tex.release()
            self._star_tex = self.ctx.texture(shape, 4, dtype="f4")
            self._star_tex.filter = moderngl.NEAREST, moderngl.NEAREST
        self._star_tex.write(data.tobytes())

    # ---- frame -------------------------------------------------------------
    def draw(self, view: np.ndarray, proj: np.ndarray, planets: PlanetBatch, stars: StarBatch,
             pulses: PulseBatch, comets: CometBatch, now: float,
             overlay: tuple[bytes, tuple[int, int]] | None, *, eye=None, bounds=None) -> None:
        """Render one frame into the target; overlay is RGBA bytes, bottom row first."""
        ctx = self.ctx
        self._upload(planets, stars, pulses, comets)
        v = np.asarray(view, "f4").T.tobytes()
        p = np.asarray(proj, "f4").T.tobytes()
        fog = eye is not None and bounds is not None
        common = {"u_vh": float(self.size[1]), "u_now": float(now), "u_fog": int(fog)}
        if fog:
            common.update(u_eye=tuple(map(float, eye)), u_fog_center=tuple(map(float, bounds[0])),
                          u_fog_radius=max(float(bounds[1]), 1e-6))
        for prog in (self.p_starfield, self.p_planet, self.p_star, self.p_pulse, self.p_comet):
            for name, m in (("u_view", v), ("u_proj", p)):
                if name in prog:
                    prog[name].write(m)
            for k, val in common.items():
                _set(prog, k, val)

        fbo = self._scene_fbo
        fbo.use()
        ctx.viewport = (0, 0, *self.size)
        fbo.clear(*_HDR_BG, 1.0, depth=1.0)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.disable(moderngl.DEPTH_TEST)
        self._sf_vao.render(moderngl.POINTS)

        # opaque impostors first so the additive passes are depth-tested against them
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        if self.planets.n:
            self._star_tex.use(0)
            _set(self.p_planet, "u_stars", 0)
            _set(self.p_planet, "u_n_stars", self._n_stars)
            _set(self.p_planet, "u_floor_px", PLANET_FLOOR_PX)
            self.planets.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.planets.n)

        ctx.enable(moderngl.BLEND)
        fbo.depth_mask = False
        if self.stars.n:
            _set(self.p_star, "u_floor_px", STAR_FLOOR_PX)
            self.stars.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.stars.n)
        if self.pulses.n:
            self.pulses.vao.render(moderngl.POINTS, vertices=self.pulses.n)
        if self.comets.n:
            _set(self.p_comet, "u_tail_n", COMET_TAIL_N)
            _set(self.p_comet, "u_tail_len", COMET_TAIL_LEN)
            _set(self.p_comet, "u_center", (0.0, 0.0, 0.0))
            self.comets.vao.render(moderngl.POINTS, vertices=COMET_TAIL_N,
                                   instances=self.comets.n)
        fbo.depth_mask = True
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self._bloom()

        self.target.use()
        ctx.viewport = (0, 0, *self.size)
        self._scene_tex.use(0)
        self._bloom_tex[0].use(1)
        _set(self.p_composite, "u_scene", 0)
        _set(self.p_composite, "u_bloom", 1)
        _set(self.p_composite, "u_bloom_strength", BLOOM_STRENGTH)
        self._quad[self.p_composite].render(moderngl.TRIANGLE_STRIP, vertices=4)
        if overlay is not None:
            self._draw_overlay(*overlay)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _bloom(self) -> None:
        ctx, (bw, bh) = self.ctx, self.bloom_size
        ctx.viewport = (0, 0, bw, bh)
        self._bloom_fbo[0].use()
        self._scene_tex.use(0)
        _set(self.p_down, "u_scene", 0)
        _set(self.p_down, "u_texel", (1.0 / self.size[0], 1.0 / self.size[1]))
        _set(self.p_down, "u_threshold", BLOOM_THRESHOLD)
        self._quad[self.p_down].render(moderngl.TRIANGLE_STRIP, vertices=4)
        _set(self.p_blur, "u_src", 0)
        for _ in range(BLOOM_PASSES):
            for src, dst, step in ((0, 1, (1.0 / bw, 0.0)), (1, 0, (0.0, 1.0 / bh))):
                self._bloom_fbo[dst].use()
                self._bloom_tex[src].use(0)
                _set(self.p_blur, "u_step", step)
                self._quad[self.p_blur].render(moderngl.TRIANGLE_STRIP, vertices=4)

    def _draw_overlay(self, data: bytes, size: tuple[int, int]) -> None:
        size = (int(size[0]), int(size[1]))
        if self._overlay_tex is None or self._overlay_tex.size != size:
            if self._overlay_tex is not None:
                self._overlay_tex.release()
            self._overlay_tex = self.ctx.texture(size, 4)
            self._overlay_tex.repeat_x = self._overlay_tex.repeat_y = False
        self._overlay_tex.write(data)
        self._overlay_tex.use(0)
        _set(self.p_overlay, "u_overlay", 0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._quad[self.p_overlay].render(moderngl.TRIANGLE_STRIP, vertices=4)

    def read_pixels(self) -> np.ndarray:
        """Target contents as (h, w, 4) uint8, top row first."""
        w, h = self.size
        raw = self.target.read(viewport=(0, 0, w, h), components=4)
        return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[::-1].copy()
