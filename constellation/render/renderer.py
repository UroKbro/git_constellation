"""HDR galaxy renderer: starfield, stars, impostor planets, pulses, comets, bloom, overlay."""
from __future__ import annotations

from importlib import resources

import moderngl
import numpy as np

from constellation.scene import CometBatch, PlanetBatch, PulseBatch, StarBatch
from constellation.themes import DEEP_SPACE, Theme

BACKGROUND = (*DEEP_SPACE.background, 1.0)
STARFIELD_N, STARFIELD_SEED = 2000, 7
PLANET_FLOOR_PX, STAR_FLOOR_PX = 2.0, 4.0
BLOOM_THRESHOLD, BLOOM_STRENGTH, BLOOM_PASSES = 1.0, 1.0, 2
COMET_TAIL_N, COMET_TAIL_LEN = 12, 3.0  # points per braid strand, world length
STAR_MAX_INTENSITY, STAR_FALLOFF_K = 4.0, 5.0
# The heaviest stars (root children) sit close together and their glows merge into one
# overexposed blob: damp them when the galaxy has enough stars for that to happen.
HEAVY_STARS, HEAVY_DAMP, HEAVY_MIN_STARS = 5, 0.6, 20
NEAREST_CHUNK = 1 << 22  # planet x star distance entries per numpy chunk
_KEY = np.dtype([("x", "<u4"), ("y", "<u4"), ("z", "<u4")])


def hdr_clear(rgb) -> tuple[float, float, float]:
    """The HDR value that tonemaps (1 - exp(-x)) back to the display colour `rgb`."""
    return tuple(float(-np.log(1.0 - c)) for c in rgb[:3])


def load_shader(name: str) -> str:
    return resources.files("constellation.render").joinpath("shaders", name).read_text("utf-8")


def _set(prog: moderngl.Program, name: str, value) -> None:
    """Set a uniform if the compiler kept it."""
    if name in prog:
        prog[name].value = value


def _f4(*cols: np.ndarray) -> bytes:
    n = len(cols[0])
    if n == 0:
        return b""
    return np.hstack([np.asarray(c, "f4").reshape(n, -1) for c in cols]).tobytes()


def star_alive(radius: np.ndarray) -> np.ndarray:
    """Alive files in each star's subtree, inverted from scene's 0.6 + 0.12*sqrt(alive)."""
    return np.maximum((np.asarray(radius, np.float64) - 0.6) / 0.12, 0.0) ** 2


def star_intensity(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(peak HDR intensity <= 4, min billboard px) per star from its subtree size.

    Intensity is log2(1 + alive) normalised to the batch max, with the 5 heaviest
    stars damped x0.6 once there are >= 20 stars; only stars above the median subtree
    size get the on-screen size floor.
    """
    alive = star_alive(radius)
    if not len(alive):
        return np.zeros(0, "f4"), np.zeros(0, "f4")
    w = np.log2(1.0 + alive)
    w = w / w.max() if w.max() > 0 else np.ones_like(w)
    floor = np.where(alive > np.median(alive), STAR_FLOOR_PX, 0.0)
    if len(w) >= HEAVY_MIN_STARS:
        w[np.argsort(-alive, kind="stable")[:HEAVY_STARS]] *= HEAVY_DAMP
    return (STAR_MAX_INTENSITY * w).astype("f4"), floor.astype("f4")


def star_falloff(d: np.ndarray, k: float = STAR_FALLOFF_K) -> np.ndarray:
    """Billboard profile at normalised distance d: 1 at the centre, 0 at the edge, smooth.

    Mirrors star.frag.
    """
    d = np.clip(np.asarray(d, np.float64), 0.0, 1.0)
    return (np.exp(-k * d * d) - np.exp(-k)) / (1.0 - np.exp(-k))


def nearest_star(pos: np.ndarray, stars: np.ndarray) -> np.ndarray:
    """Index of the nearest star for each position (chunked brute force); -1 if no stars."""
    pos, stars = np.asarray(pos, np.float64), np.asarray(stars, np.float64)
    if not len(stars):
        return np.full(len(pos), -1, np.int64)
    out = np.empty(len(pos), np.int64)
    step = max(1, NEAREST_CHUNK // len(stars))
    s2 = (stars ** 2).sum(axis=1)
    for i in range(0, len(pos), step):
        p = pos[i:i + step]
        d2 = s2[None, :] - 2.0 * p @ stars.T
        out[i:i + step] = d2.argmin(axis=1)
    return out


class _LightCache:
    """Per-position light (xyz + has-light flag), memoised across planet batches."""

    def __init__(self):
        self.reset(None)

    def reset(self, stars_pos) -> None:
        self.stars = None if stars_pos is None else np.asarray(stars_pos, "f4")
        self.keys = np.zeros(0, _KEY)
        self.vals = np.zeros((0, 4), "f4")

    def lights(self, pos: np.ndarray) -> np.ndarray:
        pos = np.ascontiguousarray(pos, "f4").reshape(-1, 3)
        if self.stars is None or not len(self.stars):
            return np.zeros((len(pos), 4), "f4")
        keys = pos.view(_KEY).ravel()
        at = np.minimum(np.searchsorted(self.keys, keys), max(len(self.keys) - 1, 0))
        hit = (self.keys[at] == keys) if len(self.keys) else np.zeros(len(keys), bool)
        out = np.empty((len(pos), 4), "f4")
        out[hit] = self.vals[at[hit]]
        miss = np.flatnonzero(~hit)
        if len(miss):
            new = np.ones((len(miss), 4), "f4")
            new[:, :3] = self.stars[nearest_star(pos[miss], self.stars)]
            out[miss] = new
            mk, first = np.unique(keys[miss], return_index=True)
            keys_all = np.concatenate([self.keys, mk])
            order = np.argsort(keys_all, kind="stable")
            self.keys = keys_all[order]
            self.vals = np.concatenate([self.vals, new[first]])[order]
        return out


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
            self.buf.orphan(max(len(data), self.buf.size))
            self.buf.write(data)

    def release(self) -> None:
        self.vao.release()
        self.buf.release()


class Renderer:
    def __init__(self, ctx: moderngl.Context, size: tuple[int, int],
                 target: moderngl.Framebuffer | None = None):
        self.ctx = ctx
        self._screen = target is None
        self.target = ctx.screen if target is None else target
        self.overlay_uploads = 0
        self.nebula_offset = (0.0, 0.0)  # backdrop parallax, set by the app from the camera
        self.set_theme(DEEP_SPACE)
        self._overlay_src = None
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

        self.planets = _Instances(ctx, self.p_planet, "3f 1f 4f 4f/i",
                                  "in_pos", "in_radius", "in_color", "in_light")
        self.stars = _Instances(ctx, self.p_star, "3f 1f 3f 1f 1f/i", "in_pos", "in_radius",
                                "in_color", "in_intensity", "in_floor_px")
        self._lights = _LightCache()
        self._planet_light = None
        self.pulses = _Instances(ctx, self.p_pulse, "3f 3f 4f 1f 1f",
                                 "in_start", "in_end", "in_color", "in_born", "in_life")
        self.comets = _Instances(ctx, self.p_comet, "3f 4f 4f 1f 1f/i",
                                 "in_pos", "in_color", "in_color2", "in_born", "in_life")
        self._quad = {p: ctx.vertex_array(p, []) for p in
                      (self.p_down, self.p_blur, self.p_composite, self.p_overlay)}
        self._overlay_tex: moderngl.Texture | None = None
        self.resize(size)

    def _program(self, vs: str, fs: str) -> moderngl.Program:
        return self.ctx.program(vertex_shader=load_shader(vs), fragment_shader=load_shader(fs))

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._hdr_bg = hdr_clear(theme.background)

    # ---- targets -----------------------------------------------------------
    @property
    def bloom_scale(self) -> float:
        return self._bloom_scale

    def set_bloom_scale(self, s: float) -> None:
        self._bloom_scale = float(s)
        self._build_bloom()

    def set_target(self, target: moderngl.Framebuffer | None) -> None:
        """Render into `target`, or the (re-read) default framebuffer when None."""
        self._screen = target is None
        self.target = self.ctx.screen if target is None else target

    def resize(self, size: tuple[int, int]) -> None:
        if self._screen:  # the app may have called ctx.detect_framebuffer()
            self.target = self.ctx.screen
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
                comets: CometBatch, planet_light: np.ndarray | None) -> None:
        stars_changed = self.stars.changed(stars)
        if stars_changed:
            self._lights.reset(stars.pos)
            intensity, floor = star_intensity(stars.radius)
            self.stars.write(stars, _f4(stars.pos, stars.radius, stars.color, intensity, floor),
                             len(stars.pos))
            self.uploads += 1
        if (planet_light is not self._planet_light or self.planets.changed(planets)
                or (planet_light is None and stars_changed)):
            self._planet_light = planet_light
            if planet_light is None:
                light = self._lights.lights(planets.pos)
            else:
                light = np.ones((len(planets.pos), 4), "f4")
                light[:, :3] = np.asarray(planet_light, "f4").reshape(-1, 3)
            self.planets.write(planets, _f4(planets.pos, planets.radius, planets.color, light),
                               len(planets.pos))
            self.uploads += 1
        if self.pulses.changed(pulses):
            self.pulses.write(pulses, _f4(pulses.start, pulses.end, pulses.color, pulses.born,
                                          pulses.life), len(pulses.start))
            self.uploads += 1
        if self.comets.changed(comets):
            self.comets.write(comets, _f4(comets.pos, comets.color, comets.color2, comets.born,
                                          comets.life), len(comets.pos))
            self.uploads += 1

    # ---- frame -------------------------------------------------------------
    def draw(self, view: np.ndarray, proj: np.ndarray, planets: PlanetBatch, stars: StarBatch,
             pulses: PulseBatch, comets: CometBatch, now: float,
             overlay: tuple[bytes, tuple[int, int]] | None = None, *, eye=None, bounds=None,
             planet_light: np.ndarray | None = None) -> None:
        """Render one frame into the target.

        `overlay` (RGBA bytes bottom row first, size) is uploaded via `set_overlay`; None
        redraws the last uploaded overlay. `planet_light` (n, 3) is each planet's light
        (its dir star); when None the nearest star is used.
        """
        ctx = self.ctx
        self._upload(planets, stars, pulses, comets, planet_light)
        if overlay is not None:
            self.set_overlay(*overlay)
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
        fbo.clear(*self._hdr_bg, 1.0, depth=1.0)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.disable(moderngl.DEPTH_TEST)
        _set(self.p_starfield, "u_tint", tuple(self.theme.starfield))
        self._sf_vao.render(moderngl.POINTS)

        # opaque impostors first so the additive passes are depth-tested against them
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        if self.planets.n:
            _set(self.p_planet, "u_floor_px", PLANET_FLOOR_PX)
            self.planets.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.planets.n)

        ctx.enable(moderngl.BLEND)
        fbo.depth_mask = False
        if self.stars.n:
            # core hides behind planets; the glow shows through its own planet cloud
            _set(self.p_star, "u_falloff_k", STAR_FALLOFF_K)
            for part in (0, 1):
                _set(self.p_star, "u_part", part)
                (ctx.disable if part else ctx.enable)(moderngl.DEPTH_TEST)
                self.stars.vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                                      instances=self.stars.n)
            ctx.enable(moderngl.DEPTH_TEST)
        if self.pulses.n:
            self.pulses.vao.render(moderngl.POINTS, vertices=self.pulses.n)
        if self.comets.n:
            _set(self.p_comet, "u_tail_n", COMET_TAIL_N)
            _set(self.p_comet, "u_tail_len", COMET_TAIL_LEN)
            centre = (0.0, 0.0, 0.0) if bounds is None else tuple(map(float, bounds[0]))
            _set(self.p_comet, "u_center", centre)
            self.comets.vao.render(moderngl.POINTS, vertices=2 * COMET_TAIL_N,
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
        _set(self.p_composite, "u_bloom_strength", BLOOM_STRENGTH * self.theme.bloom)
        _set(self.p_composite, "u_aspect", self.size[0] / self.size[1])
        _set(self.p_composite, "u_nebula_a", tuple(self.theme.nebula[0]))
        _set(self.p_composite, "u_nebula_b", tuple(self.theme.nebula[1]))
        _set(self.p_composite, "u_nebula_amount", self.theme.nebula_amount)
        _set(self.p_composite, "u_nebula_offset", tuple(map(float, self.nebula_offset)))
        self._quad[self.p_composite].render(moderngl.TRIANGLE_STRIP, vertices=4)
        if self._overlay_tex is not None:
            self._draw_overlay()
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

    def set_overlay(self, data: bytes | None, size: tuple[int, int] | None = None) -> None:
        """Upload the UI overlay (RGBA, bottom row first); None removes it.

        Re-passing the same bytes object is a no-op.
        """
        if data is None:
            if self._overlay_tex is not None:
                self._overlay_tex.release()
            self._overlay_tex = self._overlay_src = None
            return
        if data is self._overlay_src:
            return
        size = (int(size[0]), int(size[1]))
        if self._overlay_tex is None or self._overlay_tex.size != size:
            if self._overlay_tex is not None:
                self._overlay_tex.release()
            self._overlay_tex = self.ctx.texture(size, 4)
            self._overlay_tex.repeat_x = self._overlay_tex.repeat_y = False
        self._overlay_tex.write(data)
        self._overlay_src = data
        self.overlay_uploads += 1

    def _draw_overlay(self) -> None:
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
