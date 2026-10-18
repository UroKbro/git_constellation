"""Galaxy window: loader-driven startup, timeline playback, input, picking, panel and tooltips.

Pure helpers (input mapping, click detection, camera fitting, picking, bloom switch) come
first and need no window; the GL loop is below them.
"""
from __future__ import annotations

import math
import queue
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from constellation.camera import OrbitCamera
from constellation.details import aggregate_tooltip, commit_details, file_details, tooltip
from constellation.layout import Layout
from constellation.loader import Loaded, start_loading
from constellation.model import RepoHistory, Window
from constellation.palette import lineage_color
from constellation.picking import pick
from constellation.scene import CometBatch, PlanetBatch, PulseBatch, Scene, StarBatch
from constellation.timeline import Timeline

WINDOW_SIZE = (1280, 800)
FPS = 60
NO_DISPLAY = 3  # exit code when no window/GL context can be opened
SCREENSHOT_FRAMES = 30
SCREENSHOT_DT = 1 / 60
SCREENSHOT_LEAD = 512        # commits replayed before --at (2 s at 64x; > every effect's life)
PLAY_SPEED = 64.0            # --screenshot and --bench playback speed
DOUBLE_CLICK_S, DOUBLE_CLICK_PT = 0.35, 4.0
CLICK_SLOP_PT = 4.0          # a press that travels further is a drag, not a click
FILL = 0.85                  # framing: larger projected extent / half view
COMET_PICK_R = 1.5
PICK_MIN_PX = 3.0            # tiny planets stay hoverable
BLOOM_FRAMES, BLOOM_LIMIT_MS = 60, 20.0
OVERLAY_PLAYING_DT = 1 / 20  # overlay redraw cadence while playing
LENS = "1 History"


# ---- pure helpers ----------------------------------------------------------
def alive_mask(h: RepoHistory, layout: Layout, playhead: int,
               in_window: np.ndarray | None = None) -> np.ndarray:
    """Placements drawn at the playhead: one per alive (in-window) file."""
    mask = np.zeros(len(layout.place_file), dtype=bool)
    if not h.commits:
        return mask
    alive = h.files_alive_at(playhead)
    if in_window is not None:
        alive &= in_window
    for fid in np.flatnonzero(alive):
        mask[layout.place_at(int(fid), playhead)] = True
    return mask


def initial_mask(h: RepoHistory, layout: Layout, w: Window) -> np.ndarray:
    """Placements drawn at the window end."""
    return alive_mask(h, layout, w.end if not w.empty else len(h.commits) - 1)


def project(camera: OrbitCamera, points: np.ndarray, aspect: float) -> np.ndarray:
    """(n, 2) NDC xy of world points (in front of the camera)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    m = camera.proj(aspect).astype(np.float64) @ camera.view().astype(np.float64)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    return hom[:, :2] / hom[:, 3:4]


def _extent(camera: OrbitCamera, pts: np.ndarray, aspect: float, region):
    """(AABB centre offset from the region centre in NDC, larger half-extent relative
    to the region's), or None if a point is behind the eye."""
    v = camera.view().astype(np.float64)
    if not np.all(pts @ v[2, :3] + v[2, 3] < -1e-6):
        return None
    xy = project(camera, pts, aspect)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    cx, cy, hx, hy = region
    return (lo + hi) / 2 - (cx, cy), float(((hi - lo) / 2 / (hx, hy)).max())


def fit_camera(camera: OrbitCamera, points: np.ndarray, aspect: float,
               fill: float = FILL, region=(0.0, 0.0, 1.0, 1.0), rounds: int = 4) -> None:
    """Refine a sphere fit: centre the projected AABB of `points` on `region`
    (NDC centre x, y and half-extents; default the whole view) by moving the target
    in the view plane at its depth, and set the distance so the larger half-extent
    is `fill` of the region's (bisection; perspective makes it non-linear)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return
    th = math.tan(math.radians(camera.fov_deg) / 2)
    for _ in range(rounds):
        for _ in range(4):
            e = _extent(camera, pts, aspect, region)
            if e is None or e[1] < 1e-9:
                return
            v = camera.view().astype(np.float64)
            half_h = camera.distance * th
            camera.target = (camera.target + v[0, :3] * e[0][0] * half_h * aspect
                             + v[1, :3] * e[0][1] * half_h)
        toward = float(((pts - camera.target) @ camera.view().astype(np.float64)[2, :3]).max())
        lo, hi = max(1.05 * toward, 2.0), camera.distance

        def ext(d: float) -> float:
            camera.distance = d
            e = _extent(camera, pts, aspect, region)
            return math.inf if e is None else e[1]
        while ext(hi) > fill and hi < 1e7:
            hi *= 2
        if ext(lo) <= fill:
            camera.distance = lo
            continue
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ext(mid) > fill else (lo, mid)
        camera.distance = hi


class ClickDetector:
    """Turns clicks into "click" / "dblclick" (two within 0.35 s and 4 pt)."""

    def __init__(self) -> None:
        self._last: tuple[float, float, float] | None = None

    def feed(self, x: float, y: float, t: float) -> str:
        last = self._last
        if (last is not None and t - last[2] <= DOUBLE_CLICK_S
                and math.hypot(x - last[0], y - last[1]) <= DOUBLE_CLICK_PT):
            self._last = None
            return "dblclick"
        self._last = (x, y, t)
        return "click"


_FASTER, _SLOWER = {"+", "=", "[+]"}, {"-", "[-]"}


class InputState:
    """Maps window-free events to Timeline/OrbitCamera actions.

    Events (window points): ("key", name), ("click", x, y), ("dblclick", x, y),
    ("down", button, x, y, t), ("up", button, x, y, t), ("motion", x, y, dx, dy),
    ("wheel", steps). `handle` returns app-level actions: ("select", x, y) for a
    scene click, ("activate", x, y) for a scene double-click.
    """

    def __init__(self, timeline: Timeline, camera: OrbitCamera,
                 timeline_hit: Callable[[float, float], float | None] | None = None,
                 scrub_x: Callable[[float], float] | None = None,
                 viewport_h: float = WINDOW_SIZE[1]) -> None:
        self.timeline, self.camera = timeline, camera
        self.timeline_hit = timeline_hit or (lambda x, y: None)
        self.scrub_x = scrub_x
        self.viewport_h = viewport_h
        self.panel, self.show_fps, self.quit = True, False, False
        self.mouse: tuple[float, float] | None = None
        self.drag: str | None = None
        self._press: tuple[int, float, float] | None = None
        self._travel = 0.0
        self._clicks = ClickDetector()

    def handle(self, ev: tuple) -> list[tuple]:
        kind = ev[0]
        if kind == "key":
            self._key(ev[1])
        elif kind in ("click", "dblclick"):
            frac = self.timeline_hit(ev[1], ev[2])
            if frac is not None:
                self.timeline.scrub(frac)
            else:
                return [("select" if kind == "click" else "activate", ev[1], ev[2])]
        elif kind == "down":
            _, b, x, y, _t = ev
            self._press, self._travel, self.mouse = (b, x, y), 0.0, (x, y)
            frac = self.timeline_hit(x, y) if b == 1 else None
            if frac is not None:
                self.drag = "scrub"
                self.timeline.scrub(frac)
            else:
                self.drag = {1: "orbit", 3: "pan"}.get(b)
        elif kind == "motion":
            _, x, y, dx, dy = ev
            self.mouse = (x, y)
            self._travel += abs(dx) + abs(dy)
            if self.drag == "scrub":
                frac = self.scrub_x(x) if self.scrub_x else self.timeline_hit(x, y)
                if frac is not None:
                    self.timeline.scrub(frac)
            elif self.drag == "orbit":
                self.camera.orbit(dx, dy)
            elif self.drag == "pan":
                self.camera.pan(dx, dy, self.viewport_h)
        elif kind == "up":
            _, b, x, y, t = ev
            drag, press = self.drag, self._press
            self.drag = self._press = None
            if press is None or press[0] != b or b != 1 or drag == "scrub":
                return []
            moved = max(self._travel, math.hypot(x - press[1], y - press[2]))
            if moved <= CLICK_SLOP_PT:
                return self.handle((self._clicks.feed(x, y, t), x, y))
        elif kind == "wheel":
            self.camera.zoom(ev[1])
        return []

    def _key(self, name: str) -> None:
        tl = self.timeline
        if name == "space":
            tl.toggle()
        elif name == "left":
            tl.step(-1)
        elif name == "right":
            tl.step(1)
        elif name == "home":
            tl.home()
        elif name == "end":
            tl.end()
        elif name in _FASTER:
            tl.faster()
        elif name in _SLOWER:
            tl.slower()
        elif name == "i":
            self.panel = not self.panel
        elif name == "f3":
            self.show_fps = not self.show_fps
        elif name == "escape":
            self.quit = True
        # 1 (the only lens), 2-5, /, A, T, C, F12 belong to later phases: ignored


class BloomSwitch:
    """True exactly once, when the rolling mean work time over 60 frames exceeds 20 ms."""

    def __init__(self, frames: int = BLOOM_FRAMES, limit_ms: float = BLOOM_LIMIT_MS) -> None:
        self._ms: deque[float] = deque(maxlen=frames)
        self._sum = 0.0
        self.limit_ms = limit_ms
        self.switched = False

    def add(self, work_ms: float) -> bool:
        if self.switched:
            return False
        if len(self._ms) == self._ms.maxlen:
            self._sum -= self._ms[0]
        self._ms.append(work_ms)
        self._sum += work_ms
        if len(self._ms) == self._ms.maxlen and self._sum / len(self._ms) > self.limit_ms:
            self.switched = True
            return True
        return False


def _pick(camera: OrbitCamera, px: float, py: float, viewport: tuple[int, int],
          planets: PlanetBatch, comets: CometBatch | None, stars: StarBatch | None = None,
          min_px: float = PICK_MIN_PX) -> tuple[str, int] | None:
    """(kind, batch index): comets first (sphere 1.5), then planets, then stars."""
    dw, dh = viewport
    o, d = camera.ray(px, py, dw, dh)
    if comets is not None and len(comets.pos):
        c = np.asarray(comets.pos, np.float64)
        i = pick(o, d, c, np.full(len(c), COMET_PICK_R), np.ones(len(c), bool))
        if i is not None:
            return "comet", i
    floor = min_px * 2 * camera.distance * math.tan(math.radians(camera.fov_deg) / 2) / dh
    for kind, batch in (("planet", planets), ("star", stars)):
        if batch is None or not len(batch.pos):
            continue
        r = np.asarray(batch.radius, np.float64)
        i = pick(o, d, np.asarray(batch.pos, np.float64), np.maximum(r, floor), r > 0)
        if i is not None:
            return kind, i
    return None


def pick_object(camera: OrbitCamera, px: float, py: float, viewport: tuple[int, int],
                planets: PlanetBatch, comets: CometBatch | None,
                stars: StarBatch | None = None) -> tuple[str, int] | None:
    """("comet", commit index) | ("planet", planet id) | ("star", star index) | None.

    px, py are drawable pixels (window points x scale), origin top-left."""
    hit = _pick(camera, px, py, viewport, planets, comets, stars)
    if hit is None:
        return None
    kind, i = hit
    if kind == "comet":
        return kind, int(comets.commit[i])
    if kind == "planet":
        return kind, int(planets.ids[i])
    return hit


def histogram_colors(h: RepoHistory, w: Window, bins: int) -> np.ndarray | None:
    """Mean lineage colour of the commits in each timeline bin (same binning as Timeline)."""
    if w.empty or bins <= 0:
        return None
    t = np.maximum.accumulate(h.commit_ts)[w.start:w.end + 1]
    span = int(t[-1] - t[0])
    idx = (np.zeros(len(t), np.int64) if span == 0
           else np.minimum((t - t[0]) * bins // span, bins - 1).astype(np.int64))
    lin = h.lineage[w.start:w.end + 1]
    table = np.array([lineage_color(k) for k in range(int(lin.max(initial=0)) + 1)])
    rgb = table[lin]
    cnt = np.maximum(np.bincount(idx, minlength=bins), 1)
    return np.stack([np.bincount(idx, weights=rgb[:, a], minlength=bins) / cnt
                     for a in range(3)], axis=1)


def tag_fractions(h: RepoHistory, timeline: Timeline) -> list[float]:
    w = timeline.window
    if w.empty:
        return []
    index = {c.sha: i for i, c in enumerate(h.commits[w.start:w.end + 1], w.start)}
    return [timeline.fraction_of(index[t.sha]) for t in h.tags if t.sha in index]


class DetailCache:
    """Panel/tooltip text, recomputed only when its (object, playhead) key changes."""

    def __init__(self, loaded: Loaded) -> None:
        self.d = loaded
        self._tip: tuple | None = None
        self._panel: tuple | None = None

    def tooltip(self, hit: tuple[str, int], playhead: int) -> str:
        key = (hit, playhead)
        if self._tip is None or self._tip[0] != key:
            d, (kind, i) = self.d, hit
            if kind == "comet":
                c = d.h.commits[i]
                text = f"{c.sha[:10]} · {c.subject}"
            elif i >= 0:
                text = tooltip(d.h, playhead, i)
            else:
                text = aggregate_tooltip(d.h, d.plan, -i - 1, playhead)
            self._tip = (key, text)
        return self._tip[1]

    def panel(self, selection: tuple[str, int] | None, playhead: int,
              stale_ok: bool = False) -> list[str]:
        """Lines for the selected object, or the playhead commit. With `stale_ok`
        (during playback) a selected file keeps its last lines."""
        key = (selection, playhead)
        if self._panel is not None and (self._panel[0] == key or (
                stale_ok and selection is not None and selection[0] == "planet"
                and self._panel[0][0] == selection)):
            return self._panel[1]
        d, h, w = self.d, self.d.h, self.d.window
        if selection is None:
            if w.empty:
                lines = [d.root.name, "No commits in the selected range"]
            else:
                lines = [d.root.name, f"Commit {playhead - w.start + 1:,} of {w.size:,}", "",
                         *commit_details(h, playhead)]
        elif selection[0] == "comet":
            lines = commit_details(h, selection[1])
        elif selection[1] >= 0:
            lines = file_details(h, w, playhead, selection[1], d.coupling)
        else:
            lines = [aggregate_tooltip(h, d.plan, -selection[1] - 1, playhead)]
        self._panel = (key, lines)
        return lines


def _empty_batches() -> tuple[PlanetBatch, StarBatch, PulseBatch, CometBatch]:
    z3, z4, z = np.zeros((0, 3), "f4"), np.zeros((0, 4), "f4"), np.zeros(0, "f4")
    return (PlanetBatch(z3, z, z4, np.zeros(0, np.int32)), StarBatch(z3, z, z3),
            PulseBatch(z3, z3, z4, z, z), CometBatch(z3, z4, z4, np.zeros(0, np.int32), z, z))


# ---- GL loop ---------------------------------------------------------------
from constellation.render import gl  # noqa: E402
from constellation.render.gl import moderngl, pygame  # noqa: E402
from constellation.render.renderer import Renderer  # noqa: E402
from constellation.render.ui import BAR_H, BAR_PAD, CONTROLS_W, PANEL_W, Overlay  # noqa: E402

_KEYS = {pygame.K_SPACE: "space", pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
         pygame.K_HOME: "home", pygame.K_END: "end", pygame.K_PLUS: "+",
         pygame.K_EQUALS: "+", pygame.K_KP_PLUS: "+", pygame.K_MINUS: "-",
         pygame.K_KP_MINUS: "-", pygame.K_i: "i", pygame.K_F3: "f3",
         pygame.K_ESCAPE: "escape"}
_RESIZE_EVENTS = {pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED, pygame.WINDOWDISPLAYCHANGED}


class _Window:
    """Window, context, renderer and overlay, kept at the current drawable size."""

    def __init__(self, title: str, hidden: bool, display: int | None, vsync: bool | None):
        pygame.init()
        self.window, self.ctx = gl.open_window(WINDOW_SIZE, title, hidden=hidden, display=display or 0,
                                     vsync=vsync)
        size = gl.drawable_size() or self.ctx.screen.size
        self.scale = size[0] / max(self.window.size[0], 1)
        self.renderer = Renderer(self.ctx, size)
        self.overlay = Overlay(self.renderer.size, self.scale)
        self.resized = True

    def sync_size(self) -> None:
        """Re-derive the drawable size and scale (after resize / display change)."""
        self.ctx.detect_framebuffer()  # moderngl caches the default framebuffer
        ww, wh = self.window.size
        size = gl.drawable_size() or (round(ww * self.scale), round(wh * self.scale))
        scale = size[0] / max(ww, 1)
        if tuple(size) != self.renderer.size:
            self.renderer.resize(size)
        if (tuple(size), scale) != (self.overlay.size, self.overlay.scale):
            self.overlay.resize(size, scale)
            self.resized = True
        self.scale = scale

    @property
    def aspect(self) -> float:
        dw, dh = self.renderer.size
        return dw / max(dh, 1)

    def save(self, path: str) -> None:
        px = self.renderer.read_pixels()[:, :, :3]
        h, w = px.shape[:2]
        pygame.image.save(pygame.image.frombytes(px.tobytes(), (w, h), "RGB"), path)
        print(f"Screenshot written to {path} ({w}x{h})")

    def upload_overlay(self) -> None:
        self.renderer.set_overlay(*self.overlay.to_bytes())


class _Galaxy:
    """Loaded repo state plus one frame of update + draw."""

    def __init__(self, win: _Window, loaded: Loaded):
        self.win, self.d = win, loaded
        h, w = loaded.h, loaded.window
        self.scene = Scene(h, loaded.layout, loaded.plan, w)
        self.timeline = Timeline(w, h.commit_ts)
        self.camera = OrbitCamera()
        self.input = InputState(self.timeline, self.camera, self._timeline_hit, self._scrub_x,
                                WINDOW_SIZE[1])
        self.details = DetailCache(loaded)
        self.selection: tuple[str, int] | None = None
        self.hover: tuple[str, int] | None = None
        self._hover_key = None
        self._tags = tag_fractions(h, self.timeline)
        self._hist_bins = -1
        self._hist = self._hist_rgb = None
        self._overlay_key = None
        self._overlay_t = -math.inf
        self._planets = self._planets_key = self._light = None
        self._stars = self._stars_key = None
        self._pulses = self._comets = None
        self.fps: float | None = None
        self.bounds = None
        self.frame_view()

    # framing
    def frame_view(self) -> None:
        d, ph = self.d, self.timeline.playhead
        mask = alive_mask(d.h, d.layout, ph, self.scene.in_window)
        self.bounds = d.layout.bounds(mask)
        aspect = self.win.aspect
        self.camera.frame(*self.bounds, aspect=aspect)
        fit_camera(self.camera, d.layout.place_pos[mask], aspect, region=self._scene_region())

    def _scene_region(self) -> tuple[float, float, float, float]:
        """NDC centre and half-extents of the view left of the panel, above the bar."""
        w, h = self.win.overlay.size_pt
        right = 1 - 2 * PANEL_W / w if self.input.panel and w > 2 * PANEL_W else 1.0
        bottom = -1 + 2 * BAR_H / h if h > 2 * BAR_H else -1.0
        return (right - 1) / 2, (1 + bottom) / 2, (right + 1) / 2, (1 - bottom) / 2

    # overlay geometry (window points)
    def _timeline_hit(self, x: float, y: float) -> float | None:
        return self.win.overlay.timeline_hit(x, y)

    def _scrub_x(self, x: float) -> float:
        x0, x1 = CONTROLS_W, self.win.overlay.size_pt[0] - BAR_PAD
        return min(max((x - x0) / max(x1 - x0, 1e-9), 0.0), 1.0)

    def _in_scene(self, x: float, y: float) -> bool:
        w, h = self.win.overlay.size_pt
        return y < h - BAR_H and not (self.input.panel and x >= w - PANEL_W)

    # events
    def handle(self, ev: tuple, now: float) -> None:
        self.input.viewport_h = self.win.overlay.size_pt[1]
        for action, x, y in self.input.handle(ev):
            if not self._in_scene(x, y):
                continue
            px, py = x * self.win.scale, y * self.win.scale
            if action == "select":
                self.selection = pick_object(self.camera, px, py, self.win.renderer.size,
                                             self._planets, self._comets)
                if self.selection is not None and self.selection[0] == "star":
                    self.selection = None
            else:
                self._activate(px, py, now)

    def _activate(self, px: float, py: float, now: float) -> None:
        hit = _pick(self.camera, px, py, self.win.renderer.size, self._planets, None,
                    self._stars)
        if hit is None:
            return
        kind, i = hit
        batch = self._planets if kind == "planet" else self._stars
        if kind == "planet" and batch.ids[i] < 0:
            self.d.plan.expand(int(-batch.ids[i] - 1))
            self.selection = None
            return
        r = float(batch.radius[i])
        self.camera.fly_to(batch.pos[i], max(4 * r, 6.0), now)

    # per-frame batches
    def _light_for(self, ids: np.ndarray, playhead: int) -> np.ndarray:
        """Each planet's own directory star; aggregates are lit by their parent dir."""
        lay, plan, h = self.d.layout, self.d.plan, self.d.h
        out = np.zeros((len(ids), 3), "f4")
        f = ids >= 0
        if f.any():
            places = self.scene._place(ids[f].astype(np.int64), playhead)
            out[f] = lay.dir_pos[lay.place_dir[places]]
        if (~f).any():
            gdir = np.asarray(plan.group_dir, np.int64)[-ids[~f].astype(np.int64) - 1]
            parent = np.asarray(h.dir_parent, np.int64)[gdir]
            out[~f] = lay.dir_pos[np.where(parent >= 0, parent, gdir)]
        return out

    def _batches(self, ph: int, now: float):
        sc = self.scene
        key = (ph, frozenset(self.d.plan.expanded))
        # reuse the planet batch (no re-upload) once no add/delete/rename animation runs
        if self._planets is None or key != self._planets_key or now < sc._anim_until:
            self._planets = sc.planets(ph, now)
            lkey = (key, len(self._planets.ids))
            if self._light is None or self._light[0] != lkey:
                self._light = (lkey, self._light_for(self._planets.ids, ph))
            self._planets_key = key
        if key != self._stars_key:
            self._stars, self._stars_key = sc.stars(ph), key
        pulses, comets = sc.pulses(now), sc.comets(now)
        if self._pulses is None or len(pulses.born) or len(self._pulses.born):
            self._pulses = pulses
        if self._comets is None or len(comets.born) or len(self._comets.born):
            self._comets = comets

    def _update_hover(self, ph: int) -> None:
        m = self.input.mouse
        cam = self.camera
        key = (m, ph, tuple(cam.target), cam.distance, cam.yaw, cam.pitch,
               len(self._planets.ids), len(self._comets.born), self.input.panel)
        if key == self._hover_key:
            return
        self._hover_key = key
        if m is None or self.input.drag is not None or not self._in_scene(*m):
            self.hover = None
            return
        s = self.win.scale
        hit = pick_object(cam, m[0] * s, m[1] * s, self.win.renderer.size,
                          self._planets, self._comets)
        self.hover = hit if hit is not None and hit[0] != "star" else None

    def _draw_overlay(self, ph: int, now: float) -> None:
        tl, ov, win = self.timeline, self.win.overlay, self.win
        playing = tl.playing
        if playing and not win.resized and now - self._overlay_t < OVERLAY_PLAYING_DT:
            return
        self._update_hover(ph)
        track_pt = ov.size_pt[0] - CONTROLS_W - BAR_PAD
        bins = max(1, int(track_pt / 3))
        if bins != self._hist_bins:
            self._hist_bins = bins
            self._hist = tl.histogram(bins)
            self._hist_rgb = histogram_colors(self.d.h, self.d.window, bins)
        frac = tl.fraction_of(ph)
        panel = (self.details.panel(self.selection, ph, stale_ok=playing)
                 if self.input.panel else None)
        tip = None
        if self.hover is not None and self.input.mouse is not None:
            tip = (self.details.tooltip(self.hover, ph), *self.input.mouse)
        fps = round(self.fps) if self.input.show_fps and self.fps is not None else None
        key = (ov.size, ov.scale, round(frac * track_pt * ov.scale), playing, tl.speed,
               id(panel), tip, fps)
        self._overlay_t = now
        if key == self._overlay_key and not win.resized:
            return
        self._overlay_key = key
        win.resized = False
        ov.clear()
        ov.draw_lens(LENS)
        ov.draw_timeline(self._hist, frac, playing, tl.speed, self._tags, self._hist_rgb)
        ov.draw_panel(panel or [], panel is not None)
        ov.draw_fps(fps)  # after the panel so F3 stays visible over it
        if tip is not None:
            ov.draw_tooltip(*tip)
        win.upload_overlay()

    def frame(self, prev: int, now: float, dt: float) -> None:
        """Advance the clock by dt (the playhead was `prev` before this frame's input) and draw."""
        tl = self.timeline
        tl.update(dt)
        ph = tl.playhead
        self.scene.advance(prev, ph, now)
        self.camera.update(now)
        self._batches(ph, now)
        self._draw_overlay(ph, now)
        cam, r = self.camera, self.win.renderer
        r.draw(cam.view(), cam.proj(self.win.aspect), self._planets, self._stars,
               self._pulses, self._comets, now, None, eye=cam.eye(), bounds=self.bounds,
               planet_light=self._light[1])


def _events(win: _Window) -> list:
    """Pending pygame events as pure events; resizes are applied once per batch."""
    out, resize = [], False
    now = time.perf_counter()
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            out.append(("key", "escape"))
        elif ev.type in _RESIZE_EVENTS:
            resize = True
        elif ev.type == pygame.KEYDOWN and ev.key in _KEYS:
            out.append(("key", _KEYS[ev.key]))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
            out.append(("down", ev.button, *ev.pos, now))
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button in (1, 3):
            out.append(("up", ev.button, *ev.pos, now))
        elif ev.type == pygame.MOUSEMOTION:
            out.append(("motion", *ev.pos, *ev.rel))
        elif ev.type == pygame.MOUSEWHEEL:
            out.append(("wheel", ev.precise_y))
    if resize:
        win.sync_size()
    return out


def _load(win: _Window, q: queue.Queue, interactive: bool) -> tuple[str, object]:
    """Show progress until the loader finishes: ("done", Loaded | None), ("error", msg)
    or ("quit", None)."""
    cam = OrbitCamera()
    empty = _empty_batches()
    text, frac, start = "Reading history…", None, time.perf_counter()
    clock = pygame.time.Clock()
    drawn = None
    while True:
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "progress":
                    text, frac = msg[1], msg[2]
                else:
                    return msg[0], msg[1]
        except queue.Empty:
            pass
        for ev in _events(win):
            if ev == ("key", "escape"):
                return "quit", None
        if drawn != (text, frac) or win.resized:
            ov = win.overlay
            ov.clear()
            ov.draw_progress(text, frac)
            win.upload_overlay()
            drawn, win.resized = (text, frac), False
        win.renderer.draw(cam.view(), cam.proj(win.aspect), *empty,
                          time.perf_counter() - start, None)
        if interactive:
            win.window.flip()
        clock.tick(FPS)


def _message(win: _Window, text: str, screenshot: str | None) -> int:
    """Centred message until closed (empty repo)."""
    cam, empty, clock = OrbitCamera(), _empty_batches(), pygame.time.Clock()
    start = time.perf_counter()
    while True:
        if win.resized:
            win.overlay.clear()
            win.overlay.draw_lens(LENS)
            win.overlay.draw_message(text)
            win.upload_overlay()
            win.resized = False
        win.renderer.draw(cam.view(), cam.proj(win.aspect), *empty,
                          time.perf_counter() - start, None)
        if screenshot is not None:
            win.save(screenshot)
            return 0
        if ("key", "escape") in _events(win):
            return 0
        win.window.flip()
        clock.tick(FPS)


def _interactive(g: _Galaxy, start: float) -> int:
    win, clock = g.win, pygame.time.Clock()
    bloom = BloomSwitch()
    walls: deque[float] = deque(maxlen=30)
    last = time.perf_counter() - start
    fps_t = 0.0
    while True:
        t0 = time.perf_counter()
        now = t0 - start
        dt, last = now - last, now
        walls.append(dt)
        if now - fps_t >= 0.5 and sum(walls) > 0:
            g.fps, fps_t = len(walls) / sum(walls), now
        prev = g.timeline.playhead
        for ev in _events(win):
            g.handle(ev, now)
        if g.input.quit:
            return 0
        g.frame(prev, now, dt)
        win.ctx.finish()
        if bloom.add((time.perf_counter() - t0) * 1000):
            win.renderer.set_bloom_scale(0.25)
        win.window.flip()
        clock.tick(FPS)


def _screenshot(g: _Galaxy, path: str, at: float) -> int:
    """Replay up to the --at fraction at 64x on a fixed 60 Hz clock, settle, save.

    Commits more than SCREENSHOT_LEAD before the target are skipped with one jump:
    every pulse, comet and animation they would spawn has expired by then."""
    tl = g.timeline
    if not tl.window.empty:
        tl.scrub(at)
        target = tl.playhead
        tl.home()
        tl.step(max(0, target - SCREENSHOT_LEAD - tl.playhead))
        tl.speed, tl.playing = PLAY_SPEED, tl.playhead < target
    else:
        target = tl.playhead
    now, settled = 0.0, 0
    while settled < SCREENSHOT_FRAMES:
        now += SCREENSHOT_DT
        prev = tl.playhead
        if tl.playing and tl.cursor + SCREENSHOT_DT * tl.speed * 4.0 >= target:
            tl.step(target - prev)  # land exactly on the target (pauses), then settle
        if not tl.playing:
            settled += 1
        g.frame(prev, now, SCREENSHOT_DT)
    print(f"playhead {tl.playhead} · {len(g._pulses.born)} pulses · "
          f"{len(g._comets.born)} comets", file=sys.stderr)
    g.win.save(path)
    return 0


def _bench(g: _Galaxy, seconds: float, start: float) -> int:
    win, tl = g.win, g.timeline
    for name, scale in (("half", 0.5), ("quarter", 0.25)):
        win.renderer.set_bloom_scale(scale)
        tl.home()
        tl.speed, tl.playing = PLAY_SPEED, True
        work, wall = [], []
        t_end = time.perf_counter() + seconds
        last = None
        while True:
            t0 = time.perf_counter()
            if t0 >= t_end:
                break
            now = t0 - start
            if last is not None:
                wall.append((t0 - last) * 1000)
            dt = 0.0 if last is None else t0 - last
            last = t0
            prev = tl.playhead
            for ev in _events(win):
                if ev == ("key", "escape"):
                    return 0
            if not tl.playing:  # reached the end: replay
                tl.home()
                tl.playing = True
            g.frame(prev, now, dt)
            win.ctx.finish()
            work.append((time.perf_counter() - t0) * 1000)
            win.window.flip()
        w, f = np.array(work[1:] or [0.0]), np.array(wall[1:] or [0.0])
        print(f"bench {name}: work mean {w.mean():.1f} p95 {np.percentile(w, 95):.1f} ms · "
              f"frame mean {f.mean():.1f} p95 {np.percentile(f, 95):.1f} ms")
    return 0


def run(root: Path, ref: str, since: int | None, until: int | None, use_cache: bool,
        title: str, *, bench: float | None = None, screenshot: str | None = None,
        at: float = 1.0, display: int | None = None) -> int:
    """Open the window, load in the background with progress, then show the galaxy
    at the window end, paused. Returns 0, 1 (load error) or 3 (no display)."""
    start = time.perf_counter()
    try:
        win = _Window(title, hidden=screenshot is not None, display=display,
                      vsync=False if bench is not None else None)
    except (pygame.error, moderngl.Error) as e:
        print(f"error: cannot open an OpenGL window: {e}", file=sys.stderr)
        pygame.quit()
        return NO_DISPLAY
    try:
        if screenshot is not None or bench is not None:
            dw, dh = win.renderer.size
            print(f"drawable {dw}x{dh} (scale {win.scale:g})", file=sys.stderr)
        q = start_loading(root, ref, since, until, use_cache)
        kind, value = _load(win, q, interactive=screenshot is None)
        if kind == "quit":
            return 0
        if kind == "error":
            print(value, file=sys.stderr)
            return 1
        if value is None:
            return _message(win, "No commits yet", screenshot)
        g = _Galaxy(win, value)
        if screenshot is not None:
            return _screenshot(g, screenshot, at)
        if bench is not None:
            return _bench(g, bench, start)
        return _interactive(g, start)
    finally:
        pygame.quit()
