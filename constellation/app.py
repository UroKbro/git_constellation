"""Galaxy window: event loop, camera input, and the hidden --screenshot mode."""
from __future__ import annotations

import sys

import numpy as np

from constellation.camera import OrbitCamera
from constellation.collapse import plan
from constellation.layout import Layout
from constellation.model import RepoHistory, Window
from constellation.render import gl
from constellation.render.gl import drawable_scale, moderngl, pygame
from constellation.render.renderer import BACKGROUND, Renderer
from constellation.scene import Scene

WINDOW_SIZE = (1280, 800)
FPS = 60
SCREENSHOT_FRAMES = 30
NO_DISPLAY = 3  # exit code when no window/GL context can be opened


def playhead_of(h: RepoHistory, w: Window) -> int:
    return w.end if not w.empty else len(h.commits) - 1


def initial_mask(h: RepoHistory, layout: Layout, w: Window) -> np.ndarray:
    """Placements drawn at the window end: one per file alive there."""
    ph = playhead_of(h, w)
    mask = np.zeros(len(layout.place_file), dtype=bool)
    for fid in np.flatnonzero(h.files_alive_at(ph)):
        mask[layout.place_at(int(fid), ph)] = True
    return mask


def _open(title: str, hidden: bool) -> tuple[pygame.Surface, moderngl.Context] | None:
    pygame.init()
    try:
        if not hidden:
            return gl.open_window(WINDOW_SIZE, title)
        gl.configure_gl_attributes()
        surface = pygame.display.set_mode(
            WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        pygame.display.set_caption(title)
        ctx = moderngl.create_context()
        ctx.enable(moderngl.BLEND)
        return surface, ctx
    except (pygame.error, moderngl.Error) as e:
        print(f"error: cannot open an OpenGL window: {e}", file=sys.stderr)
        pygame.quit()
        return None


def _drawable_size(scale: float) -> tuple[int, int]:
    """Drawable pixels now; ctx.screen.size is fixed at context creation."""
    ww, wh = pygame.display.get_window_size()
    return round(ww * scale), round(wh * scale)


def _save_screen(ctx: moderngl.Context, size: tuple[int, int], path: str) -> None:
    data = ctx.screen.read(viewport=(0, 0, *size), components=3)
    pygame.image.save(pygame.image.frombytes(data, size, "RGB", True), path)
    print(f"Screenshot written to {path} ({size[0]}x{size[1]})")


def run(h: RepoHistory, layout: Layout, title: str, w: Window, *,
        screenshot: str | None = None, at: float = 1.0) -> int:
    """Show the galaxy until closed (or render and save one frame); returns 0."""
    opened = _open(title, hidden=screenshot is not None)
    if opened is None:
        return NO_DISPLAY
    _, ctx = opened
    scale = drawable_scale(ctx, pygame.display.get_window_size())
    renderer = Renderer(ctx, ctx.screen.size)
    mask = initial_mask(h, layout, w)
    scene = Scene(h, layout, plan(h, w), w)  # interim: static frame at the playhead
    ph = playhead_of(h, w)
    planets, stars = scene.planets(ph, 1e9), scene.stars(ph)
    pulses, comets = scene.pulses(1e9), scene.comets(1e9)
    camera = OrbitCamera()
    dw, dh = renderer.size
    bounds = layout.bounds(mask)
    camera.frame(*bounds, aspect=dw / max(dh, 1))
    clock = pygame.time.Clock()
    frames = 0
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    return 0
                if ev.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                    renderer.resize(_drawable_size(scale))
                elif ev.type == pygame.MOUSEMOTION:
                    if ev.buttons[0]:
                        camera.orbit(*ev.rel)
                    elif ev.buttons[2]:
                        camera.pan(*ev.rel, pygame.display.get_window_size()[1])
                elif ev.type == pygame.MOUSEWHEEL:
                    camera.zoom(ev.y)
            dw, dh = renderer.size
            renderer.draw(camera.view(), camera.proj(dw / max(dh, 1)), planets, stars,
                          pulses, comets, 0.0, None, eye=camera.eye(), bounds=bounds)
            frames += 1
            if screenshot is not None and frames >= SCREENSHOT_FRAMES:
                _save_screen(ctx, renderer.size, screenshot)
                return 0
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()


def run_empty(title: str, *, screenshot: str | None = None) -> int:
    """Blank dark window for a repo with no commits (Task 10 adds the message)."""
    opened = _open(title, hidden=screenshot is not None)
    if opened is None:
        return NO_DISPLAY
    _, ctx = opened
    scale = drawable_scale(ctx, pygame.display.get_window_size())
    clock = pygame.time.Clock()
    try:
        while True:
            size = _drawable_size(scale)
            ctx.screen.use()
            ctx.viewport = (0, 0, *size)
            ctx.screen.clear(*BACKGROUND)
            if screenshot is not None:
                _save_screen(ctx, size, screenshot)
                return 0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    return 0
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()
