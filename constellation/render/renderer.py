"""Point renderer: draws file dots into the screen (or a given framebuffer)."""
from __future__ import annotations

from importlib import resources

import moderngl
import numpy as np

BACKGROUND = (0.02, 0.02, 0.05, 1.0)


def load_shader(name: str) -> str:
    return resources.files("constellation.render").joinpath("shaders", name).read_text("utf-8")


class Renderer:
    def __init__(self, ctx: moderngl.Context, size: tuple[int, int],
                 target: moderngl.Framebuffer | None = None):
        self.ctx = ctx
        self.target = ctx.screen if target is None else target
        self.size = tuple(size)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.prog = ctx.program(vertex_shader=load_shader("points.vert"),
                                fragment_shader=load_shader("points.frag"))
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None
        self._n = 0

    def resize(self, size: tuple[int, int]) -> None:
        self.size = tuple(size)

    def set_points(self, pos: np.ndarray, color: np.ndarray, size_px: np.ndarray) -> None:
        n = len(pos)
        data = np.empty((n, 8), dtype="f4")
        data[:, 0:3], data[:, 3:7], data[:, 7] = pos, color, size_px
        if self._vao is not None:
            self._vao.release()
            self._vbo.release()
            self._vao = self._vbo = None
        self._n = n
        if n:
            self._vbo = self.ctx.buffer(data.tobytes())
            self._vao = self.ctx.vertex_array(
                self.prog, [(self._vbo, "3f 4f 1f", "in_pos", "in_color", "in_size")])

    def render(self, view: np.ndarray, proj: np.ndarray) -> None:
        self.target.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.target.clear(*BACKGROUND)
        if not self._n:
            return
        self.prog["u_view"].write(np.asarray(view, "f4").T.tobytes())
        self.prog["u_proj"].write(np.asarray(proj, "f4").T.tobytes())
        self._vao.render(moderngl.POINTS)

    def read_pixels(self) -> np.ndarray:
        """Target contents as (h, w, 4) uint8, top row first."""
        w, h = self.size
        raw = self.target.read(viewport=(0, 0, w, h), components=4)
        return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[::-1].copy()
