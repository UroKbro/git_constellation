"""2D UI overlay drawn with pygame onto an RGBA surface the renderer uploads as a texture.

Layout constants are in window points; drawing multiplies by `scale` (drawable px per point).
"""
from __future__ import annotations

import numpy as np
import pygame

BAR_H = 64
CONTROLS_W = 110  # left of the bar: play icon + speed text
BAR_PAD = 16
PANEL_W = 360
TOOLTIP_MAX_W = 600
HELP_LINE = "Space play · ←/→ step · Home/End · I panel · F3 fps"

TEXT = (230, 232, 240, 255)
DIM = (150, 155, 170, 255)
BAR_BG = (14, 16, 24, 200)
PANEL_BG = (14, 16, 24, 215)
HIST_DEFAULT = (110, 150, 230)
ACCENT = (255, 214, 102, 255)


class Overlay:
    def __init__(self, size: tuple[int, int], scale: float) -> None:
        pygame.font.init()
        self._fonts: dict[tuple[str, int], pygame.font.Font] = {}
        self.resize(size, scale)

    def resize(self, size: tuple[int, int], scale: float) -> None:
        self.size = (int(size[0]), int(size[1]))
        self.scale = float(scale)
        self.surface = pygame.Surface(self.size, pygame.SRCALPHA)
        self._fonts = {}
        self.dirty = True

    @property
    def size_pt(self) -> tuple[float, float]:
        return self.size[0] / self.scale, self.size[1] / self.scale

    def clear(self) -> None:
        self.surface.fill((0, 0, 0, 0))
        self.dirty = True

    def to_bytes(self) -> tuple[bytes, tuple[int, int]]:
        data = pygame.image.tobytes(self.surface, "RGBA", True)
        self.dirty = False
        return data, self.size

    # helpers
    def _font(self, size_pt: float, mono: bool = False) -> pygame.font.Font:
        px = max(1, int(round(size_pt * self.scale)))
        key = ("m" if mono else "p", px)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("Menlo,Helvetica", px)
        return self._fonts[key]

    def _fit(self, text: str, max_pt: float, font: pygame.font.Font) -> str:
        """Truncate with '…' so the rendered width is within max_pt points."""
        limit = max_pt * self.scale
        if font.size(text)[0] <= limit:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.size(text[:mid] + "…")[0] <= limit:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "…"

    def _text(self, text: str, x_pt: float, y_pt: float, font, color=TEXT) -> pygame.Rect:
        img = font.render(text, True, color)
        r = self.surface.blit(img, (round(x_pt * self.scale), round(y_pt * self.scale)))
        return r

    def _rect(self, x, y, w, h, color) -> None:
        s = self.scale
        self.surface.fill(color, pygame.Rect(round(x * s), round(y * s), round(w * s), round(h * s)))

    # timeline
    def _track_rect(self) -> tuple[float, float, float, float]:
        """Track (x0, y0, x1, y1) in points: the bar minus left controls and right padding."""
        w, h = self.size_pt
        return CONTROLS_W, h - BAR_H, w - BAR_PAD, h

    def timeline_hit(self, x_pt: float, y_pt: float) -> float | None:
        x0, y0, x1, y1 = self._track_rect()
        if not (x0 <= x_pt <= x1 and y0 <= y_pt <= y1) or x1 <= x0:
            return None
        return (x_pt - x0) / (x1 - x0)

    def draw_timeline(self, hist, playhead_frac: float, playing: bool, speed: float,
                      tag_fracs, lineage_colors_per_bin) -> None:
        s = self.scale
        x0, y0, x1, y1 = self._track_rect()
        w_pt, _ = self.size_pt
        self._rect(0, y0, w_pt, BAR_H, BAR_BG)
        # play / pause icon
        cx, cy = 28, y0 + BAR_H / 2
        if playing:
            self._rect(cx - 8, cy - 9, 5, 18, TEXT)
            self._rect(cx + 3, cy - 9, 5, 18, TEXT)
        else:
            pygame.draw.polygon(self.surface, TEXT, [
                (round((cx - 7) * s), round((cy - 10) * s)),
                (round((cx - 7) * s), round((cy + 10) * s)),
                (round((cx + 9) * s), round(cy * s))])
        font = self._font(13)
        label = f"{speed:g}×"
        self._text(label, 52, cy - 8, font, DIM)
        # histogram
        hist = np.asarray(hist, dtype=float)
        n = len(hist)
        track_w, track_h = x1 - x0, BAR_H - 16
        if n and hist.max() > 0:
            peak = hist.max()
            for i in range(n):
                bh = max(1.0, track_h * hist[i] / peak) if hist[i] > 0 else 0
                if not bh:
                    continue
                if lineage_colors_per_bin is not None and i < len(lineage_colors_per_bin):
                    r, g, b = (int(max(0, min(1, c)) * 255) for c in lineage_colors_per_bin[i][:3])
                else:
                    r, g, b = HIST_DEFAULT
                bx = x0 + track_w * i / n
                self._rect(bx, y1 - 8 - bh, max(track_w / n - 0.5, 0.5), bh, (r, g, b, 220))
        # tags
        for f in tag_fracs:
            tx = x0 + track_w * min(max(f, 0.0), 1.0)
            self._rect(tx - 0.5, y0 + 4, 1, 8, ACCENT)
        # playhead
        px = x0 + track_w * min(max(playhead_frac, 0.0), 1.0)
        self._rect(px - 1, y0 + 4, 2, BAR_H - 8, TEXT)
        self.dirty = True

    # panel
    def draw_panel(self, lines: list[str], visible: bool) -> None:
        self.dirty = True
        if not visible:
            return
        w, h = self.size_pt
        x = w - PANEL_W
        self._rect(x, 0, PANEL_W, h - BAR_H, PANEL_BG)
        font = self._font(12, mono=True)
        lh = font.get_linesize() / self.scale
        y = 12.0
        for line in lines:
            if y + lh > h - BAR_H - 28:
                break
            self._text(self._fit(line, PANEL_W - 24, font), x + 12, y, font)
            y += lh
        small = self._font(11)
        self._text(self._fit(HELP_LINE, PANEL_W - 24, small), x + 12,
                   h - BAR_H - 22, small, DIM)

    # small widgets
    def draw_tooltip(self, text: str, x_pt: float, y_pt: float) -> None:
        font = self._font(12)
        text = self._fit(text, TOOLTIP_MAX_W, font)
        tw, th = font.size(text)
        tw, th = tw / self.scale, th / self.scale
        pad = 6
        w, h = self.size_pt
        bx = min(max(x_pt + 14, 0), max(w - tw - 2 * pad, 0))
        by = min(max(y_pt + 14, 0), max(h - th - 2 * pad, 0))
        if x_pt + 14 + tw + 2 * pad > w:
            bx = max(x_pt - 14 - tw - 2 * pad, 0)
        if y_pt + 14 + th + 2 * pad > h:
            by = max(y_pt - 14 - th - 2 * pad, 0)
        self._rect(bx, by, tw + 2 * pad, th + 2 * pad, PANEL_BG)
        self._text(text, bx + pad, by + pad, font)
        self.dirty = True

    def draw_lens(self, text: str) -> None:
        font = self._font(13)
        tw = font.size(text)[0] / self.scale
        self._rect(12, 12, tw + 16, 26, BAR_BG)
        self._text(text, 20, 16, font)
        self.dirty = True

    def draw_progress(self, text: str, fraction: float | None) -> None:
        w, h = self.size_pt
        bw, bh = 320, 6
        x, y = (w - bw) / 2, h / 2
        font = self._font(13)
        tw = font.size(text)[0] / self.scale
        self._text(text, (w - tw) / 2, y - 26, font)
        self._rect(x, y, bw, bh, (60, 64, 80, 255))
        if fraction is None:
            self._rect(x + bw * 0.35, y, bw * 0.3, bh, ACCENT)
        else:
            self._rect(x, y, bw * min(max(fraction, 0.0), 1.0), bh, ACCENT)
        self.dirty = True

    def draw_message(self, text: str) -> None:
        w, h = self.size_pt
        font = self._font(18)
        tw, th = (v / self.scale for v in font.size(text))
        self._text(text, (w - tw) / 2, (h - th) / 2, font)
        self.dirty = True

    def draw_fps(self, fps: float | None) -> None:
        self.dirty = True
        if fps is None:
            return
        font = self._font(12, mono=True)
        text = f"{fps:.0f} fps"
        tw = font.size(text)[0] / self.scale
        self._text(text, self.size_pt[0] - tw - 12, 12, font, DIM)
