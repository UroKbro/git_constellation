"""2D UI overlay drawn with pygame onto an RGBA surface the renderer uploads as a texture.

Layout constants are in window points; drawing multiplies by `scale` (drawable px per point).
Colours come from the active theme (themes.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygame

from constellation.themes import DEEP_SPACE, Theme

BAR_H = 64
CONTROLS_W = 110  # left of the bar: play icon + speed text
BAR_PAD = 16
PANEL_W = 360
TOOLTIP_MAX_W = 600
RADIUS = 10
KEYS = [("Space", "play"), ("← →", "step"), ("+ −", "speed"), ("P", "theme"),
        ("M", "sound"), ("F12", "screenshot"), ("R", "record"), ("E / G", "MP4 / GIF"),
        ("I", "panel"), ("Esc", "quit")]
ADDED, DELETED = (120, 220, 140, 255), (240, 110, 110, 255)


@dataclass
class PanelInfo:
    """Header and legend for the side panel (the body is the detail lines)."""
    repo: str = ""
    date: str = ""
    position: str = ""
    progress: float = 0.0
    stats: list[tuple[str, str]] = field(default_factory=list)   # (value, label)
    heading: str = ""
    legend: list[tuple[tuple, str, int]] = field(default_factory=list)  # (rgb, name, count)


def compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 100_000:
        return f"{n / 1e3:.0f}k"
    return f"{n:,}"


class Overlay:
    def __init__(self, size: tuple[int, int], scale: float, theme: Theme = DEEP_SPACE) -> None:
        pygame.font.init()
        self._fonts: dict[tuple, pygame.font.Font] = {}
        self.theme = theme
        self.buttons: dict[str, tuple[float, float, float, float]] = {}
        self.resize(size, scale)

    def resize(self, size: tuple[int, int], scale: float) -> None:
        self.size = (int(size[0]), int(size[1]))
        self.scale = float(scale)
        self.surface = pygame.Surface(self.size, pygame.SRCALPHA)
        self._fonts = {}
        self.dirty = True

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.dirty = True

    @property
    def size_pt(self) -> tuple[float, float]:
        return self.size[0] / self.scale, self.size[1] / self.scale

    def clear(self) -> None:
        self.surface.fill((0, 0, 0, 0))
        self.buttons = {}
        self.dirty = True

    def to_bytes(self) -> tuple[bytes, tuple[int, int]]:
        data = pygame.image.tobytes(self.surface, "RGBA", True)
        self.dirty = False
        return data, self.size

    # helpers
    def _c(self, name: str, a: int = 255):
        return self.theme.ui(name, a)

    def _font(self, size_pt: float, mono: bool = False, bold: bool = False) -> pygame.font.Font:
        px = max(1, int(round(size_pt * self.scale)))
        key = (mono, bold, px)
        if key not in self._fonts:
            names = "menlo,monaco,couriernew" if mono else "sfns,helveticaneue,helvetica,arial"
            f = pygame.font.SysFont(names, px)
            f.bold = bold and not mono
            self._fonts[key] = f
        return self._fonts[key]

    def _w(self, text: str, font) -> float:
        return font.size(text)[0] / self.scale

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

    def _wrap(self, text: str, max_pt: float, font, max_lines: int) -> list[str]:
        out, rest = [], text
        while rest and len(out) < max_lines - 1 and self._w(rest, font) > max_pt:
            cut = len(self._fit(rest, max_pt, font)) - 1
            brk = max(rest.rfind(" ", 0, cut), rest.rfind("/", 0, cut))
            cut = brk + 1 if brk > 0 else cut
            out.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip()
        if rest:
            out.append(self._fit(rest, max_pt, font))
        return out

    def _text(self, text: str, x_pt: float, y_pt: float, font, color=None) -> pygame.Rect:
        img = font.render(text, True, color or self._c("text"))
        return self.surface.blit(img, (round(x_pt * self.scale), round(y_pt * self.scale)))

    def _rect(self, x, y, w, h, color, radius: float = 0, width: float = 0) -> None:
        """Alpha-blended (rounded) rectangle in points."""
        s = self.scale
        r = pygame.Rect(round(x * s), round(y * s), max(round(w * s), 1), max(round(h * s), 1))
        if r.w <= 0 or r.h <= 0:
            return
        tmp = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(tmp, color, tmp.get_rect(), width=round(width * s),
                         border_radius=round(radius * s))
        self.surface.blit(tmp, r.topleft)

    def _circle(self, x, y, r, color, width: float = 0) -> None:
        s = self.scale
        pygame.draw.circle(self.surface, color, (round(x * s), round(y * s)), max(r * s, 1),
                           width=round(width * s))

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
                      tag_fracs, lineage_colors_per_bin, ticks=(), date: str = "") -> None:
        s = self.scale
        x0, y0, x1, y1 = self._track_rect()
        w_pt, _ = self.size_pt
        self._rect(0, y0, w_pt, BAR_H, self._c("panel", 225))
        self._rect(0, y0, w_pt, 1, self._c("dim", 60))
        # play / pause button
        cx, cy = 30, y0 + BAR_H / 2
        self._circle(cx, cy, 17, self._c("accent", 40))
        self._circle(cx, cy, 17, self._c("accent"), width=1.5)
        if playing:
            self._rect(cx - 6, cy - 7, 4, 14, self._c("text"), 1)
            self._rect(cx + 2, cy - 7, 4, 14, self._c("text"), 1)
        else:
            pygame.draw.polygon(self.surface, self._c("text"), [
                (round((cx - 4) * s), round((cy - 8) * s)),
                (round((cx - 4) * s), round((cy + 8) * s)),
                (round((cx + 8) * s), round(cy * s))])
        self._text(f"{speed:g}×", 56, cy - 9, self._font(14, bold=True))
        self._text("speed", 56, cy + 6, self._font(9.5), self._c("dim"))
        # histogram: played part bright, the future dimmed
        hist = np.asarray(hist, dtype=float)
        n = len(hist)
        track_w = x1 - x0
        top, base = y0 + 10, y1 - 18
        if n and hist.max() > 0:
            peak = hist.max()
            ph_x = x0 + track_w * min(max(playhead_frac, 0.0), 1.0)
            for i in range(n):
                if hist[i] <= 0:
                    continue
                bh = max(1.0, (base - top) * (hist[i] / peak) ** 0.7)
                if lineage_colors_per_bin is not None and i < len(lineage_colors_per_bin):
                    r, g, b = (int(max(0, min(1, c)) * 255) for c in lineage_colors_per_bin[i][:3])
                else:
                    r, g, b = self._c("accent")[:3]
                bx = x0 + track_w * i / n
                a = 230 if bx <= ph_x else 70
                self._rect(bx, base - bh, max(track_w / n - 0.6, 0.5), bh, (r, g, b, a))
        # year ticks
        small = self._font(9.5)
        last_x = -1e9
        for f, label in ticks:
            tx = x0 + track_w * f
            self._rect(tx, base + 1, 1, 4, self._c("dim", 140))
            if tx - last_x > 34 and tx + 30 < x1:
                self._text(label, tx + 3, base + 3, small, self._c("dim"))
                last_x = tx
        # tags
        for f in tag_fracs:
            tx = x0 + track_w * min(max(f, 0.0), 1.0)
            self._rect(tx - 0.5, y0 + 3, 1, 5, self._c("accent", 200))
        # playhead + date bubble
        px = x0 + track_w * min(max(playhead_frac, 0.0), 1.0)
        self._rect(px - 1, y0 + 4, 2, BAR_H - 8, self._c("accent"))
        self._circle(px, y0 + 4, 4, self._c("accent"))
        if date:
            f = self._font(11, bold=True)
            tw = self._w(date, f) + 14
            bx = min(max(px - tw / 2, 4), w_pt - tw - 4)
            self._rect(bx, y0 - 26, tw, 20, self._c("accent", 235), 10)
            self._text(date, bx + 7, y0 - 23.5, f, self._c("panel"))
        self.dirty = True

    # panel
    def draw_panel(self, lines: list[str], visible: bool, info: PanelInfo | None = None) -> None:
        self.dirty = True
        if not visible:
            return
        info = info or PanelInfo()
        w, h = self.size_pt
        x, top, pw = w - PANEL_W, 12.0, PANEL_W - 12
        bottom = h - BAR_H - 36
        self._rect(x, top, pw, bottom - top, self._c("panel", 215), 14)
        self._rect(x, top, pw, bottom - top, self._c("dim", 50), 14, width=1)
        x += 18
        iw = pw - 36
        y = top + 16
        if info.repo:
            f = self._font(21, bold=True)
            self._text(self._fit(info.repo, iw, f), x, y, f)
            y += 30
        if info.date:
            self._text(info.date, x, y, self._font(13), self._c("accent"))
            y += 20
        if info.position:
            self._text(info.position, x, y, self._font(10.5), self._c("dim"))
            y += 16
            self._rect(x, y, iw, 3, self._c("dim", 60), 1.5)
            self._rect(x, y, max(iw * info.progress, 3), 3, self._c("accent"), 1.5)
            y += 14
        if info.stats:
            tw = (iw - 8 * (len(info.stats) - 1)) / len(info.stats)
            for k, (val, label) in enumerate(info.stats):
                tx = x + k * (tw + 8)
                self._rect(tx, y, tw, 46, self._c("text", 14), 8)
                self._text(val, tx + 10, y + 6, self._font(16, bold=True))
                self._text(label.upper(), tx + 10, y + 28, self._font(9), self._c("dim"))
            y += 60
        # footer: keycaps, then legend above it
        keys_h = 18 * ((len(KEYS) + 1) // 2) + 26
        legend_rows = (len(info.legend) + 1) // 2
        legend_h = 22 + 10 + 18 * legend_rows if info.legend else 0
        foot = bottom - keys_h
        if foot - legend_h - y < 90:
            legend_h = 0
        if foot - y >= 70:
            self._keycaps(x, foot, iw)
        else:
            foot = bottom - 12
        if legend_h:
            self._legend(x, foot - legend_h, iw, info.legend)
        limit = foot - legend_h - 8
        # body
        if info.heading:
            self._text(info.heading.upper(), x, y, self._font(9.5, bold=True), self._c("dim"))
            y += 18
        self._body(lines, x, y, iw, limit)

    def _body(self, lines: list[str], x, y, iw, limit) -> None:
        body, mono = self._font(12), self._font(11, mono=True)
        title = self._font(14, bold=True)
        lh = body.get_linesize() / self.scale
        for i, line in enumerate(lines):
            if y + lh > limit:
                return
            if i == 0 and " · " in line and len(line.split(" · ")[0]) == 10:
                sha, _, rest = line.partition(" · ")   # commit meta: sha · author · date
                r = self._text(sha, x, y + 1, mono, self._c("accent"))
                self._text(self._fit(rest, iw - r.w / self.scale - 8, body),
                           x + r.w / self.scale + 8, y, body, self._c("dim"))
            elif line.startswith(("+", "−")) or " · +" in line:
                head, _, tail = line.rpartition(" · ")
                cx = x
                if head:
                    cx += self._text(head + "  ", cx, y, body, self._c("dim")).w / self.scale
                for part in tail.split():
                    col = ADDED if part.startswith("+") else DELETED
                    cx += self._text(part + "  ", cx, y, body, col).w / self.scale
            elif line.startswith("Hotspot "):
                v = float(line.split()[1])
                self._text("Hotspot", x, y, body, self._c("dim"))
                bx = x + 70
                self._rect(bx, y + 6, iw - 120, 5, self._c("dim", 60), 2.5)
                self._rect(bx, y + 6, max((iw - 120) * min(v, 1.0), 4), 5,
                           self._c("accent"), 2.5)
                self._text(f"{v:.2f}", x + iw - 40, y, body)
            elif line.startswith("  "):
                self._text(self._fit(line.strip(), iw - 12, body), x + 12, y, body, self._c("dim"))
            elif i <= 1 and not line.startswith(("Authors", "First", "Coupled")):
                for row in self._wrap(line, iw, title, 3):
                    if y + lh > limit:
                        return
                    self._text(row, x, y, title)
                    y += title.get_linesize() / self.scale
                y += 4
                continue
            else:
                self._text(self._fit(line, iw, body), x, y, body)
            y += lh + 2

    def _legend(self, x, y, iw, legend) -> None:
        self._text("LANGUAGES", x, y, self._font(9.5, bold=True), self._c("dim"))
        y += 18
        total = max(sum(c for _, _, c in legend), 1)
        cx = x
        for rgb, _, c in legend:   # composition bar
            sw = iw * c / total
            self._rect(cx, y, max(sw - 1, 1), 6, (*(round(v * 255) for v in rgb), 255), 2)
            cx += sw
        y += 16
        f, colw = self._font(11), iw / 2
        for k, (rgb, name, c) in enumerate(legend):
            lx, ly = x + (k % 2) * colw, y + (k // 2) * 18
            self._circle(lx + 5, ly + 8, 4.5, (*(round(v * 255) for v in rgb), 255))
            self._text(name, lx + 15, ly, f)
            cs = compact(c)
            self._text(cs, lx + colw - 12 - self._w(cs, f), ly, f, self._c("dim"))

    def _keycaps(self, x, y, iw) -> None:
        self._rect(x, y, iw, 1, self._c("dim", 50))
        y += 10
        kf, lf, colw = self._font(9.5, bold=True), self._font(10.5), iw / 2
        for k, (key, label) in enumerate(KEYS):
            lx, ly = x + (k % 2) * colw, y + (k // 2) * 18
            kw = self._w(key, kf) + 10
            self._rect(lx, ly, kw, 15, self._c("text", 30), 4)
            self._text(key, lx + 5, ly + 1.5, kf)
            self._text(label, lx + kw + 6, ly + 1, lf, self._c("dim"))

    # toolbar
    def draw_toolbar(self, items: list[tuple[str, str, bool]], hover: str | None = None,
                     rec: bool = False) -> None:
        """Pill buttons top-left: (id, label, active). Hit-test with `button_hit`."""
        f = self._font(12)
        x, y, h = 14.0, 14.0, 28.0
        for bid, label, active in items:
            dot = bid in ("lens", "theme") or (bid == "rec" and rec)
            w = self._w(label, f) + 24 + (12 if dot else 0)
            bg = self._c("accent", 70) if active else self._c("panel", 200 if hover != bid else 245)
            self._rect(x, y, w, h, bg, 14)
            self._rect(x, y, w, h, self._c("accent" if hover == bid else "dim", 110), 14, width=1)
            tx = x + 12
            if dot:
                col = ((235, 70, 70, 255) if bid == "rec" else
                       (*(round(v * 255) for v in self.theme.languages[0]), 255)
                       if bid == "theme" else self._c("accent"))
                self._circle(tx + 4, y + h / 2, 4, col)
                tx += 12
            self._text(label, tx, y + 6, f)
            self.buttons[bid] = (x, y, w, h)
            x += w + 8
        self.dirty = True

    def button_hit(self, x_pt: float, y_pt: float) -> str | None:
        for bid, (x, y, w, h) in self.buttons.items():
            if x <= x_pt <= x + w and y <= y_pt <= y + h:
                return bid
        return None

    # small widgets
    def draw_tooltip(self, text: str, x_pt: float, y_pt: float) -> None:
        font = self._font(12)
        text = self._fit(text, TOOLTIP_MAX_W, font)
        tw, th = font.size(text)
        tw, th = tw / self.scale, th / self.scale
        pad = 8
        w, h = self.size_pt
        bx = min(max(x_pt + 14, 0), max(w - tw - 2 * pad, 0))
        by = min(max(y_pt + 14, 0), max(h - th - 2 * pad, 0))
        if x_pt + 14 + tw + 2 * pad > w:
            bx = max(x_pt - 14 - tw - 2 * pad, 0)
        if y_pt + 14 + th + 2 * pad > h:
            by = max(y_pt - 14 - th - 2 * pad, 0)
        self._rect(bx, by, tw + 2 * pad, th + 2 * pad, self._c("panel", 235), 8)
        self._rect(bx, by, tw + 2 * pad, th + 2 * pad, self._c("accent", 120), 8, width=1)
        self._text(text, bx + pad, by + pad, font)
        self.dirty = True

    def draw_toast(self, text: str) -> None:
        w, h = self.size_pt
        f = self._font(12.5)
        text = self._fit(text, w * 0.6, f)
        tw = self._w(text, f) + 32
        right = w - PANEL_W
        bx, by = max((right - tw) / 2, 8), h - BAR_H - 70
        self._rect(bx, by, tw, 32, self._c("panel", 240), 16)
        self._rect(bx, by, tw, 32, self._c("accent", 160), 16, width=1)
        self._text(text, bx + 16, by + 8, f)
        self.dirty = True

    def draw_lens(self, text: str) -> None:
        font = self._font(13, bold=True)
        tw = self._w(text, font)
        self._rect(14, 14, tw + 22, 28, self._c("panel", 200), 14)
        self._text(text, 25, 19, font)
        self.dirty = True

    def draw_progress(self, text: str, fraction: float | None) -> None:
        w, h = self.size_pt
        bw, bh = 340, 6
        x, y = (w - bw) / 2, h / 2
        big = self._font(28, bold=True)
        title = "git-constellation"
        self._text(title, (w - self._w(title, big)) / 2, y - 80, big)
        font = self._font(13)
        self._text(text, (w - self._w(text, font)) / 2, y - 28, font, self._c("dim"))
        self._rect(x, y, bw, bh, self._c("dim", 70), 3)
        frac = 0.3 if fraction is None else min(max(fraction, 0.0), 1.0)
        off = bw * 0.35 if fraction is None else 0
        self._rect(x + off, y, max(bw * frac, bh), bh, self._c("accent"), 3)
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
        tw = self._w(text, font)
        self._text(text, self.size_pt[0] - PANEL_W - tw - 12, 16, font, self._c("dim"))
