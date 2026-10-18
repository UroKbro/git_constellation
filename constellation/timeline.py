"""Playback clock: a float commit cursor over a window, advanced at N commits/s."""
from __future__ import annotations

import numpy as np

from .model import Window

COMMITS_PER_SECOND = 4.0
SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]


class Timeline:
    def __init__(self, window: Window, commit_ts: np.ndarray):
        self.window = window
        self._n = len(commit_ts)
        self._runmax = (np.maximum.accumulate(np.asarray(commit_ts, dtype=np.int64))
                        if self._n else np.zeros(0, dtype=np.int64))
        self.playing = False
        self.speed: float = 1.0
        self.cursor: float = float(window.end)

    @property
    def playhead(self) -> int:
        if self._n == 0:
            return 0
        w = self.window
        if w.empty:
            return min(max(w.end, 0), self._n - 1)
        return min(max(int(np.floor(self.cursor)), w.start), w.end)

    def toggle(self) -> None:
        if self.window.empty:
            return
        if not self.playing and self.cursor >= self.window.end:
            self.cursor = float(self.window.start)
        self.playing = not self.playing

    def _speed_index(self) -> int:
        return min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - self.speed))

    def faster(self) -> None:
        self.speed = float(SPEEDS[min(self._speed_index() + 1, len(SPEEDS) - 1)])

    def slower(self) -> None:
        self.speed = float(SPEEDS[max(self._speed_index() - 1, 0)])

    def _set(self, cursor: float) -> None:
        w = self.window
        if not w.empty:
            self.cursor = float(min(max(cursor, w.start), w.end))

    def step(self, n: int) -> None:
        self.playing = False
        self._set(self.playhead + n)

    def home(self) -> None:
        self._set(self.window.start)

    def end(self) -> None:
        self._set(self.window.end)

    def _times(self) -> np.ndarray:
        w = self.window
        return self._runmax[w.start:w.end + 1]

    def scrub(self, fraction: float) -> None:
        if self.window.empty:
            return
        t = self._times()
        target = t[0] + min(max(fraction, 0.0), 1.0) * (t[-1] - t[0])
        i = int(np.searchsorted(t, target, side="right")) - 1
        self._set(self.window.start + max(i, 0))

    def update(self, dt: float) -> tuple[int, int]:
        prev = self.playhead
        if self.playing and not self.window.empty:
            self._set(self.cursor + dt * self.speed * COMMITS_PER_SECOND)
            if self.cursor >= self.window.end:
                self.playing = False
        return prev, self.playhead

    def histogram(self, bins: int) -> np.ndarray:
        out = np.zeros(bins, dtype=np.int64)
        if self.window.empty or bins <= 0:
            return out
        t = self._times()
        span = int(t[-1] - t[0])
        if span == 0:
            out[0] = len(t)
            return out
        idx = np.minimum(((t - t[0]) * bins // span).astype(np.int64), bins - 1)
        return np.bincount(idx, minlength=bins).astype(np.int64)

    def fraction_of(self, index: int) -> float:
        if self.window.empty:
            return 0.0
        w = self.window
        t = self._times()
        index = min(max(index, w.start), w.end)
        span = int(t[-1] - t[0])
        if span == 0:
            return (index - w.start) / max(w.size - 1, 1)
        return float(t[index - w.start] - t[0]) / span
