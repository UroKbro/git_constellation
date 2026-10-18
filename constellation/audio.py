"""Cosmic soundscape: an ambient pad plus soft chimes for commits and bells for merges.

Everything is synthesised with numpy (no audio files). Pure synth functions come first;
`Soundscape` wraps pygame.mixer and degrades to silence when no audio device exists."""
from __future__ import annotations

import math

import numpy as np

from constellation.themes import Theme

RATE = 44_100
PAD_S = 12.0              # seamless loop: every partial completes whole cycles
PAD_VOLUME = 0.22
CHIME_VOLUME = 0.32
MAX_CHIMES_PER_S = 7.0
MAX_STEP_FOR_SOUND = 64   # bigger playhead jumps (scrubs) stay silent
CHANNELS = 24


def note_hz(root_hz: float, semitones: float) -> float:
    return root_hz * 2.0 ** (semitones / 12.0)


def scale_notes(theme: Theme, octaves: int = 2) -> list[float]:
    """Chime pitches: the theme's scale over `octaves`, starting an octave above root."""
    return [note_hz(theme.root_hz * 2, 12 * o + s) for o in range(octaves) for s in theme.scale]


def _stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """(n, 2) int16 from a float mono signal in [-1, 1]; pan in [-1, 1] (equal power)."""
    a = (pan + 1.0) * math.pi / 4
    st = np.stack([mono * math.cos(a), mono * math.sin(a)], axis=1)
    return (np.clip(st, -1.0, 1.0) * 32767).astype(np.int16)


def pad(theme: Theme, seconds: float = PAD_S, rate: int = RATE) -> np.ndarray:
    """A slow, breathing drone on the theme's root (root, fifth, octave, ninth); loops cleanly."""
    t = np.arange(int(seconds * rate)) / rate
    out = np.zeros_like(t)
    voices = [(-12, 0.32), (-5, 0.18), (0, 0.22), (7, 0.12), (14, 0.07)]
    for i, (semi, amp) in enumerate(voices):
        for detune in (-0.004, 0.004):
            f = note_hz(theme.root_hz, semi) * (1 + detune)
            f = round(f * seconds) / seconds          # whole cycles per loop
            lfo = 0.6 + 0.4 * np.sin(2 * math.pi * (i + 1) * t / seconds + i)
            out += amp * lfo * np.sin(2 * math.pi * f * t)
    out /= max(np.abs(out).max(), 1e-9)
    return _stereo(0.8 * out)


def chime(freq: float, seconds: float = 1.8, pan: float = 0.0, rate: int = RATE,
          bell: bool = False) -> np.ndarray:
    """A soft struck tone with exponential decay; `bell` adds inharmonic partials."""
    t = np.arange(int(seconds * rate)) / rate
    partials = ([(1.0, 1.0), (2.76, 0.45), (5.40, 0.22), (8.93, 0.08)] if bell
                else [(1.0, 1.0), (2.0, 0.28), (3.01, 0.10)])
    out = np.zeros_like(t)
    for mult, amp in partials:
        out += amp * np.exp(-t * (2.2 + mult * 0.9)) * np.sin(2 * math.pi * freq * mult * t)
    attack = np.minimum(t / 0.006, 1.0)
    out *= attack
    out /= max(np.abs(out).max(), 1e-9)
    return _stereo(0.9 * out, pan)


def commit_note(author: str, n_notes: int) -> int:
    """Stable scale degree per author, so each contributor has their own voice."""
    return sum(author.encode()) * 2654435761 % (2 ** 32) % max(n_notes, 1)


class Soundscape:
    """Ambient pad + rate-limited chimes. `enabled` is False when audio is unavailable."""

    def __init__(self, theme: Theme, muted: bool = False) -> None:
        self.muted = muted
        self.enabled = False
        self._budget = 2.0
        self._last_t = None
        self._theme = None
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=RATE, size=-16, channels=2, buffer=1024)
            pygame.mixer.set_num_channels(CHANNELS)
            self._pg = pygame
            self._pad_channel = pygame.mixer.Channel(0)
            pygame.mixer.set_reserved(1)
            self.enabled = True
        except Exception:  # noqa: BLE001 - no device / no SDL audio: stay silent
            return
        self.set_theme(theme)

    def set_theme(self, theme: Theme) -> None:
        if not self.enabled or theme is self._theme:
            return
        self._theme = theme
        snd = self._pg.sndarray.make_sound
        self._pad = snd(pad(theme))
        notes = scale_notes(theme)
        self._chimes = [snd(chime(f, pan=-0.6 + 1.2 * i / max(len(notes) - 1, 1)))
                        for i, f in enumerate(notes)]
        self._bells = [snd(chime(note_hz(theme.root_hz, s), 3.0, pan, bell=True))
                       for s, pan in ((0, -0.3), (7, 0.3))]
        self._pad.set_volume(PAD_VOLUME)
        for s in self._chimes + self._bells:
            s.set_volume(CHIME_VOLUME)
        if not self.muted:
            self._pad_channel.play(self._pad, loops=-1, fade_ms=1500)

    def toggle(self) -> bool:
        self.set_muted(not self.muted)
        return self.muted

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if not self.enabled:
            return
        if muted:
            self._pg.mixer.fadeout(300)
        else:
            self._pad_channel.play(self._pad, loops=-1, fade_ms=1500)

    def commits(self, authors: list[str], merges: list[bool], sizes: list[int],
                now: float) -> int:
        """Play chimes for newly replayed commits (oldest first). Returns sounds played."""
        if not self.enabled or self.muted or not authors:
            return 0
        if self._last_t is not None:
            self._budget = min(self._budget + (now - self._last_t) * MAX_CHIMES_PER_S, 2.0)
        self._last_t = now
        played = 0
        for author, merge, size in zip(authors, merges, sizes):
            if self._budget < 1.0:
                break
            self._budget -= 1.0
            if merge:
                for b in self._bells:
                    b.play()
            else:
                s = self._chimes[commit_note(author, len(self._chimes))]
                ch = s.play()
                if ch is not None:
                    v = 0.45 + 0.55 * min(size / 20.0, 1.0)
                    ch.set_volume(v, v)
            played += 1
        return played

    def stop(self) -> None:
        if self.enabled:
            self._pg.mixer.stop()
