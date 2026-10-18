"""Themes, soundscape synthesis, recorder commands and UI chrome."""
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from constellation.app import year_ticks
from constellation.audio import chime, commit_note, pad, scale_notes
from constellation.palette import FAMILY_NAMES, language_family
from constellation.recorder import VideoWriter, ffmpeg_cmd, output_path
from constellation.render.ui import Overlay, PanelInfo, compact
from constellation.themes import DEEP_SPACE, THEMES, next_theme, theme_named


def test_themes_distinct_language_colours_and_cycle():
    for t in THEMES:
        assert len(t.languages) == len(FAMILY_NAMES)
        assert len(set(t.languages)) == len(t.languages)
    assert next_theme(THEMES[-1]) is THEMES[0]
    assert theme_named("nope") is DEEP_SPACE
    assert theme_named("Solar Flare").name == "Solar Flare"
    assert FAMILY_NAMES[language_family("a/b.py")] == "Python"
    assert FAMILY_NAMES[language_family("Makefile")] == "Other"


def test_synth_shapes_and_loop():
    p = pad(DEEP_SPACE, seconds=1.0, rate=8000)
    assert p.shape == (8000, 2) and p.dtype == np.int16
    step = np.abs(np.diff(p[:, 0].astype(int))).max()
    assert abs(int(p[0, 0]) - int(p[-1, 0])) <= step  # whole cycles: loops without a click
    c = chime(440.0, seconds=0.5, pan=1.0, rate=8000)
    assert c.shape == (4000, 2) and abs(c[:, 0]).max() < abs(c[:, 1]).max()
    assert len(scale_notes(DEEP_SPACE)) == 10
    assert commit_note("ada", 10) == commit_note("ada", 10) < 10


def test_ffmpeg_cmds_and_paths(tmp_path):
    mp4 = ffmpeg_cmd("ffmpeg", Path("a.mp4"), (2560, 1600), 30)
    assert "libx264" in mp4 and "2560x1600" in mp4 and mp4[-1] == "a.mp4"
    gif = ffmpeg_cmd("ffmpeg", Path("a.gif"), (2560, 1600), 30)
    assert any("palettegen" in a for a in gif)
    p = output_path("my repo", "png", datetime(2026, 1, 2, 3, 4, 5), tmp_path)
    assert p.name == "my_repo-20260102-030405.png"


def test_video_writer_png_fallback(tmp_path):
    w = VideoWriter(tmp_path / "v.mp4", (8, 6), 30, ffmpeg="")
    for _ in range(3):
        w.write(np.zeros((6, 8, 3), np.uint8))
    out = w.close()
    assert out == tmp_path / "v" and len(list(out.glob("*.png"))) == 3


def test_year_ticks():
    t0, t1 = datetime(2019, 6, 1).timestamp(), datetime(2022, 6, 1).timestamp()
    ticks = year_ticks(np.array([t0, t1]))
    assert [label for _, label in ticks] == ["2020", "2021", "2022"]
    assert all(0 < f < 1 for f, _ in ticks)
    assert year_ticks(np.array([t0])) == []


def test_overlay_toolbar_panel_and_themes():
    pygame.font.init()
    ov = Overlay((1600, 1000), 2.0)
    for t in THEMES:
        ov.set_theme(t)
        ov.clear()
        ov.draw_toolbar([("theme", t.name, False), ("sound", "Sound on", True)], "sound")
        info = PanelInfo("repo", "Mon 01 Jan 2024", "Commit 1 of 2", 0.5,
                         [("10", "files"), ("1.2M", "lines"), ("3", "authors")], "File",
                         [(t.languages[0], "Python", 7), (t.languages[1], "JS / TS", 3)])
        ov.draw_panel(["0123456789 · ada · 2024-01-01 10:00", "subject", "2 files · +3 −1",
                       "Hotspot 0.40", "  other/path.py (50%)"], True, info)
        ov.draw_toast("Saved")
        ov.draw_timeline(np.array([1, 2, 3]), 0.5, False, 4, [0.2], None,
                         [(0.3, "2020")], "01 Jan 2024")
    x, y, w, h = ov.buttons["sound"]
    assert ov.button_hit(x + w / 2, y + h / 2) == "sound"
    assert ov.button_hit(5, 400) is None
    assert compact(1234) == "1,234" and compact(1_234_567) == "1.2M"
