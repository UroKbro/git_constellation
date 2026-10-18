import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from constellation.render.ui import Overlay


@pytest.fixture
def ov():
    pygame.font.init()
    return Overlay((1600, 1000), 2.0)  # 800x500 points


def test_timeline_hit_edges(ov):
    x0, y0, x1, y1 = ov._track_rect()
    ymid = (y0 + y1) / 2
    assert ov.timeline_hit(x0, ymid) == 0.0
    assert ov.timeline_hit(x1, ymid) == 1.0
    assert ov.timeline_hit((x0 + x1) / 2, ymid) == pytest.approx(0.5)
    assert ov.timeline_hit(x0 - 1, ymid) is None
    assert ov.timeline_hit(x0 + 10, y0 - 5) is None
    assert ov.timeline_hit(x0 + 10, 5) is None


def test_to_bytes_length_and_dirty(ov):
    ov.clear()
    ov.draw_message("No commits yet")
    assert ov.dirty
    data, (w, h) = ov.to_bytes()
    assert (w, h) == (1600, 1000)
    assert len(data) == w * h * 4
    assert not ov.dirty


def test_draw_methods_mark_dirty(ov):
    ov.to_bytes()
    ov.draw_panel(["a", "b"], True)
    assert ov.dirty
    ov.to_bytes()
    ov.draw_timeline(np.array([1, 5, 2, 0]), 0.5, True, 0.25, [0.1], None)
    assert ov.dirty
    ov.to_bytes()
    ov.draw_lens("1 History")
    ov.draw_progress("Loading", None)
    ov.draw_progress("Loading", 0.4)
    ov.draw_fps(60.0)
    ov.draw_fps(None)
    assert ov.dirty


def test_tooltip_truncated(ov):
    long = "src/" + "very_long_directory_name/" * 40 + "file.py"
    out = ov._fit(long, 600, ov._font(11))
    assert out.endswith("…")
    assert ov._font(11).size(out)[0] <= 600 * ov.scale
    ov.draw_tooltip(long, 790, 490)


def test_resize(ov):
    ov.resize((800, 500), 1.0)
    data, (w, h) = ov.to_bytes()
    assert (w, h) == (800, 500) and len(data) == w * h * 4
