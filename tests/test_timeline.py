import numpy as np

from constellation.model import Window
from constellation.timeline import Timeline

DAY = 86400


def _tl(n=100, start=0, end=None):
    ts = np.arange(n, dtype=np.int64) * DAY
    return Timeline(Window(start, n - 1 if end is None else end), ts)


def test_initial_state():
    tl = _tl()
    assert tl.cursor == 99 and tl.playhead == 99
    assert not tl.playing and tl.speed == 1.0


def test_one_x_for_one_second_advances_four_commits():
    tl = _tl()
    tl.home()
    tl.toggle()
    prev, new = tl.update(1.0)
    assert (prev, new) == (0, 4)


def test_play_at_end_restarts_from_start():
    tl = _tl(start=10)
    tl.toggle()
    assert tl.playing and tl.playhead == 10


def test_fast_speed_clamps_and_pauses():
    tl = _tl(20)
    tl.home()
    for _ in range(10):
        tl.faster()
    assert tl.speed == 64
    tl.toggle()
    prev, new = tl.update(10.0)
    assert new == 19 and not tl.playing and tl.cursor == 19


def test_speed_bounds():
    tl = _tl()
    for _ in range(20):
        tl.slower()
    assert tl.speed == 0.25
    for _ in range(20):
        tl.faster()
    assert tl.speed == 64
    tl.slower()
    assert tl.speed == 32


def test_step_pauses_and_clamps():
    tl = _tl(10)
    tl.home()
    tl.toggle()
    tl.step(-1)
    assert tl.playhead == 0 and not tl.playing
    tl.step(3)
    assert tl.playhead == 3
    tl.step(100)
    assert tl.playhead == 9


def test_scrub_by_time_uses_running_max():
    ts = np.array([0, 1, 2, 10], dtype=np.int64) * DAY
    tl = Timeline(Window(0, 3), ts)
    tl.scrub(0.5)
    assert tl.playhead == 2
    tl.scrub(0.0)
    assert tl.playhead == 0
    tl.scrub(1.0)
    assert tl.playhead == 3
    assert 0.0 <= tl.fraction_of(2) <= 1.0


def test_scrub_non_monotonic():
    ts = np.array([0, 10, 3, 4], dtype=np.int64)
    tl = Timeline(Window(0, 3), ts)
    tl.scrub(0.5)
    assert tl.playhead == 0
    tl.scrub(1.0)
    assert tl.playhead == 3


def test_scrub_respects_window():
    tl = _tl(10, start=3, end=6)
    tl.scrub(0.0)
    assert tl.playhead == 3
    tl.scrub(1.0)
    assert tl.playhead == 6


def test_histogram_sums_to_window_size():
    ts = np.array([0, 1, 2, 10, 11, 50], dtype=np.int64)
    tl = Timeline(Window(1, 5), ts)
    h = tl.histogram(8)
    assert h.shape == (8,) and h.sum() == 5


def test_histogram_same_timestamp_single_bin():
    tl = Timeline(Window(0, 4), np.full(5, 7, dtype=np.int64))
    h = tl.histogram(4)
    assert h.sum() == 5 and (h > 0).sum() == 1


def test_empty_window_with_commits():
    tl = Timeline(Window(0, -1), np.arange(3, dtype=np.int64))
    assert tl.playhead == 0
    tl.toggle()
    assert not tl.playing
    p = tl.playhead
    assert tl.update(1.0) == (p, p)
    assert tl.histogram(5).tolist() == [0] * 5
    tl.step(2); tl.home(); tl.end(); tl.scrub(0.5)
    assert tl.fraction_of(0) == 0.0


def test_empty_window_zero_commits():
    tl = Timeline(Window(0, -1), np.zeros(0))
    assert tl.playhead == 0
    tl.toggle()
    assert tl.update(1.0) == (0, 0)
    assert tl.histogram(3).sum() == 0
