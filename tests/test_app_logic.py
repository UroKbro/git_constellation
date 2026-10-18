import numpy as np
import pytest

from constellation.app import (BloomSwitch, ClickDetector, InputState, fit_camera,
                               pick_object, project)
from constellation.camera import OrbitCamera
from constellation.model import Window
from constellation.scene import CometBatch, PlanetBatch
from constellation.timeline import Timeline


def _timeline(n=10):
    return Timeline(Window(0, n - 1), np.arange(n, dtype=np.int64) * 100)


def _track(x, y):  # fake timeline: bottom strip y >= 700, x in [100, 1100]
    return (x - 100) / 1000 if y >= 700 and 100 <= x <= 1100 else None


@pytest.fixture
def ui():
    tl = _timeline()
    cam = OrbitCamera()
    return InputState(tl, cam, timeline_hit=_track), tl, cam


def test_space_toggles_playback(ui):
    s, tl, _ = ui
    tl.home()
    s.handle(("key", "space"))
    assert tl.playing
    s.handle(("key", "space"))
    assert not tl.playing


def test_right_steps_and_pauses(ui):
    s, tl, _ = ui
    tl.home()
    tl.playing = True
    s.handle(("key", "right"))
    assert tl.playhead == 1 and not tl.playing
    s.handle(("key", "left"))
    assert tl.playhead == 0


def test_home_end(ui):
    s, tl, _ = ui
    s.handle(("key", "home"))
    assert tl.playhead == 0
    s.handle(("key", "end"))
    assert tl.playhead == 9


def test_plus_doubles_speed_minus_halves(ui):
    s, tl, _ = ui
    s.handle(("key", "+"))
    assert tl.speed == 2
    s.handle(("key", "-"))
    s.handle(("key", "-"))
    assert tl.speed == 0.5


def test_click_inside_timeline_scrubs(ui):
    s, tl, _ = ui
    assert s.handle(("click", 100 + 500, 750)) == []
    assert tl.playhead == 4  # times 0..900, 50% = 450 -> commit 4
    assert s.handle(("click", 100, 750)) == [] and tl.playhead == 0


def test_click_in_scene_selects_and_dblclick_activates(ui):
    s, _, _ = ui
    assert s.handle(("click", 300, 200)) == [("select", 300, 200)]
    assert s.handle(("dblclick", 300, 200)) == [("activate", 300, 200)]


def test_i_toggles_panel_f3_toggles_fps_escape_quits(ui):
    s, _, _ = ui
    panel = s.panel
    s.handle(("key", "i"))
    assert s.panel is not panel
    assert not s.show_fps
    s.handle(("key", "f3"))
    assert s.show_fps
    assert not s.quit
    s.handle(("key", "escape"))
    assert s.quit


def test_later_phase_keys_are_ignored(ui):
    s, tl, cam = ui
    before = (tl.playhead, tl.playing, tl.speed, s.panel, s.show_fps, cam.yaw, cam.distance)
    for k in ("1", "2", "3", "4", "5", "/", "a", "t", "c", "f12"):
        assert s.handle(("key", k)) == []
    assert (tl.playhead, tl.playing, tl.speed, s.panel, s.show_fps, cam.yaw,
            cam.distance) == before and not s.quit


def test_left_drag_orbits_right_drag_pans_wheel_zooms(ui):
    s, _, cam = ui
    yaw, target, dist = cam.yaw, cam.target.copy(), cam.distance
    s.handle(("down", 1, 300, 300, 0.0))
    s.handle(("motion", 340, 300, 40, 0))
    assert s.handle(("up", 1, 340, 300, 0.1)) == []  # a drag is not a click
    assert cam.yaw != yaw
    s.handle(("down", 3, 300, 300, 1.0))
    s.handle(("motion", 320, 310, 20, 10))
    s.handle(("up", 3, 320, 310, 1.1))
    assert not np.allclose(cam.target, target)
    s.handle(("wheel", 2))
    assert cam.distance < dist


def test_press_release_in_place_is_click_then_double_click(ui):
    s, _, _ = ui
    s.handle(("down", 1, 300, 300, 0.0))
    assert s.handle(("up", 1, 301, 300, 0.05)) == [("select", 301, 300)]
    s.handle(("down", 1, 302, 301, 0.2))
    assert s.handle(("up", 1, 302, 301, 0.25)) == [("activate", 302, 301)]


def test_drag_on_timeline_scrubs(ui):
    s, tl, cam = ui
    yaw = cam.yaw
    s.handle(("down", 1, 100, 750, 0.0))
    assert tl.playhead == 0
    s.handle(("motion", 1100, 750, 1000, 0))
    assert tl.playhead == 9 and cam.yaw == yaw
    s.handle(("up", 1, 1100, 750, 0.1))


def test_click_detector_time_and_distance():
    d = ClickDetector()
    assert d.feed(10, 10, 0.0) == "click"
    assert d.feed(13, 10, 0.3) == "dblclick"
    assert d.feed(13, 10, 0.4) == "click"      # a third click starts over
    assert d.feed(13, 10, 1.0) == "click"      # too slow
    assert d.feed(20, 10, 1.1) == "click"      # too far (7 pt)


def test_bloom_switch_once_on_slow_rolling_mean():
    b = BloomSwitch()
    assert not any(b.add(25.0) for _ in range(59))
    assert b.add(25.0)
    assert not any(b.add(40.0) for _ in range(100))
    fast = BloomSwitch()
    assert not any(fast.add(10.0) for _ in range(200))
    mixed = BloomSwitch()
    for _ in range(200):
        assert not mixed.add(30.0 if _ % 2 else 5.0)  # mean 17.5 ms


@pytest.mark.parametrize("seed,aspect", [(0, 1.6), (1, 1.0), (2, 0.6), (3, 2.4)])
def test_fit_camera_fills_85_percent_and_centres(seed, aspect):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(500, 3)) * np.array([40.0, 4.0, 12.0]) + np.array([30.0, -5.0, 8.0])
    cam = OrbitCamera()
    c = (pts.min(0) + pts.max(0)) / 2
    cam.frame(c, float(np.linalg.norm(pts - c, axis=1).max()), aspect=aspect)
    fit_camera(cam, pts, aspect)
    xy = project(cam, pts, aspect)
    lo, hi = xy.min(0), xy.max(0)
    span = float(((hi - lo) / 2).max())
    assert 0.75 <= span <= 0.90
    assert np.all(np.abs((lo + hi) / 2) <= 0.05)
    assert np.all(xy >= -1) and np.all(xy <= 1)


def test_fit_camera_into_a_sub_region():
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(400, 3)) * np.array([20.0, 6.0, 9.0])
    cam = OrbitCamera()
    cam.frame((0, 0, 0), float(np.linalg.norm(pts, axis=1).max()), aspect=1.6)
    region = (-0.28, 0.08, 0.72, 0.92)  # left of a side panel, above a bottom bar
    fit_camera(cam, pts, 1.6, region=region)
    xy = project(cam, pts, 1.6)
    lo, hi = xy.min(0), xy.max(0)
    span = ((hi - lo) / 2 / np.array(region[2:])).max()
    assert 0.75 <= span <= 0.90
    assert np.all(np.abs((lo + hi) / 2 - np.array(region[:2])) <= 0.05)


def test_fit_camera_degenerate_inputs_keep_camera_finite():
    cam = OrbitCamera()
    cam.frame((0, 0, 0), 10.0)
    fit_camera(cam, np.zeros((0, 3)), 1.6)
    fit_camera(cam, np.array([[1.0, 2.0, 3.0]]), 1.6)
    assert np.isfinite(cam.distance) and np.all(np.isfinite(cam.target))


def _empty_comets():
    z = np.zeros((0, 4), "f4")
    return CometBatch(np.zeros((0, 3), "f4"), z, z, np.zeros(0, np.int32),
                      np.zeros(0, "f4"), np.zeros(0, "f4"))


def test_comet_wins_over_planet_on_same_ray():
    cam = OrbitCamera()
    cam.frame((0, 0, 0), 20.0, aspect=1.0)
    eye = cam.eye()
    fwd = -eye / np.linalg.norm(eye)
    planets = PlanetBatch(np.array([[0, 0, 0]], "f4"), np.array([2.0], "f4"),
                          np.ones((1, 4), "f4"), np.array([7], np.int32))
    comet_pos = (eye + fwd * (np.linalg.norm(eye) + 10)).astype("f4")[None]  # behind the planet
    one = np.ones((1, 4), "f4")
    comets = CometBatch(comet_pos, one, one, np.array([42], np.int32),
                        np.zeros(1, "f4"), np.ones(1, "f4"))
    assert pick_object(cam, 400, 400, (800, 800), planets, comets) == ("comet", 42)
    assert pick_object(cam, 400, 400, (800, 800), planets, _empty_comets()) == ("planet", 7)
    assert pick_object(cam, 5, 5, (800, 800), planets, _empty_comets()) is None


def test_pick_uses_drawable_pixels():
    cam = OrbitCamera()
    cam.frame((0, 0, 0), 20.0, aspect=1.6)
    planets = PlanetBatch(np.array([[0, 0, 0]], "f4"), np.array([1.0], "f4"),
                          np.ones((1, 4), "f4"), np.array([3], np.int32))
    # centre of a 1600x1000 drawable (Retina) == point (400, 250) x 2
    assert pick_object(cam, 800, 500, (1600, 1000), planets, _empty_comets()) == ("planet", 3)


def _wait_for(fn, timeout=5.0):
    import time
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.005)
    raise AssertionError("timed out")


def test_detail_worker_never_blocks_and_latest_wins():
    import threading
    import time
    from constellation.app import DetailWorker
    w = DetailWorker()
    gate, started = threading.Event(), threading.Event()

    def slow():
        started.set()
        gate.wait(5)
        return ["old"]
    t = time.perf_counter()
    assert w.get(("f", 1), slow) is None
    assert w.get(("f", 1), slow) is None          # same key: not re-queued
    assert started.wait(5)
    assert w.get(("f", 2), lambda: ["new"]) is None
    assert time.perf_counter() - t < 0.05
    gate.set()
    assert _wait_for(lambda: w.get(("f", 2), lambda: ["new"])) == ["new"]
    assert _wait_for(lambda: w.discarded or None) == 1   # ("f", 1) finished stale
    assert w.latest() == (("f", 2), ["new"])


def _loaded():
    from types import SimpleNamespace
    from pathlib import Path
    from constellation.collapse import plan
    from constellation.model import RepoHistory
    from tests.history_helpers import A, M, c
    h = RepoHistory.build([c(0, [A("a/x.py", 10)]), c(1, [M("a/x.py", 2, 1)])])
    w = h.window()
    return SimpleNamespace(root=Path("/tmp/repo"), h=h, window=w, plan=plan(h, w),
                           coupling=None)


def test_panel_shows_pending_path_then_result_and_drops_stale():
    import threading
    import time
    from constellation.app import PENDING, DetailCache
    gate, started = threading.Event(), threading.Event()
    calls = []

    def fake_details(h, w, playhead, fid, coupling):
        calls.append(playhead)
        if playhead == 0:
            started.set()
            gate.wait(5)
        return [f"details@{playhead}"]
    cache = DetailCache(_loaded(), details_fn=fake_details)
    sel = ("planet", 0)
    t = time.perf_counter()
    assert cache.panel(sel, 0) == ["a/x.py", PENDING]
    assert cache.panel(sel, 0) is cache.panel(sel, 0)  # stable while pending (no redraws)
    assert time.perf_counter() - t < 0.05
    assert started.wait(5)
    assert cache.panel(sel, 1) == ["a/x.py", PENDING]  # playhead moved: playhead 0 is stale
    gate.set()
    assert _wait_for(lambda: cache.panel(sel, 1) if cache.panel(sel, 1)[0] != "a/x.py"
                     else None) == ["details@1"]
    assert _wait_for(lambda: cache.worker.discarded or None) == 1
    assert cache.panel(sel, 1) == ["details@1"]
    # during playback the last lines for the file are kept, nothing is queued
    n = len(calls)
    assert cache.panel(sel, 0, stale_ok=True) == ["details@1"] and len(calls) == n
