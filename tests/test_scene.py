import dataclasses
import time

import numpy as np
import pytest

from constellation.collapse import plan
from constellation.layout import Layout, compute_layout
from constellation.model import RepoHistory, Window
from constellation.palette import language_color, lineage_color
from constellation.scene import Scene
from constellation.types import Commit
from tests.history_helpers import A, D, M, R, c

FAR = 1e6


def _hist():
    return RepoHistory.build([
        c(0, [A("a/x.py", 10), A("b/y.py", 5), A("a/z.py", 3)]),
        c(1, [M("a/x.py", 2, 1), A("c/new.py", 4)]),
        c(2, [R("a/z.py", "b/z.py", 0, 0), D("b/y.py", 5)]),
        c(3, [M("a/x.py", 1, 1), M("c/new.py", 1, 0)]),
        c(4, [M("a/x.py", 1, 0)]),
        c(5, [M("a/x.py", 1, 0)]),
    ])


def _scene(h, w=None):
    w = h.window() if w is None else w
    return Scene(h, compute_layout(h), plan(h, w), w)


def _fid(h, path):
    return next(i for i, f in enumerate(h.files) if f.path == path)


def test_planet_ids_match_alive_when_idle():
    h = _hist()
    s = _scene(h)
    for i in range(len(h.commits)):
        b = s.planets(i, FAR)
        assert np.array_equal(b.ids, np.flatnonzero(h.files_alive_at(i)))
        assert b.ids.dtype == np.int32 and b.pos.dtype == np.float32
        assert b.color.shape == (len(b.ids), 4)
        assert np.allclose(b.color[:, 3], 1.0)


def test_planet_radius():
    h = RepoHistory.build([c(0, [A("z.py", 0), A("h.py", 100), A("m.py", 10**6)])])
    b = _scene(h).planets(0, FAR)
    assert np.allclose(b.radius, [0.25, 1.5, 3.0])


def test_advance_spawns_pulses_and_comets():
    h = _hist()
    s = _scene(h)
    s.advance(0, 3, 10.0)
    assert len(s.pulses(10.0).born) == 6
    assert len(s.pulses(11.3).born) == 0
    cm = s.comets(10.0)
    assert cm.commit.tolist() == [1, 2, 3]
    assert np.array_equal(cm.color, cm.color2)
    assert np.allclose(cm.color[:, :3], lineage_color(0))
    assert len(s.comets(11.6).commit) == 0
    pb = s.pulses(10.0)
    assert ((pb.color[:, 3] >= 0.4) & (pb.color[:, 3] <= 1.0)).all()
    assert np.allclose(pb.life, 1.2)
    # comet: 4 units beyond the centroid of the touched files' stars
    L = s.layout
    stars = L.dir_pos[[h.dirs.index("a"), h.dirs.index("c")]].astype(np.float64)
    cen = stars.mean(axis=0)
    exp = cen + 4 * cen / np.linalg.norm(cen)
    assert np.allclose(cm.pos[2], exp, atol=1e-4)


@pytest.mark.parametrize("prev,cur", [(5, 2), (0, 1000)])
def test_jump_or_backwards_clears(prev, cur):
    h = _hist()
    s = _scene(h)
    s.advance(0, 3, 10.0)
    s.advance(prev, cur, 10.1)
    assert len(s.pulses(10.1).born) == 0
    assert len(s.comets(10.1).commit) == 0
    b = s.planets(3, 10.1)
    assert np.array_equal(b.ids, np.flatnonzero(h.files_alive_at(3)))
    assert np.allclose(b.color[:, 3], 1.0)


def test_same_playhead_is_noop():
    s = _scene(_hist())
    s.advance(0, 3, 10.0)
    s.advance(3, 3, 10.1)
    assert len(s.pulses(10.1).born) == 6


def test_pulse_cap():
    h = RepoHistory.build([c(0, [A("seed.py", 1)]),
                           c(1, [A(f"d{i % 50}/f{i}.py", 1) for i in range(25_000)])])
    s = Scene(h, compute_layout(h), plan(h, Window(0, 0)), h.window())
    s.advance(0, 1, 1.0)
    pb = s.pulses(1.0)
    assert len(pb.born) == 20_000


def test_add_flash():
    h = _hist()
    s = _scene(h)
    s.advance(0, 1, 5.0)
    f = _fid(h, "c/new.py")
    early = s.planets(1, 5.1)
    k = int(np.flatnonzero(early.ids == f)[0])
    idle = s.planets(1, FAR)
    assert early.radius[k] < idle.radius[k]
    assert early.color[k, 3] > 1.0
    assert s.planets(1, 5.7).color[k, 3] == pytest.approx(1.0)


def test_delete_fades_then_vanishes():
    h = _hist()
    s = _scene(h)
    s.advance(0, 3, 10.0)
    f = _fid(h, "b/y.py")
    mid = s.planets(3, 10.4)
    assert f in mid.ids
    k = int(np.flatnonzero(mid.ids == f)[0])
    assert 0 < mid.radius[k] < 0.15 * np.sqrt(5)
    assert (mid.color[k, :3] < np.array(s.file_rgb[f]) - 1e-3).all()
    assert f not in s.planets(3, 10.9).ids


def test_rename_slides():
    h = _hist()
    s = _scene(h)
    s.advance(1, 3, 20.0)
    f = _fid(h, "b/z.py")
    L = s.layout
    p0, p1 = L.place_pos[L.place_at(f, 1)], L.place_pos[L.place_at(f, 2)]
    assert not np.allclose(p0, p1)
    b = s.planets(3, 20.3)
    pos = b.pos[int(np.flatnonzero(b.ids == f)[0])]
    assert np.allclose(pos, (p0 + p1) / 2, atol=1e-4)
    b = s.planets(3, FAR)
    assert np.allclose(b.pos[int(np.flatnonzero(b.ids == f)[0])], p1)


def test_files_outside_window_never_appear():
    h = _hist()
    s = _scene(h, Window(3, 5))
    gone = _fid(h, "b/y.py")
    for i in range(len(h.commits)):
        assert gone not in s.planets(i, FAR).ids
    s.advance(0, 3, 1.0)
    assert gone not in s.planets(3, 1.4).ids


def test_merge_comet_braids():
    base = [c(0, [A("a.py", 1)]), c(1, [A("x/b.py", 1)], parents=("s0",)),
            c(2, [M("a.py", 1, 0)], parents=("s0",))]
    merge = dataclasses.replace(c(3, [], parents=("s2", "s1")),
                                merge_changes=(A("x/b.py", 1), D("old.py", 1)))
    h = RepoHistory.build(base + [merge])
    s = _scene(h)
    s.advance(2, 3, 1.0)
    cm = s.comets(1.0)
    assert cm.commit.tolist() == [3]
    assert not np.allclose(cm.color, cm.color2)
    assert np.allclose(cm.color2[0, :3], lineage_color(1))
    assert len(s.pulses(1.0).born) == 0
    # merge-only commit: centroid from the files its merge_changes resolve to
    star = s.layout.dir_pos[h.dirs.index("x")].astype(np.float64)
    exp = star + 4 * star / np.linalg.norm(star)
    assert np.allclose(cm.pos[0], exp, atol=1e-4)


def test_merge_comet_without_resolvable_files_points_up():
    base = [c(0, [A("a.py", 1)]), c(1, [A("b.py", 1)], parents=("s0",)),
            c(2, [M("a.py", 1, 0)], parents=("s0",))]
    merge = dataclasses.replace(c(3, [], parents=("s2", "s1")),
                                merge_changes=(D("zzz.py", 1),))
    s = _scene(RepoHistory.build(base + [merge]))
    s.advance(2, 3, 1.0)
    assert np.allclose(s.comets(1.0).pos[0], [0, 4, 0])


def test_stars():
    h = _hist()
    s = _scene(h)
    st = s.stars(0)
    alive_dirs = {"", "a", "b"}
    assert len(st.pos) == len(alive_dirs)
    # root holds all 3 alive files in its subtree
    assert np.isclose(st.radius.max(), 0.6 + 0.12 * np.sqrt(3))
    assert len(s.stars(1).pos) == 4


def test_aggregates_replace_collapsed_members():
    files = ([A(f"p/q/r/s/f{i}.py", 4) for i in range(300)]
             + [A(f"o/g{i}.md", 1) for i in range(9_701)])
    h = RepoHistory.build([c(0, files)])
    p = plan(h, h.window())
    assert len(p.group_dir) == 1
    L = compute_layout(h)
    s = Scene(h, L, p, h.window())
    b = s.planets(0, FAR)
    assert len(b.ids) == 9_702
    agg = int(np.flatnonzero(b.ids == -1)[0])
    assert np.isclose(b.radius[agg], 2 * min(3.0, 0.15 * np.sqrt(1200)))
    assert np.allclose(b.color[agg, :3], s.file_rgb[0])
    assert np.allclose(b.pos[agg], L.dir_pos[p.group_dir[0]])
    p.expand(0)
    b = s.planets(0, FAR)
    assert np.array_equal(b.ids, np.arange(len(h.files)))


def test_aggregate_colour_is_most_common_language():
    files = ([A(f"p/q/r/s/t{i}.ts", 1) for i in range(100)]
             + [A(f"p/q/r/s/x{i}.tsx", 1) for i in range(100)]
             + [A(f"p/q/r/s/m{i}.md", 1) for i in range(150)]
             + [A(f"o/g{i}.py", 1) for i in range(9_700)])
    h = RepoHistory.build([c(0, files)])
    s = _scene(h)
    b = s.planets(0, FAR)
    agg = int(np.flatnonzero(b.ids == -1)[0])
    assert np.allclose(b.color[agg, :3], language_color("a.ts"))


@pytest.mark.slow
def test_mass_delete_advance_is_fast():
    paths = [f"d{i % 40}/f{i}.py" for i in range(20_000)]
    h = RepoHistory.build([c(0, [A(p, 3) for p in paths]), c(1, [D(p, 3) for p in paths])])
    s = _scene(h)
    t0 = time.perf_counter()
    s.advance(0, 1, 1.0)
    dt = time.perf_counter() - t0
    assert dt < 0.020, f"{dt * 1000:.1f} ms"
    assert len(s.planets(1, 1.4).ids) == 20_000
    assert len(s.planets(1, 1.9).ids) == 0


def _synthetic(n_files=10_500, n_commits=50_000, seed=0):
    rng = np.random.default_rng(seed)
    paths = [f"big/a/b/c{i % 10}/f{i}.py" if i < 5_000 else f"d{i % 20}/e{i % 7}/f{i}.ts"
             for i in range(n_files)]
    commits = [c(0, [A(p, 50) for p in paths])]
    touched = rng.integers(0, n_files, size=(n_commits, 3))
    for i in range(1, n_commits):
        commits.append(Commit(sha=f"s{i}", parents=(f"s{i-1}",), author="Ana",
                              email="a@x", ts=1_600_000_000 + i, subject="",
                              changes=tuple(M(paths[j], 2, 1) for j in set(touched[i]))))
    return RepoHistory.build(commits)


@pytest.mark.slow
def test_scene_frame_budget():
    h = _synthetic()
    w = h.window()
    s = Scene(h, compute_layout(h), plan(h, w), w)
    now = 0.0
    for k in range(64):
        s.advance(k * 64, (k + 1) * 64, now)
        now += 1 / 60
    ph = 64 * 64
    t0 = time.perf_counter()
    for _ in range(100):
        s.advance(ph, ph + 64, now)
        ph += 64
        s.planets(ph, now)
        s.stars(ph)
        s.pulses(now)
        s.comets(now)
        now += 1 / 600
    mean = (time.perf_counter() - t0) / 100
    assert mean < 0.004, f"{mean * 1000:.2f} ms"
