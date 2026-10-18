import time

import numpy as np
import pytest

from constellation.layout import compute_layout
from constellation.metrics import CouplingPair
from constellation.model import RepoHistory
from tests.history_helpers import A, D, R, c


def _hist():
    return RepoHistory.build([
        c(0, [A("a/x.py", 5), A("b/y.py", 5), A("a/sub/z.py", 3), A("top.py", 1)]),
        c(1, [R("a/x.py", "b/x.py", 0, 0)]),
        c(2, [D("b/y.py", 5)]),
        c(3, [A("b/y.py", 2), A("c/d/e.py", 1)]),
    ])


def _arrays(L):
    return [L.dir_pos, L.dir_radius, L.place_file, L.place_dir, L.place_pos,
            L.file_place_start, L.place_commit]


def test_deterministic():
    h = _hist()
    l1, l2 = compute_layout(h, seed=3), compute_layout(h, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(_arrays(l1), _arrays(l2)))
    assert not np.array_equal(l1.place_pos, compute_layout(h, seed=4).place_pos)


def test_finite():
    h = RepoHistory.build([c(0, [A("a/b/c/f.py", 1)]), c(1, [D("a/b/c/f.py", 1)]),
                           c(2, [A("a/b/c/f.py", 1), R("a/b/c/f.py", "q/f.py", 0, 0)])])
    for hh in (_hist(), h, RepoHistory.build([])):
        for arr in _arrays(compute_layout(hh)):
            assert np.isfinite(arr).all()


def _own_nearest_fraction(h, L):
    """Fraction of placements nearer their own star than every unrelated star."""
    par = h.dir_parent
    n = len(h.dirs)
    related = np.zeros((n, n), dtype=bool)
    for d in range(n):
        a = d
        while a >= 0:
            related[d, a] = related[a, d] = True
            a = par[a]
    ok = 0
    for p in range(len(L.place_dir)):
        d = L.place_dir[p]
        dist = np.linalg.norm(L.dir_pos - L.place_pos[p], axis=1)
        ok += dist[d] < dist[~related[d]].min(initial=np.inf)
    return ok / max(1, len(L.place_dir))


def test_siblings_nearer_own_star():
    h = _hist()
    assert _own_nearest_fraction(h, compute_layout(h)) == 1.0


def test_own_star_nearest_high_fanout():
    rng = np.random.default_rng(1)
    changes = [A(f"tests/t{i}/f{j}.py", 1) for i in range(220)
               for j in range(int(rng.integers(2, 41)))]
    h = RepoHistory.build([c(0, changes)])
    n = len(h.files)
    pairs = []
    while len(pairs) < 50:
        a, b = (int(x) for x in rng.integers(0, n, 2))
        if h.files[a].paths[-1][1].split("/")[1] != h.files[b].paths[-1][1].split("/")[1]:
            pairs.append(CouplingPair(a, b, 3, 1.0))
    assert _own_nearest_fraction(h, compute_layout(h, pairs)) >= 0.98


def test_own_star_nearest_big_dir_with_subdirs():
    rng = np.random.default_rng(2)
    changes = [A(f"docs/releases/r{i}.md", 1) for i in range(600)]
    changes += [A(f"docs/releases/v{i}/f{j}.md", 1) for i in range(30)
                for j in range(int(rng.integers(1, 6)))]
    h = RepoHistory.build([c(0, changes)])
    assert _own_nearest_fraction(h, compute_layout(h)) >= 0.98


def test_rename_has_two_placements():
    h = _hist()
    L = compute_layout(h)
    fid = next(i for i, f in enumerate(h.files) if len(f.paths) == 2 and f.paths[0][1] == "a/x.py")
    s = int(L.file_place_start[fid])
    assert list(L.place_file[s:s + 2]) == [fid, fid]
    assert [h.dirs[d] for d in L.place_dir[s:s + 2]] == ["a", "b"]
    assert L.place_at(fid, 0) == s
    assert L.place_at(fid, 1) == s + 1
    assert L.place_at(fid, 3) == s + 1


def test_place_at_before_birth():
    h = RepoHistory.build([c(0, [A("a.py", 1)]), c(1, [A("b.py", 1)]),
                           c(2, [A("c.py", 1)]), c(3, [A("d/x.py", 1)])])
    L = compute_layout(h)
    fid = next(i for i, f in enumerate(h.files) if f.paths[0][1] == "d/x.py")
    assert L.place_at(fid, 0) == L.file_place_start[fid]


def test_many_couplings_stable():
    h = RepoHistory.build([c(0, [A(f"d{i % 7}/f{i}.py", 1) for i in range(301)])])
    pairs = [CouplingPair(0, i, 3, 1.0) for i in range(1, 301)]
    l60, l61 = compute_layout(h, pairs, iterations=60), compute_layout(h, pairs, iterations=61)
    assert np.isfinite(l60.place_pos).all()
    assert np.abs(l60.place_pos - l61.place_pos).max() < 1e-2


def test_coupling_pulls_pair_closer():
    h = RepoHistory.build([c(0, [A("a/x.py", 1), A("b/y.py", 1), A("a/p.py", 1), A("b/q.py", 1)])])
    base = compute_layout(h)
    pulled = compute_layout(h, [CouplingPair(0, 1, 5, 1.0)])
    d0 = np.linalg.norm(base.place_pos[0] - base.place_pos[1])
    d1 = np.linalg.norm(pulled.place_pos[0] - pulled.place_pos[1])
    assert d1 < d0
    for p in (0, 1):
        dd = pulled.place_dir[p]
        assert (np.linalg.norm(pulled.place_pos[p] - pulled.dir_pos[dd])
                <= 1.5 * pulled.dir_radius[dd] + 1e-4)


def test_bounds_contains_everything():
    L = compute_layout(_hist())
    ctr, r = L.bounds()
    for pts in (L.place_pos, L.dir_pos):
        assert (np.linalg.norm(pts - ctr, axis=1) <= r + 1e-4).all()
    ctr, r = compute_layout(RepoHistory.build([])).bounds()
    assert np.array_equal(ctr, np.zeros(3)) and r == 10.0
    ctr, r = L.bounds(np.zeros(len(L.place_pos), dtype=bool))
    assert np.array_equal(ctr, np.zeros(3)) and r == 10.0


@pytest.mark.slow
def test_layout_10k_under_2s():
    from constellation.types import FileChange
    changes = [FileChange("A", f"d{i % 8}/s{i % 400}/t{i % 3}/f{i}.py", None, 1, 0, False)
               for i in range(10000)]
    h = RepoHistory.build([c(0, changes)])
    pairs = [CouplingPair(i, i + 1, 3, 0.5) for i in range(0, 9000, 3)]
    t = time.perf_counter()
    compute_layout(h, pairs)
    assert time.perf_counter() - t < 2.0


@pytest.mark.slow
def test_scale_django_like():
    rng = np.random.default_rng(5)
    dirs = ["tests"]
    dirs += [f"tests/t{i}" for i in range(274)]
    pool = list(dirs)
    while len(dirs) < 5000:
        par = pool[int(rng.integers(len(pool)))]
        if par.count("/") < 9:
            d = f"{par}/n{len(dirs)}"
            dirs.append(d)
            pool.append(d)
    changes = [A(f"{dirs[int(rng.integers(len(dirs)))]}/f{i}.py", 1) for i in range(7000)]
    h = RepoHistory.build([c(0, changes)])
    L = compute_layout(h)
    _, radius = L.bounds()
    pos = L.dir_pos.astype(np.float64)
    nn = np.array([np.partition(np.linalg.norm(pos - q, axis=1), 1)[1] for q in pos])
    assert radius / np.median(nn) <= 50
    # measured: 0.76 synthetic, 0.83 on real Django; >= 98% is required only for the fanout/big-dir tests
    assert _own_nearest_fraction(h, L) >= 0.70
