import random
from collections import Counter
from itertools import combinations

import pytest

from constellation.metrics.coupling import CouplingIndex
from constellation.model import RepoHistory
from tests.history_helpers import c, A, M


def brute_force_shared(h, upto):
    per_commit: dict[int, set[int]] = {}
    for f, cm in zip(h._ev_file.tolist(), h._ev_commit.tolist()):
        if cm <= upto:
            per_commit.setdefault(cm, set()).add(f)
    out: Counter = Counter()
    for files in per_commit.values():
        if 2 <= len(files) <= 50:
            out.update(combinations(sorted(files), 2))
    return out


def test_strength_and_giant_commit_ignored():
    cs = [c(0, [A("a", 1), A("b", 1)])]
    cs += [c(i, [M("a", 1, 0), M("b", 1, 0)]) for i in range(1, 5)]      # shared = 5 incl. creation
    cs.append(c(5, [M("a", 1, 0)]))                                     # a alone
    cs.append(c(6, [M("a", 1, 0), M("b", 1, 0)] + [A(f"z{k}", 1) for k in range(49)]))  # 51 files
    h = RepoHistory.build(cs)
    (p,) = CouplingIndex(h).pairs(0, 6)
    assert (p.a, p.b, p.shared) == (0, 1, 5)
    assert p.strength == pytest.approx(5 / 6)       # min(changes a=7, b=6)


def test_snapshots_match_brute_force():
    rng = random.Random(1234)
    paths = [f"f{k}" for k in range(40)]
    cs = [c(0, [A(p, 1) for p in paths[:3]])]
    seen = set(paths[:3])
    for i in range(1, 1200):
        chosen = rng.sample(paths, rng.randint(1, 6))
        cs.append(c(i, [M(p, 1, 0) if p in seen else A(p, 1) for p in chosen]))
        seen.update(chosen)
    h = RepoHistory.build(cs)
    idx = CouplingIndex(h, snapshot_every=100)
    for upto in (0, 99, 100, 101, 777, 1199):
        assert idx.shared_counts(upto) == brute_force_shared(h, upto)


def test_window_subtracts_earlier_pairs():
    cs = [c(0, [A("a", 1), A("b", 1)])]
    cs += [c(i, [M("a", 1, 0), M("b", 1, 0)]) for i in range(1, 5)]
    h = RepoHistory.build(cs)
    idx = CouplingIndex(h)
    assert len(idx.pairs(0, 4)) == 1
    assert idx.pairs(3, 4) == []                    # shared 2 < min_shared 5
    (p,) = idx.pairs(3, 4, min_shared=2)
    assert (p.shared, p.strength) == (2, 1.0)


def test_empty_window():
    h = RepoHistory.build([c(0, [A("a", 1), A("b", 1)])])
    idx = CouplingIndex(h)
    assert idx.pairs(3, 2) == []
    assert idx.shared_counts(-1) == Counter()


def test_snapshot_count_is_capped():
    cs = [c(i, [A(f"f{i % 7}", 1) if i < 7 else M(f"f{i % 7}", 1, 0),
                A(f"g{i % 5}", 1) if i < 5 else M(f"g{i % 5}", 1, 0)])
          for i in range(5000)]
    h = RepoHistory.build(cs)
    idx = CouplingIndex(h, snapshot_every=10)
    assert len(idx._snapshots) <= 20
    assert idx.shared_counts(4321) == brute_force_shared(h, 4321)
