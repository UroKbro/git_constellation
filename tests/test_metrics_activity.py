import time

import numpy as np
import pytest

from constellation.metrics import (FOSSIL_DAYS, HALF_LIFE_DAYS, change_counts,
                                   fossil_age_days, hotspot_raw, hotspots)
from constellation.model import RepoHistory, Window
from tests.history_helpers import A, D, M, c


def test_constants():
    assert FOSSIL_DAYS == 365 and HALF_LIFE_DAYS == 180


def test_hotspot_raw_exact():
    h = RepoHistory.build([c(0, [A("a", 2)], day=0), c(1, [M("a", 1, 0)], day=180)])
    assert hotspot_raw(h, 0, 1).tolist() == pytest.approx([1.5 * 2.0])


def test_hotspots_normalized_to_window_end_and_binary_excluded():
    h = RepoHistory.build([c(0, [A("a", 1), A("img", 0, binary=True)]),
                           c(1, [M("a", 1, 0)]), c(2, [M("a", 1, 0)])])
    w = h.window()
    assert hotspots(h, w, 2)[0] == 1.0
    assert 0 < hotspots(h, w, 0)[0] < 1.0
    assert hotspots(h, w, 2)[1] == 0.0


def test_change_counts_respect_window():
    h = RepoHistory.build([c(i, [M("a", 1, 0)] if i else [A("a", 1)]) for i in range(4)])
    assert change_counts(h, 1, 2).tolist() == [2]


def test_fossil_age():
    h = RepoHistory.build([c(0, [A("a", 1), A("b", 1)], day=0),
                           c(1, [M("b", 1, 0)], day=400), c(2, [D("b", 2)], day=401)])
    ages = fossil_age_days(h, 1)
    assert ages[0] == 400 and ages[1] == 0
    assert np.isnan(fossil_age_days(h, 2)[1])


def test_fossil_age_uses_latest_of_many_events():
    h = RepoHistory.build([c(0, [A("a", 1), A("b", 1)], day=0), c(1, [M("a", 1, 0)], day=10),
                           c(2, [M("a", 1, 0)], day=20), c(3, [M("b", 1, 0)], day=50)])
    ages = fossil_age_days(h, 3)
    assert ages[0] == 30 and ages[1] == 0


def test_empty_window_and_negative_age_clamp():
    h = RepoHistory.build([c(0, [A("a", 1)], day=10), c(1, [M("a", 1, 0)], day=5)])
    assert change_counts(h, 2, 1).tolist() == [0]
    assert hotspot_raw(h, 2, 1).tolist() == [0.0]
    assert hotspots(h, Window(2, 1), 1).tolist() == [0.0]
    # event at day 10 is "in the future" of playhead (day 5): age clamps to 0
    assert hotspot_raw(h, 0, 1).tolist() == pytest.approx([2.0 * np.log2(1 + 2)])
    assert fossil_age_days(h, 1)[0] == 0


@pytest.mark.slow
def test_perf_10k_files_under_100ms():
    rng = np.random.default_rng(0)
    seen: set[int] = set()
    commits = []
    for i in range(50_000):
        ch = []
        for f in rng.choice(10_000, 3, replace=False).tolist():
            ch.append(M(f"d{f % 50}/f{f}.py", 1, 0) if f in seen else A(f"d{f % 50}/f{f}.py", 5))
            seen.add(f)
        commits.append(c(i, ch, day=i / 20))
    h = RepoHistory.build(commits)
    last = len(commits) - 1
    t = time.perf_counter()
    out = hotspots(h, h.window(), last)
    assert time.perf_counter() - t < 0.1
    t = time.perf_counter()
    ages = fossil_age_days(h, last)
    assert time.perf_counter() - t < 0.1
    assert ages.shape == (len(h.files),)
    assert out.shape == (len(h.files),) and out.max() == 1.0
