import pytest

from constellation.metrics.ownership import bus_factor_risk, ownership
from constellation.model import RepoHistory
from tests.history_helpers import A, M, c


def test_share_by_churn():
    h = RepoHistory.build([c(0, [A("f", 8)], author="Ana"), c(1, [M("f", 2, 0)], author="Bo")])
    o = ownership(h, 0, 1)
    assert h.authors[o.dominant[0]] == "Ana" and o.share[0] == pytest.approx(0.8)
    assert [(h.authors[a], s) for a, s in o.top3[0]] == [("Ana", pytest.approx(0.8)), ("Bo", pytest.approx(0.2))]


def test_zero_churn_falls_back_to_event_count_and_ties_lowest_id():
    h = RepoHistory.build([c(0, [A("f", 0)], author="Ana"), c(1, [M("f", 0, 0)], author="Bo"),
                           c(2, [M("f", 0, 0)], author="Bo")])
    o = ownership(h, 0, 2)
    assert h.authors[o.dominant[0]] == "Bo" and o.share[0] == pytest.approx(2 / 3)
    o = ownership(h, 0, 1)
    assert h.authors[o.dominant[0]] == "Ana" and o.share[0] == pytest.approx(0.5)


def test_empty_window():
    h = RepoHistory.build([c(0, [A("f", 3)])])
    o = ownership(h, 1, 0)
    assert o.dominant.tolist() == [-1] and o.share.tolist() == [0.0] and o.top3 == [[]]
    assert bus_factor_risk(h, 1, 0).tolist() == [False]


def _quiet_owner(n_ana):
    cs = [c(i, [A("f", 1)] if i == 0 else [M("f", 1, 0)], author="Ana", day=i) for i in range(n_ana)]
    cs.append(c(n_ana, [A("g", 1)], author="Bo", day=400))
    return RepoHistory.build(cs)


def test_bus_factor_risk_when_owner_went_quiet():
    h = _quiet_owner(10)
    assert bus_factor_risk(h, 0, 10).tolist() == [True, False]
    assert bus_factor_risk(h, 0, 9).tolist()[0] is False  # Ana still active at day 9


def test_no_risk_below_10_changes():
    h = _quiet_owner(9)
    assert bus_factor_risk(h, 0, 9).tolist() == [False, False]


@pytest.mark.slow
def test_perf_ownership_10k_files():
    import time

    import numpy as np

    rng = np.random.default_rng(0)
    seen: set[int] = set()
    cs = []
    for i in range(50_000):
        changes = []
        for f in rng.choice(10_000, size=3, replace=False).tolist():
            changes.append(M(f"f{f}", 2, 1) if f in seen else A(f"f{f}", 5))
            seen.add(f)
        cs.append(c(i, changes, author=f"a{i % 20}"))
    h = RepoHistory.build(cs)
    t = time.perf_counter()
    ownership(h, 0, 49_999)
    bus_factor_risk(h, 0, 49_999)
    assert time.perf_counter() - t < 0.1
