import time
from datetime import datetime

import pytest

from constellation.insights import Finding, findings
from constellation.model import RepoHistory
from tests.history_helpers import A, M, c
from tests.repo_builder import DAY, T0


def _history():
    cs = [c(0, [A("old.txt", 7), A("legacy/core.py", 10)], author="Ana", day=0)]
    for i in range(1, 10):
        cs.append(c(i, [M("legacy/core.py", 1, 0)], author="Ana", day=i))
    for i in range(10, 15):  # cross-folder pair
        ch = [A("x/a.py", 3), A("y/b.py", 4)] if i == 10 else [M("x/a.py", 1, 0), M("y/b.py", 1, 0)]
        cs.append(c(i, ch, author="Bo", day=700 + i))
    for i in range(15, 20):  # same-folder pair
        ch = [A("x/c.py", 3), A("x/d.py", 4)] if i == 15 else [M("x/c.py", 1, 0), M("x/d.py", 1, 0)]
        cs.append(c(i, ch, author="Bo", day=700 + i))
    return RepoHistory.build(cs)


def _by_kind(fs, kind):
    return [f for f in fs if f.kind == kind]


def test_empty_history():
    h = RepoHistory.build([])
    assert findings(h, h.window(), -1) == []


def _local_day(ts: int) -> str:
    return f"{datetime.fromtimestamp(ts):%Y-%m-%d}"


def test_summary_exact():
    h = _history()
    fs = findings(h, h.window(), 19)
    assert fs[0] == Finding(
        "summary", "20 commits by 2 authors, 6 files",
        f"From {_local_day(T0)} to {_local_day(T0 + 719 * DAY)}.", ())


@pytest.fixture
def far_east_tz(monkeypatch):
    """UTC+14: T0 (22:13 UTC) is already the next calendar day locally."""
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_summary_dates_are_local(far_east_tz):
    h = _history()
    assert findings(h, h.window(), 19)[0].detail == "From 2023-11-15 to 2025-11-03."


def test_empty_window_summary():
    h = _history()
    fs = findings(h, h.window(since=10**12), 19)
    assert fs[0].kind == "summary" and fs[0].detail == "No commits in the selected range."
    assert fs[0].title == "0 commits by 0 authors, 6 files"


def test_kind_order():
    h = _history()
    kinds = [f.kind for f in findings(h, h.window(), 19)]
    assert kinds[0] == "summary"
    order = ["summary", "hotspot", "coupling", "bus_factor", "fossil"]
    assert kinds == sorted(kinds, key=order.index)
    assert set(kinds) == set(order)


def test_coupling_exact_and_same_dir_excluded():
    h = _history()
    cp = _by_kind(findings(h, h.window(), 19), "coupling")
    assert len(cp) == 1
    assert cp[0].title == "x/a.py ↔ y/b.py"
    assert cp[0].detail == "Change together 100% of the time (5 commits) across different folders."
    assert cp[0].file_ids == (h.file_id("x/a.py"), h.file_id("y/b.py"))


def test_bus_factor_exact():
    h = _history()
    bf = _by_kind(findings(h, h.window(), 19), "bus_factor")
    assert [f.title for f in bf] == ["legacy/core.py depends on Ana"]
    assert bf[0].detail == "Ana wrote 100% of it and hasn't committed in 710 days."
    assert bf[0].file_ids == (h.file_id("legacy/core.py"),)


def test_fossil_exact():
    h = _history()
    fo = _by_kind(findings(h, h.window(), 19), "fossil")
    assert [f.title for f in fo] == ["legacy/core.py is a fossil", "old.txt is a fossil"]
    assert fo[1].detail == "Untouched for 719 days (7 lines)."


def test_hotspot_exact():
    cs = [c(0, [A("a.py", 8)], author="Ana", day=0), c(1, [M("a.py", 2, 0)], author="Bo", day=1),
          c(2, [A("b.py", 1)], author="Ana", day=2)]
    h = RepoHistory.build(cs)
    hs = _by_kind(findings(h, h.window(), 2), "hotspot")
    assert hs[0] == Finding("hotspot", "a.py is a hotspot",
                            "Changed 2 times, 10 lines, mostly by Ana (80%).", (0,))
    assert [f.title for f in hs] == ["a.py is a hotspot", "b.py is a hotspot"]


def test_hotspot_cap_10_with_12_hot_files():
    cs = [c(0, [A(f"f{i:02d}.py", 5) for i in range(12)], day=0),
          c(1, [M(f"f{i:02d}.py", 1, 0) for i in range(12)], day=1)]
    h = RepoHistory.build(cs)
    hs = _by_kind(findings(h, h.window(), 1), "hotspot")
    assert len(hs) == 10
    assert [f.title for f in hs] == [f"f{i:02d}.py is a hotspot" for i in range(10)]


def test_coupling_skips_dead_files():
    from constellation.types import FileChange
    cs = [c(0, [A("x/a.py", 3), A("y/b.py", 3)], day=0)]
    cs += [c(i, [M("x/a.py", 1, 0), M("y/b.py", 1, 0)], day=i) for i in range(1, 5)]
    cs.append(c(5, [FileChange("D", "y/b.py", None, 0, 3, False), M("x/a.py", 1, 0)], day=5))
    cs.append(c(6, [M("x/a.py", 1, 0)], day=6))
    h = RepoHistory.build(cs)
    before = _by_kind(findings(h, h.window(), 4), "coupling")
    assert [f.title for f in before] == ["x/a.py ↔ y/b.py"]
    for ph in (5, 6):
        assert _by_kind(findings(h, h.window(), ph), "coupling") == []
