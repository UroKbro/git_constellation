import numpy as np

from constellation.collapse import plan
from constellation.model import RepoHistory
from tests.history_helpers import A, c


def _hist(n_other: int):
    deep = [A(f"a/b/c/d/f{i}.py", 1) for i in range(300)]
    shallow = [A(f"a/b/g{i}.py", 1) for i in range(250)]
    other = [A(f"x/o{i}.py", 1) for i in range(n_other)]
    return RepoHistory.build([c(0, deep + shallow + other), c(1, [])])


def test_large_repo_collapses_deep_dir_only():
    h = _hist(10_001 - 550)
    p = plan(h, h.window())
    assert len(h.files) == 10_001
    d = h.dirs.index("a/b/c/d")
    assert p.group_dir == [d]
    deep = np.array([f.path.startswith("a/b/c/d/") for f in h.files])
    assert (p.member_to_group[deep] == 0).all()
    assert (p.member_to_group[~deep] == -1).all()
    assert p.member_to_group.dtype == np.int32
    assert p.expanded == set()
    p.expand(0)
    assert p.expanded == {0}


def test_small_repo_collapses_nothing():
    h = _hist(10_000 - 550)
    p = plan(h, h.window())
    assert p.group_dir == []
    assert (p.member_to_group == -1).all()


def test_union_counts_only_files_in_window():
    h = RepoHistory.build([c(0, [A(f"a/b/c/d/f{i}.py", 1) for i in range(300)]
                             + [A(f"x/o{i}.py", 1) for i in range(9_700)]),
                           c(1, [A("late.py", 1)])])
    from constellation.model import Window
    assert plan(h, Window(0, 0)).group_dir == []
    assert plan(h, Window(0, 1)).group_dir == [h.dirs.index("a/b/c/d")]
