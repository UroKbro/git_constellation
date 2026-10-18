import subprocess

from constellation.git_reader import iter_commits
from constellation.model import RepoHistory, Window
from tests.history_helpers import c
from tests.repo_builder import T0, DAY


def test_feature_branch_gets_merge_lineage(repo):
    repo.commit("m0", {"a": "1\n"}, day=0)
    repo.checkout("f", create=True)
    repo.commit("f1", {"b": "1\n"}, day=1)
    repo.commit("f2", {"b": "2\n"}, day=2)
    repo.checkout("main")
    repo.commit("m1", {"a": "2\n"}, day=3)
    repo.merge("f", "merge f", day=4)
    h = RepoHistory.build(list(iter_commits(repo.path)))
    by = {cm.subject: int(l) for cm, l in zip(h.commits, h.lineage)}
    assert by == {"m0": 0, "f1": 1, "f2": 1, "m1": 0, "merge f": 0}


def test_unrelated_histories(repo):
    repo.commit("a0", {"a": "1\n"}, day=0)
    repo.orphan("other")
    repo.commit("o0", {"o": "1\n"}, day=1)
    repo.checkout("main")
    repo.merge("other", "join", day=2, unrelated=True)
    h = RepoHistory.build(list(iter_commits(repo.path)))
    assert sum(1 for cm in h.commits if not cm.parents) == 2
    assert {cm.subject: int(l) for cm, l in zip(h.commits, h.lineage)}["o0"] == 1


def test_shallow_clone_does_not_crash(repo, tmp_path):
    for d in range(5):
        repo.commit(f"c{d}", {"a": f"{d}\n"}, day=d)
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "2", f"file://{repo.path}", shallow],
                   check=True)
    h = RepoHistory.build(list(iter_commits(shallow)))
    assert len(h.commits) == 2 and h.lineage.tolist() == [0, 0]


def test_window():
    h = RepoHistory.build([c(i, []) for i in range(5)])
    assert h.window() == Window(0, 4)
    assert h.window(since=T0 + DAY, until=T0 + 3 * DAY) == Window(1, 3)


def test_window_empty_cases():
    h = RepoHistory.build([c(i, []) for i in range(3)])
    w = h.window(since=T0 + 10 * DAY)
    assert w.empty and w.size == 0 and w.start == 3
    w = h.window(until=T0 - DAY)
    assert w.empty and w.end == -1
    e = RepoHistory.build([]).window()
    assert e == Window(0, -1) and e.empty and e.size == 0
    assert h.window().size == 3


def test_non_monotonic_ts_uses_running_max():
    h = RepoHistory.build([c(i, [], day=d) for i, d in enumerate([0, 5, 3, 6])])
    assert h.window(since=T0 + 4 * DAY).start == 1
    assert h.index_at_time(T0 + 4 * DAY) == 0
