"""Merge commits reconcile the file set toward the merge tree (no metric events)."""
import numpy as np
import pytest

from constellation.git_reader import head_loc, iter_commits
from constellation.metrics import change_counts
from constellation.model import RepoHistory


def _build(repo) -> RepoHistory:
    return RepoHistory.build(list(iter_commits(repo.path)))


def _alive(h: RepoHistory) -> set[str]:
    return {h.files[i].path for i in np.flatnonzero(h.files_alive_at(len(h.commits) - 1))}


def _merge_indexes(h: RepoHistory) -> list[int]:
    return [i for i, cm in enumerate(h.commits) if len(cm.parents) > 1]


def _no_merge_events(h: RepoHistory) -> None:
    merges = set(_merge_indexes(h))
    assert merges
    for f in h.files:
        assert not merges & set(f.event_commit.tolist())


def zombie_after_dir_rename(repo):
    """base has old/x; main `git mv old new` on day 2; side edits old/x on day 3;
    merge on day 4. Date order replays the side edit after the rename, so the
    plain replay resurrects old/x. The merge's diff vs parent 2 (R old/x -> new/x)
    must kill it."""
    repo.commit("base", {"old/x": "1\n2\n3\n4\n5\n"}, day=0)
    repo.checkout("side", create=True)
    repo.commit("edit", {"old/x": "1\n2\n3\n4\n5\n6\n"}, day=3)
    repo.checkout("main")
    repo.mv("old", "new")
    repo.commit("mv", day=2)
    repo.merge("side", "merge", day=4)


def evil_merge_delete_add(repo):
    """The merge itself (a conflict-style resolution) deletes keep/gone.py and
    adds born.py; no side commit does either."""
    repo.commit("base", {"keep/gone.py": "g\n", "a.py": "a\n"}, day=0)
    repo.checkout("side", create=True)
    repo.commit("s", {"s.py": "s\n"}, day=1)
    repo.checkout("main")
    repo.commit("m", {"a.py": "a\nb\n"}, day=2)
    repo.merge("side", "merge", day=3, resolve={"keep/gone.py": None, "born.py": "b\n"})


def modify_delete_kept(repo):
    """Side edits y.py (day 1), main deletes it (day 2); the merge resolves the
    modify/delete conflict by keeping side's version, so y.py is re-added by the
    merge relative to the first parent."""
    repo.commit("base", {"y.py": "1\n", "x.py": "x\n"}, day=0)
    repo.checkout("side", create=True)
    repo.commit("edit", {"y.py": "1\n2\n"}, day=1)
    repo.checkout("main")
    repo.commit("del", {"y.py": None}, day=2)
    repo.merge("side", "merge", day=3, resolve={"y.py": "1\n2\n"})


def side_rename_main_edit(repo):
    """Side renames x -> y (day 1); main edits x later (day 2), which the replay
    turns into a zombie x; the merge's diff vs parent 1 (R x -> y) kills it."""
    repo.commit("base", {"x": "1\n2\n3\n4\n5\n"}, day=0)
    repo.checkout("side", create=True)
    repo.mv("x", "y")
    repo.commit("mv", day=1)
    repo.checkout("main")
    repo.commit("edit", {"x": "1\n2\n3\n4\n5\n6\n"}, day=2)
    repo.merge("side", "merge", day=3)


def plain_merges(repo):
    repo.commit("base", {"a/x.py": "1\n", "a/y.py": "1\n"}, day=0)
    repo.checkout("f", create=True)
    repo.commit("f1", {"f.py": "1\n", "a/y.py": None}, day=1)
    repo.checkout("main")
    repo.commit("m1", {"a/x.py": "2\n"}, day=2)
    repo.merge("f", "merge f", day=3)
    repo.checkout("g", create=True)
    repo.commit("g1", {"g.py": "1\n"}, day=4)
    repo.checkout("main")
    repo.commit("m2", {"f.py": None}, day=5)
    repo.merge("g", "merge g", day=6)


SCENARIOS = [zombie_after_dir_rename, evil_merge_delete_add, modify_delete_kept,
             side_rename_main_edit, plain_merges]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda f: f.__name__)
def test_alive_paths_match_head_tree(repo, scenario):
    scenario(repo)
    h = _build(repo)
    assert _alive(h) == set(head_loc(repo.path, "HEAD"))
    _no_merge_events(h)


def test_dir_rename_zombie_killed_and_identity_kept(repo):
    zombie_after_dir_rename(repo)
    h = _build(repo)
    assert _alive(h) == {"new/x"}
    fid = h.file_id("new/x")
    assert h.files[fid].born == 0 and h.files[fid].paths[0] == (0, "old/x")
    assert not any(f.path == "old/x" for f in h.files)   # phantom folded away


def test_merge_resolved_delete_and_add(repo):
    evil_merge_delete_add(repo)
    h = _build(repo)
    merge = _merge_indexes(h)[0]
    gone = next(f for f in h.files if f.path == "keep/gone.py")
    assert gone.died == merge
    born = h.files[h.file_id("born.py")]
    assert born.born == merge and born.event_commit.tolist() == []
    counts = change_counts(h, 0, len(h.commits) - 1)
    assert counts[h.file_id("born.py")] == 0
    assert counts[h.file_id("a.py")] == 2       # base + m; the merge adds nothing


def test_phantom_after_rename_folds_into_target(repo):
    zombie_after_dir_rename(repo)
    h = _build(repo)
    assert [f.path for f in h.files] == ["new/x"]
    edit = next(i for i, cm in enumerate(h.commits) if cm.subject == "edit")
    assert edit in h.files[0].event_commit.tolist()
    assert list(h.files[0].event_commit) == sorted(h.files[0].event_commit)
    for i in range(len(h.commits)):
        assert h.files_alive_at(i).sum() == 1   # never a ghost old/x


def test_phantom_after_delete_folds_into_dead_original(repo):
    repo.commit("base", {"g.py": "1\n", "k.py": "k\n"}, day=0)
    repo.checkout("side", create=True)
    repo.commit("edit", {"g.py": "1\n2\n"}, day=3)
    repo.checkout("main")
    repo.commit("del", {"g.py": None}, day=2)
    repo.merge("side", "merge", day=4, resolve={"g.py": None})
    h = _build(repo)
    assert sorted(f.path for f in h.files) == ["g.py", "k.py"]
    g = h.files[[f.path for f in h.files].index("g.py")]
    edit = next(i for i, cm in enumerate(h.commits) if cm.subject == "edit")
    assert edit in g.event_commit.tolist()
    assert _alive(h) == {"k.py"}


def test_surviving_phantom_kept(repo):
    modify_delete_kept(repo)          # existing scenario: merge keeps y.py
    assert "y.py" in _alive(_build(repo))
