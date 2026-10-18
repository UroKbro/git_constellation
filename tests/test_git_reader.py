import pytest

from constellation.git_reader import (NotARepoError, UnknownRefError, head_loc,
                                      head_sha, iter_commits, read_tags,
                                      repo_root)
from constellation.types import FileChange, Tag
from tests.repo_builder import DAY, T0


def test_parses_add_modify_rename_delete_binary(repo):
    repo.commit("one", {"sp ace.py": "a\nb\n", "bin.dat": b"\x00\x01"}, day=0)
    repo.mv("sp ace.py", "renamed ü.py")
    repo.commit("two", {"renamed ü.py": "a\nb\nc\n"}, author="Bo", day=1)
    repo.commit("three", {"bin.dat": None}, day=2)
    commits = list(iter_commits(repo.path))
    assert [c.subject for c in commits] == ["one", "two", "three"]
    assert commits[0].parents == ()
    assert commits[1].author == "Bo" and commits[1].ts == T0 + DAY
    assert set(commits[0].changes) == {
        FileChange("A", "bin.dat", None, 0, 0, True),
        FileChange("A", "sp ace.py", None, 2, 0, False)}
    assert commits[1].changes == (FileChange("R", "renamed ü.py", "sp ace.py", 1, 0, False),)
    assert commits[2].changes == (FileChange("D", "bin.dat", None, 0, 0, True),)


def test_merge_record_folds_with_merge_changes_from_both_parents(repo):
    repo.commit("a", {"a.txt": "1\n"}, day=0)
    repo.checkout("side", create=True)
    repo.commit("b", {"b.txt": "1\n"}, day=1)
    repo.checkout("main")
    repo.commit("c", {"c.txt": "1\n"}, day=2)
    repo.merge("side", "merge", day=3)
    commits = list(iter_commits(repo.path))
    assert [c.subject for c in commits] == ["a", "b", "c", "merge"]   # one Commit per merge
    merge = commits[-1]
    assert len(merge.parents) == 2 and all(len(p) == 40 for p in merge.parents)
    assert merge.changes == ()
    assert merge.merge_changes == (FileChange("A", "b.txt", None, 1, 0, False),   # vs main
                                   FileChange("A", "c.txt", None, 1, 0, False))   # vs side
    assert all(c.merge_changes == () for c in commits[:-1])


def test_exclude_returns_only_new_commits(repo):
    first = repo.commit("one", {"a": "1\n"}, day=0)
    repo.commit("two", {"a": "2\n"}, day=1)
    repo.commit("three", {"a": "3\n"}, day=2)
    commits = list(iter_commits(repo.path, exclude=first))
    assert [c.subject for c in commits] == ["two", "three"]


def test_repo_root_from_subdirectory(repo):
    repo.commit("one", {"pkg/sub/x.py": "1\n"})
    assert repo_root(repo.path / "pkg" / "sub") == repo.path.resolve()


def test_not_a_repo(tmp_path):
    with pytest.raises(NotARepoError):
        repo_root(tmp_path)
    with pytest.raises(NotARepoError):
        repo_root(tmp_path / "missing")


def test_empty_repo_and_unknown_ref(repo):
    assert head_sha(repo.path) is None
    repo.commit("one", {"a": "1\n"})
    with pytest.raises(UnknownRefError):
        head_sha(repo.path, "nope")


def test_tags_are_peeled(repo):
    s = repo.commit("one", {"a": "1\n"})
    repo.tag("v1")
    repo.tag("v2", annotated=True)
    assert sorted(read_tags(repo.path), key=lambda t: t.name) == [Tag("v1", s), Tag("v2", s)]


def test_head_loc(repo):
    repo.commit("one", {"a.py": "1\n2\n", "b.py": "1\n2", "e.py": "", "bin": b"\x00z"})
    assert head_loc(repo.path, "HEAD") == {"a.py": 2, "b.py": 2, "e.py": 0, "bin": 0}
