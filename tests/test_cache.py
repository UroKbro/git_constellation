import subprocess

import pytest

from constellation import cache
from constellation.cache import cache_dir, load_commits
from constellation.git_reader import iter_commits


def test_round_trip_equals_fresh_parse(repo):
    repo.commit("one", {"a b.py": "1\n", "bin": b"\x00"}); repo.mv("a b.py", "c.py"); repo.commit("two", day=1)
    first = load_commits(repo.path)
    assert first == list(iter_commits(repo.path))
    assert load_commits(repo.path) == first          # served from cache


def test_incremental_append(repo, monkeypatch):
    repo.commit("one", {"a": "1\n"}); load_commits(repo.path)
    repo.commit("two", {"a": "2\n"}, day=1)
    calls = []
    real = cache.iter_commits
    monkeypatch.setattr(cache, "iter_commits", lambda *a, **k: (calls.append(k), real(*a, **k))[1])
    assert [c.subject for c in load_commits(repo.path)] == ["one", "two"]
    assert calls[0].get("exclude") is not None


def test_rewritten_history_rebuilds(repo):
    repo.commit("one", {"a": "1\n"}); repo.commit("two", {"a": "2\n"}, day=1); load_commits(repo.path)
    subprocess.run(["git", "-C", repo.path, "reset", "-q", "--hard", "HEAD~1"], check=True)
    repo.commit("other", {"a": "3\n"}, day=2)
    assert [c.subject for c in load_commits(repo.path)] == ["one", "other"]


def test_corrupt_db_rebuilds(repo):
    repo.commit("one", {"a": "1\n"}); load_commits(repo.path)
    for f in cache_dir().glob("*.db"): f.write_bytes(b"garbage")
    assert len(load_commits(repo.path)) == 1


def test_empty_repo(repo):
    assert load_commits(repo.path) == []
    assert not list(cache_dir().glob("*.db"))


def test_progress_final_total(repo):
    repo.commit("one", {"a": "1\n"}); repo.commit("two", {"a": "2\n"}, day=1)
    seen = []
    load_commits(repo.path, progress=seen.append)
    assert seen[-1] == 2


def test_use_cache_false_writes_nothing(repo):
    repo.commit("one", {"a": "1\n"})
    assert len(load_commits(repo.path, use_cache=False)) == 1
    assert not list(cache_dir().glob("*.db"))


def test_incremental_append_matches_fresh_parse_with_merge(repo):
    repo.commit("m0", {"a": "0\n"}, day=0)
    repo.checkout("f", create=True); repo.commit("f1", {"f": "1\n"}, day=1)
    repo.checkout("main"); repo.commit("m1", {"a": "1\n"}, day=5)
    load_commits(repo.path)
    repo.checkout("f"); repo.commit("f2", {"f": "2\n"}, day=2)
    repo.checkout("main"); repo.merge("f", "merge f", day=6)
    assert load_commits(repo.path) == list(iter_commits(repo.path))


def test_schema_1_db_is_rebuilt(repo, monkeypatch):
    import sqlite3
    repo.commit("one", {"a": "1\n"}); load_commits(repo.path)
    for f in cache_dir().glob("*.db"):
        con = sqlite3.connect(f)
        with con:
            con.execute("UPDATE meta SET value='1' WHERE key='schema'")
        con.close()
    calls = []
    real = cache.iter_commits
    monkeypatch.setattr(cache, "iter_commits", lambda *a, **k: (calls.append(k), real(*a, **k))[1])
    assert load_commits(repo.path) == list(real(repo.path))
    assert calls == [{}]                        # full rebuild, not served or appended


def test_merge_changes_round_trip(repo):
    repo.commit("m0", {"a": "0\n"}, day=0)
    repo.checkout("f", create=True); repo.mv("a", "b"); repo.commit("f1", day=1)
    repo.checkout("main"); repo.commit("m1", {"c": "1\n"}, day=2)
    repo.merge("f", "merge f", day=3)
    fresh = list(iter_commits(repo.path))
    assert fresh[-1].merge_changes
    assert load_commits(repo.path) == fresh
    assert load_commits(repo.path) == fresh          # served from cache


@pytest.fixture
def readonly(tmp_path):
    made = []
    def make(p):
        p.chmod(0o500 if p.is_dir() else 0o400)
        made.append(p)
    yield make
    for p in made:
        p.chmod(0o700 if p.is_dir() else 0o600)


def test_unwritable_cache_dir_falls_back(repo, tmp_path, monkeypatch, readonly):
    repo.commit("one", {"a": "1\n"}); repo.commit("two", {"a": "2\n"}, day=1)
    ro = tmp_path / "ro"; ro.mkdir(); readonly(ro)
    monkeypatch.setenv("GIT_CONSTELLATION_CACHE", str(ro))
    assert load_commits(repo.path) == list(iter_commits(repo.path))
    assert not list(ro.glob("*.db"))


def test_unwritable_existing_db_falls_back(repo, monkeypatch, readonly):
    repo.commit("one", {"a": "1\n"}); load_commits(repo.path)
    repo.commit("two", {"a": "2\n"}, day=1)          # forces an incremental write
    for f in cache_dir().glob("*.db"): readonly(f)
    readonly(cache_dir())
    assert load_commits(repo.path) == list(iter_commits(repo.path))
