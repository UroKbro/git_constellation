import subprocess

from tests.repo_builder import DAY, T0


def test_builder_makes_commits_with_fixed_dates(repo):
    sha = repo.commit("first", {"a.py": "x\n"}, author="Bo", day=2)
    out = subprocess.run(["git", "-C", repo.path, "log", "-1", "--format=%H %an %ae %at"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == [sha, "Bo", "bo@example.com", str(T0 + 2 * DAY)]


def test_builder_delete_and_merge(repo):
    repo.commit("a", {"a.py": "1\n"})
    repo.checkout("side", create=True)
    repo.commit("b", {"b.py": "2\n"}, day=1)
    repo.checkout("main")
    m = repo.merge("side", "merge", day=2)
    repo.commit("rm", {"a.py": None}, day=3)
    assert not (repo.path / "a.py").exists()
    parents = subprocess.run(["git", "-C", repo.path, "log", "-1", "--format=%P", m],
                             capture_output=True, text=True, check=True).stdout.split()
    assert len(parents) == 2
