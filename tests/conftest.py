import pytest

from tests.repo_builder import RepoBuilder


@pytest.fixture
def repo(tmp_path):
    return RepoBuilder(tmp_path / "repo")


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONSTELLATION_CACHE", str(tmp_path / "cache"))
