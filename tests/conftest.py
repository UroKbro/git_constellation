import pytest

from tests.repo_builder import RepoBuilder


@pytest.fixture
def repo(tmp_path):
    return RepoBuilder(tmp_path / "repo")


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONSTELLATION_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def gl_ctx():
    from constellation.render.gl import standalone_context

    ctx = standalone_context()
    if ctx is None:
        pytest.skip("no OpenGL context")
    yield ctx
    ctx.release()
