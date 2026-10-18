import queue
import sys
import types

import numpy as np
import pytest

from constellation.loader import Loaded, start_loading


@pytest.fixture(autouse=True)
def _collapse_stub(monkeypatch):
    """Stand in for constellation.collapse until it exists."""
    try:
        from constellation import collapse  # noqa: F401
    except ImportError:
        mod = types.ModuleType("constellation.collapse")
        mod.plan = lambda h, w: ("plan", w)
        monkeypatch.setitem(sys.modules, "constellation.collapse", mod)
        import constellation
        monkeypatch.setattr(constellation, "collapse", mod, raising=False)


def drain(q: queue.Queue, timeout: float = 10.0) -> list[tuple]:
    out = []
    while True:
        msg = q.get(timeout=timeout)
        out.append(msg)
        if msg[0] in ("done", "error"):
            return out


def test_loads_three_commits(repo):
    for i in range(3):
        repo.commit(f"c{i}", {"a.py": "x\n" * (i + 1), "b.py": "y\n"}, day=i)
    msgs = drain(start_loading(repo.path, "HEAD", None, None, False))
    assert any(m[0] == "progress" for m in msgs)
    assert msgs[-1][0] == "done"
    loaded = msgs[-1][1]
    assert isinstance(loaded, Loaded)
    assert len(loaded.h.commits) == 3
    assert np.isfinite(loaded.layout.place_pos).all()
    assert any(m[0] == "progress" and m[1] == "Laying out 2 files…" for m in msgs)


def test_not_a_repo(tmp_path):
    msgs = drain(start_loading(tmp_path, "HEAD", None, None, False))
    assert msgs[-1][0] == "error"
    assert "is not inside a git repository" in msgs[-1][1]


def test_unknown_ref(repo):
    repo.commit("a", {"a.py": "1\n"})
    msgs = drain(start_loading(repo.path, "nope", None, None, False))
    assert msgs[-1] == ("error", "error: unknown branch or ref 'nope'")


def test_empty_repo(repo):
    msgs = drain(start_loading(repo.path, "HEAD", None, None, False))
    assert msgs[-1] == ("done", None)


def test_empty_window(repo):
    repo.commit("a", {"a.py": "1\n"}, day=5)
    msgs = drain(start_loading(repo.path, "HEAD", 1, 2, False))
    assert msgs[-1][0] == "done"
    assert msgs[-1][1].window.empty
