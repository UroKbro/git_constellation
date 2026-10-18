from datetime import datetime
from types import SimpleNamespace

import numpy as np

from constellation.details import (aggregate_tooltip, commit_details,
                                   file_details, tooltip)
from constellation.git_reader import iter_commits
from constellation.metrics import CouplingIndex
from constellation.model import RepoHistory
from tests.repo_builder import DAY, T0


def _day(ts):
    return f"{datetime.fromtimestamp(ts):%Y-%m-%d}"


def _repo(repo):
    repo.commit("init", {"a.py": "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n", "b.py": "x\n"}, author="Ana", day=0)
    repo.commit("two", {"a.py": "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n", "b.py": "x\ny\n"},
                author="Bo", day=1)
    repo.commit("three", {"a.py": "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n"}, author="Ana", day=2)
    repo.commit("solo", {"solo/c.py": "z\n"}, author="Cy", day=3)
    return RepoHistory.build(list(iter_commits(repo.path)))


def test_file_details_exact(repo):
    h = _repo(repo)
    w = h.window()
    fid = h.file_id("a.py")
    lines = file_details(h, w, 3, fid, CouplingIndex(h))
    assert lines[0] == "a.py"
    assert lines[1] == "13 lines · 3 changes"
    assert lines[2] == "Authors: Ana 85%, Bo 15%"
    assert lines[3] == f"First touched {_day(T0)} · last touched {_day(T0 + 2 * DAY)}"
    assert lines[4].startswith("Hotspot ") and len(lines[4].split()[1].split(".")[1]) == 2
    assert lines[5:] == ["Coupled with:", "  b.py (100%)"]


def test_authors_match_ownership(repo):
    from constellation.metrics import ownership
    h = _repo(repo)
    fid = h.file_id("a.py")
    top = ownership(h, 0, 3).top3[fid]
    want = "Authors: " + ", ".join(f"{h.authors[a]} {round(s * 100)}%" for a, s in top)
    assert file_details(h, h.window(), 3, fid, None)[2] == want


def test_coupled_none_and_singular(repo):
    h = _repo(repo)
    lines = file_details(h, h.window(), 3, h.file_id("solo/c.py"), CouplingIndex(h))
    assert lines[1] == "1 lines · 1 change"
    assert lines[-1] == "Coupled with: none"
    assert lines[3] == f"First touched {_day(T0 + 3 * DAY)} · last touched {_day(T0 + 3 * DAY)}"
    assert file_details(h, h.window(), 3, h.file_id("solo/c.py"), None)[-1] == "Coupled with: none"


def test_commit_details(repo):
    h = _repo(repo)
    lines = commit_details(h, 1)
    sha = h.commits[1].sha
    assert lines[0] == f"{sha[:10]} · Bo · {datetime.fromtimestamp(T0 + DAY):%Y-%m-%d %H:%M}"
    assert lines[1] == "two"
    assert lines[2] == "2 files · +3 −0"
    assert commit_details(h, 3)[2] == "1 file · +1 −0"


def test_merge_line(repo):
    repo.commit("base", {"m.txt": "a\n"}, day=0)
    repo.checkout("side", create=True)
    repo.commit("side", {"s.txt": "s\n"}, day=1)
    repo.checkout("main")
    repo.commit("main", {"t.txt": "t\n"}, day=2)
    repo.merge("side", "merge it", day=3)
    h = RepoHistory.build(list(iter_commits(repo.path)))
    assert commit_details(h, len(h.commits) - 1)[2] == "merge of 2 parents"


def test_tooltips(repo):
    h = _repo(repo)
    assert tooltip(h, 3, h.file_id("a.py")) == "a.py — 13 lines"
    d = h.dirs.index("solo")
    plan = SimpleNamespace(member_to_group=np.array([-1, -1, 0], dtype=np.int32), group_dir=[d])
    assert aggregate_tooltip(h, plan, 0, 3) == "solo/ — 1 file (double-click to expand)"
    assert aggregate_tooltip(h, plan, 0, 2) == "solo/ — 0 files (double-click to expand)"
