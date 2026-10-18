import time

import numpy as np
import pytest
from constellation.model import RepoHistory
from tests.history_helpers import c, A, M, D, R
from tests.repo_builder import T0, DAY


def test_rename_keeps_identity_and_events():
    h = RepoHistory.build([c(0, [A("a.py", 3)]), c(1, [R("a.py", "b/a.py", 1, 0)])])
    assert len(h.files) == 1 and h.files[0].path == "b/a.py"
    assert h.files[0].event_commit.tolist() == [0, 1]
    assert h.files[0].paths == [(0, "a.py"), (1, "b/a.py")]
    assert h.file_id("b/a.py") == 0 and h.file_id("a.py") is None


def test_path_reuse_after_delete_is_new_file():
    h = RepoHistory.build([c(0, [A("x", 2)]), c(1, [D("x", 2)]), c(2, [A("x", 5)])])
    assert len(h.files) == 2
    assert (h.files[0].born, h.files[0].died) == (0, 1)
    assert h.files_alive_at(1).tolist() == [False, False]
    assert h.files_alive_at(2).tolist() == [False, True]


def test_loc_cumulative_and_head_exact():
    h = RepoHistory.build([c(0, [A("a", 10)]), c(1, [M("a", 2, 5)])], head_loc={"a": 99})
    assert h.loc_at(0).tolist() == [10]
    assert h.loc_at(1).tolist() == [99]


def test_loc_cumulative_without_head_clips_and_zeroes_dead():
    h = RepoHistory.build([c(0, [A("a", 3)]), c(1, [M("a", 0, 9)]), c(2, [A("b", 4)]),
                           c(3, [D("b", 4)])])
    assert h.loc_at(1).tolist() == [0, 0]
    assert h.loc_at(2).tolist() == [0, 4]
    assert h.loc_at(3).tolist() == [0, 0]


def test_dirs_and_authors():
    h = RepoHistory.build([c(0, [A("a/b/c.py", 1)], author="Ana"), c(1, [A("d.py", 1)], author="Bo")])
    assert h.dirs[0] == "" and set(h.dirs) == {"", "a", "a/b"}
    assert h.dirs[h.file_dir[0]] == "a/b" and h.dirs[h.file_dir[1]] == ""
    assert h.dir_parent[h.dirs.index("a/b")] == h.dirs.index("a")
    assert h.dir_parent[0] == -1
    assert h.authors == ["Ana", "Bo"] and h.commit_author.tolist() == [0, 1]


def test_index_at_time():
    h = RepoHistory.build([c(0, []), c(1, []), c(2, [])])
    assert h.index_at_time(T0 - 1) == -1
    assert h.index_at_time(T0 + DAY + 5) == 1


def test_modify_unknown_path_creates_file():   # shallow clone case
    h = RepoHistory.build([c(0, [M("old.py", 1, 0)], parents=("missing",))])
    assert h.files[0].born == 0


def test_binary_and_lineage_default():
    h = RepoHistory.build([c(0, [A("i.png", 0, binary=True)])])
    assert h.files[0].binary is True and h.lineage.tolist() == [0]


def test_rename_chain():
    h = RepoHistory.build([c(0, [A("a", 1)]), c(1, [R("a", "b", 0, 0)]), c(2, [R("b", "c", 0, 0)])])
    assert len(h.files) == 1
    assert h.files[0].paths == [(0, "a"), (1, "b"), (2, "c")]
    assert h.file_id("c") == 0 and h.file_id("a") is None and h.file_id("b") is None


def test_rename_onto_existing_path_kills_victim():
    h = RepoHistory.build([c(0, [A("a", 1), A("b", 2)]), c(1, [R("a", "b", 0, 0)])])
    assert len(h.files) == 2
    assert h.files[1].died == 1 and h.files[0].died == -1
    assert h.files_alive_at(1).tolist() == [True, False]
    assert h.file_id("b") == 0


def test_rename_from_unknown_old_path():
    h = RepoHistory.build([c(0, [R("ghost", "new", 1, 0)], parents=("missing",))])
    assert len(h.files) == 1 and h.files[0].born == 0 and h.file_id("new") == 0


def test_add_on_alive_path_merges():
    h = RepoHistory.build([c(0, [A("a", 1)]), c(1, [A("a", 2)])])
    assert len(h.files) == 1 and h.files[0].event_commit.tolist() == [0, 1]


def test_loc_zero_at_died_commit():
    h = RepoHistory.build([c(0, [A("a", 5)]), c(1, [D("a", 5)])])
    assert h.loc_at(0).tolist() == [5] and h.loc_at(1).tolist() == [0]


def test_head_loc_uses_latest_path_of_renamed_file():
    h = RepoHistory.build([c(0, [A("a", 5)]), c(1, [R("a", "b", 0, 0)])], head_loc={"b": 42})
    assert h.loc_at(1).tolist() == [42]


def test_build_empty():
    h = RepoHistory.build([])
    assert h.files == [] and h.dirs == [""] and h.index_at_time(5) == -1
    assert h.files_alive_at(-1).tolist() == [] and h.loc_at(-1).tolist() == []


@pytest.mark.slow
def test_perf_large_history():
    rng = np.random.default_rng(0)
    n_files, n_commits = 10_000, 50_000
    cms = []
    for i in range(n_commits):
        if i < n_files // 3 + 1:
            ch = [A(f"d{j % 50}/f{j}", 5) for j in range(i * 3, i * 3 + 3) if j < n_files]
        else:
            ch = [M(f"d{j % 50}/f{j}", 2, 1) for j in rng.integers(0, n_files, 3)]
        cms.append(c(i, ch))
    h = RepoHistory.build(cms)
    assert len(h.files) == n_files
    for fn in (h.files_alive_at, h.loc_at):
        fn(n_commits - 1)
        t = time.perf_counter()
        fn(n_commits - 1)
        assert time.perf_counter() - t < 0.02
