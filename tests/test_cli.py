from constellation.__main__ import main


def hot_repo(repo, name="hot.py"):
    for i in range(12):
        repo.commit(f"c{i}", {name: "x\n" * (i + 1), "calm.py": "y\n"}
                    if i == 0 else {name: "x\n" * (i + 1)}, day=i)
    return repo


def test_not_a_repo(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "is not inside a git repository" in capsys.readouterr().err


def test_unknown_branch(repo, capsys):
    repo.commit("a", {"a.py": "1\n"})
    assert main([str(repo.path), "--branch", "nope"]) == 1
    assert "error: unknown branch or ref 'nope'" in capsys.readouterr().err


def test_empty_repo(repo, capsys):
    assert main([str(repo.path)]) == 0
    assert "No commits yet." in capsys.readouterr().out


def test_plain_text_findings(repo, capsys):
    hot_repo(repo)
    assert main([str(repo.path)]) == 0
    cap = capsys.readouterr()
    assert "hot.py is a hotspot" in cap.out
    assert "Galaxy window arrives in Phase 2 — use --report for HTML." in cap.out
    assert "Reading history… " in cap.err


def test_report_written(repo, tmp_path, capsys):
    hot_repo(repo)
    out = tmp_path / "out.html"
    assert main([str(repo.path), "--report", str(out)]) == 0
    assert f"Report written to {out}" in capsys.readouterr().out
    html = out.read_text()
    assert html.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in html
    assert "hot.py is a hotspot" in html
    assert "Hotspots" in html
    assert "http://" not in html and "https://" not in html


def test_report_escapes_filenames(repo, tmp_path):
    hot_repo(repo, name="<script>x.py")
    out = tmp_path / "out.html"
    assert main([str(repo.path), "--report", str(out)]) == 0
    html = out.read_text()
    assert "<script>x.py" not in html
    assert "&lt;script&gt;x.py" in html


def test_since_after_all_commits(repo, capsys):
    hot_repo(repo)
    assert main([str(repo.path), "--since", "2999-01-01"]) == 0
    assert "0 commits" in capsys.readouterr().out


def test_until_before_all_commits(repo, capsys):
    hot_repo(repo)
    assert main([str(repo.path), "--until", "2000-01-01"]) == 0
    assert "0 commits" in capsys.readouterr().out


def test_bad_date_exits_2(repo, capsys):
    import pytest
    with pytest.raises(SystemExit) as e:
        main([str(repo.path), "--since", "yesterday"])
    assert e.value.code == 2


def test_report_in_missing_dir_is_an_error(repo, capsys):
    hot_repo(repo)
    assert main([str(repo.path), "--report", "/nonexistent-dir/x.html"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err


def test_git_failure_is_an_error(repo, capsys, monkeypatch):
    import subprocess
    import constellation.__main__ as cli
    hot_repo(repo)
    def boom(*a, **k):
        raise subprocess.CalledProcessError(128, ["git", "for-each-ref"])
    monkeypatch.setattr(cli, "read_tags", boom)
    assert main([str(repo.path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_since_until_same_day_selects_that_local_day(repo, capsys):
    from datetime import datetime
    from tests.repo_builder import DAY, T0
    def at(d, hour):  # RepoBuilder day offset for a local wall-clock time
        return (datetime(2024, 6, 10 + d, hour).timestamp() - T0) / DAY
    repo.commit("d1", {"a.py": "1\n"}, day=at(0, 12))
    repo.commit("d2", {"a.py": "2\n"}, day=at(1, 23))   # late: --until covers the whole day
    repo.commit("d3", {"a.py": "3\n"}, day=at(2, 12))
    assert main([str(repo.path), "--since", "2024-06-11", "--until", "2024-06-11"]) == 0
    out = capsys.readouterr().out
    assert "1 commit by 1 author, 1 file" in out
    assert "From 2024-06-11 to 2024-06-11." in out


def test_git_missing_from_path(repo, tmp_path, monkeypatch, capsys):
    repo.commit("a", {"a.py": "1\n"})
    empty = tmp_path / "empty-bin"; empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert main([str(repo.path)]) == 1
    err = capsys.readouterr().err
    assert "error: git not found on PATH" in err and "Traceback" not in err
