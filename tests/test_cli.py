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
