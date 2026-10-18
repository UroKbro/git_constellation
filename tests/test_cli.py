import subprocess
import sys
from pathlib import Path

import pytest

from constellation.__main__ import main

ROOT = Path(__file__).resolve().parent.parent


def hot_repo(repo, name="hot.py"):
    for i in range(12):
        repo.commit(f"c{i}", {name: "x\n" * (i + 1), "calm.py": "y\n"}
                    if i == 0 else {name: "x\n" * (i + 1)}, day=i)
    return repo


def test_not_a_repo(tmp_path, capsys):
    assert main([str(tmp_path), "--summary"]) == 1
    assert "is not inside a git repository" in capsys.readouterr().err


def test_unknown_branch(repo, capsys):
    repo.commit("a", {"a.py": "1\n"})
    assert main([str(repo.path), "--branch", "nope", "--summary"]) == 1
    assert "error: unknown branch or ref 'nope'" in capsys.readouterr().err


def test_empty_repo(repo, capsys):
    assert main([str(repo.path), "--summary"]) == 0
    assert "No commits yet." in capsys.readouterr().out


def test_plain_text_findings(repo, capsys):
    hot_repo(repo)
    assert main([str(repo.path), "--summary"]) == 0
    cap = capsys.readouterr()
    assert "hot.py is a hotspot" in cap.out
    assert "Galaxy window" not in cap.out
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
    assert main([str(repo.path), "--since", "2999-01-01", "--summary"]) == 0
    assert "0 commits" in capsys.readouterr().out


def test_until_before_all_commits(repo, capsys):
    hot_repo(repo)
    assert main([str(repo.path), "--until", "2000-01-01", "--summary"]) == 0
    assert "0 commits" in capsys.readouterr().out


def test_bad_date_exits_2(repo, capsys):
    import pytest
    with pytest.raises(SystemExit) as e:
        main([str(repo.path), "--since", "yesterday", "--summary"])
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
    assert main([str(repo.path), "--summary"]) == 1
    assert "error:" in capsys.readouterr().err


def test_since_until_same_day_selects_that_local_day(repo, capsys):
    from datetime import datetime
    from tests.repo_builder import DAY, T0
    def at(d, hour):  # RepoBuilder day offset for a local wall-clock time
        return (datetime(2024, 6, 10 + d, hour).timestamp() - T0) / DAY
    repo.commit("d1", {"a.py": "1\n"}, day=at(0, 12))
    repo.commit("d2", {"a.py": "2\n"}, day=at(1, 23))   # late: --until covers the whole day
    repo.commit("d3", {"a.py": "3\n"}, day=at(2, 12))
    assert main([str(repo.path), "--since", "2024-06-11", "--until", "2024-06-11",
                 "--summary"]) == 0
    out = capsys.readouterr().out
    assert "1 commit by 1 author, 1 file" in out
    assert "From 2024-06-11 to 2024-06-11." in out


def test_git_missing_from_path(repo, tmp_path, monkeypatch, capsys):
    repo.commit("a", {"a.py": "1\n"})
    empty = tmp_path / "empty-bin"; empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert main([str(repo.path), "--summary"]) == 1
    err = capsys.readouterr().err
    assert "error: git not found on PATH" in err and "Traceback" not in err


def test_summary_never_imports_pygame(repo):
    hot_repo(repo)
    code = ("import sys; from constellation.__main__ import main; "
            f"rc = main([{str(repo.path)!r}, '--summary', '--no-cache']); "
            "assert 'pygame' not in sys.modules and 'moderngl' not in sys.modules; "
            "raise SystemExit(rc)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_cli_window_mode_calls_app(repo, monkeypatch, capsys):
    import constellation.app as app
    hot_repo(repo)
    seen = {}

    def fake_run(root, ref, since, until, use_cache, title, **kw):
        seen.update(root=root, ref=ref, since=since, until=until, use_cache=use_cache,
                    title=title, kw=kw)
        return 0
    monkeypatch.setattr(app, "run", fake_run)
    assert main([str(repo.path), "--since", "2024-06-11", "--no-cache"]) == 0
    assert seen["root"] == repo.path.resolve()
    assert seen["ref"] == "HEAD" and seen["until"] is None and seen["use_cache"] is False
    assert isinstance(seen["since"], int)
    assert seen["title"] == f"git-constellation — {repo.path.name}"
    assert seen["kw"] == {"bench": None, "screenshot": None, "at": 1.0, "display": None,
                          "theme": None, "export": None}
    assert "Galaxy window" not in capsys.readouterr().out


def test_cli_hidden_flags_reach_app(repo, monkeypatch):
    import constellation.app as app
    hot_repo(repo)
    seen = {}
    monkeypatch.setattr(app, "run", lambda *a, **kw: seen.update(kw) or 0)
    assert main([str(repo.path), "--bench", "2.5", "--screenshot", "x.png", "--at", "0.6",
                 "--display", "1"]) == 0
    assert seen == {"bench": 2.5, "screenshot": "x.png", "at": 0.6, "display": 1,
                    "theme": None, "export": None}


def test_bench_flag_is_hidden(capsys):
    import pytest
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "--bench" not in out and "--screenshot" not in out and "--summary" in out


def test_cli_window_mode_empty_repo(repo, monkeypatch, capsys):
    import constellation.app as app
    titles = []
    monkeypatch.setattr(app, "run", lambda root, ref, s, u, c, title, **kw:
                        titles.append(title) or 0)
    assert main([str(repo.path)]) == 0
    assert "No commits yet." in capsys.readouterr().out
    assert len(titles) == 1


def test_cli_window_mode_errors_before_opening(tmp_path, monkeypatch, capsys):
    import constellation.app as app
    monkeypatch.setattr(app, "run", lambda *a, **kw: pytest.fail("window opened"))
    assert main([str(tmp_path)]) == 1
    assert "is not inside a git repository" in capsys.readouterr().err


def test_initial_mask_is_alive_at_window_end(repo):
    from constellation.app import initial_mask
    from constellation.layout import compute_layout
    from constellation.model import RepoHistory
    from constellation.git_reader import iter_commits
    repo.commit("a", {"a.py": "1\n", "b.py": "1\n"}, day=0)
    repo.commit("b", {"b.py": None}, day=1)
    h = RepoHistory.build(list(iter_commits(repo.path)))
    lay = compute_layout(h)
    mask = initial_mask(h, lay, h.window())
    drawn = {h.files[f].path for f in lay.place_file[mask]}
    assert drawn == {"a.py"}


def _display_env() -> dict:
    """The parent env without the dummy video driver tests/test_ui.py sets in-process."""
    import os
    env = dict(os.environ)
    if env.get("SDL_VIDEODRIVER") == "dummy":
        del env["SDL_VIDEODRIVER"]
    return env


def _screenshot(repo_path, out, *extra):
    r = subprocess.run([sys.executable, "-m", "constellation", str(repo_path),
                        "--screenshot", str(out), *extra], cwd=ROOT, capture_output=True,
                       text=True, env=_display_env())
    if r.returncode == 3:
        pytest.skip(f"no display: {r.stderr.strip()}")
    assert r.returncode == 0, r.stderr
    return r


@pytest.mark.slow
@pytest.mark.parametrize("at", ["1.0", "0.5"])
def test_screenshot_flag(repo, tmp_path, at):
    pygame = pytest.importorskip("pygame")
    import numpy as np
    for i in range(3):
        repo.commit(f"c{i}", {f"src/f{i}.py": "x\n" * 40, f"docs/d{i}.md": "y\n"}, day=i)
    out = tmp_path / "out.png"
    r = _screenshot(repo.path, out, "--at", at)
    img = pygame.image.load(str(out))
    w, h = img.get_size()
    reported = r.stdout.strip().rsplit("(", 1)[1].rstrip(")")
    assert reported == f"{w}x{h}"
    assert w > 0 and h > 0 and w * 800 == h * 1280  # drawable keeps the window aspect
    px = np.frombuffer(pygame.image.tobytes(img, "RGB"), np.uint8)
    assert px.max() > 100


@pytest.mark.slow
def test_screenshot_empty_repo(repo, tmp_path):
    pygame = pytest.importorskip("pygame")
    import numpy as np
    out = tmp_path / "empty.png"
    _screenshot(repo.path, out)
    img = pygame.image.load(str(out))
    px = np.frombuffer(pygame.image.tobytes(img, "RGB"), np.uint8).reshape(
        img.get_height(), img.get_width(), 3)
    h, w = px.shape[:2]
    assert px[h // 2 - h // 10:h // 2 + h // 10, w // 3:2 * w // 3].max() > 150  # message
