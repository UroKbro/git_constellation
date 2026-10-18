import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_wheel_installs_and_runs(tmp_path, repo):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy(ROOT / name, src / name)
    shutil.copytree(ROOT / "constellation", src / "constellation",
                    ignore=shutil.ignore_patterns("__pycache__"))
    dist = tmp_path / "dist"
    subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w",
                    str(dist), str(src)], check=True, capture_output=True)
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    wheel = next(dist.glob("*.whl"))
    subprocess.run([venv / "bin" / "python", "-m", "pip", "install", "-q", str(wheel)],
                   check=True, capture_output=True)
    repo.commit("one", {"a.py": "1\n"})
    r = subprocess.run([venv / "bin" / "git-constellation", str(repo.path), "--no-cache",
                        "--summary"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "commits by" in r.stdout or "commit by" in r.stdout
    code = ("from constellation.render.renderer import load_shader; "
            "print(len(load_shader('planet.vert')) > 0, len(load_shader('planet.frag')) > 0)")
    r = subprocess.run([venv / "bin" / "python", "-c", code], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True True"
