"""Build deterministic throwaway git repos for tests."""
import os
import subprocess
from pathlib import Path

T0 = 1_700_000_000
DAY = 86_400

_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}


class RepoBuilder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.name", "Test")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "commit.gpgsign", "false")
        self._git("config", "tag.gpgsign", "false")

    def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        full_env = {**os.environ, **_ENV, **(env or {})}
        r = subprocess.run(["git", "-C", str(self.path), *args], env=full_env,
                           capture_output=True, text=True, check=True)
        return r.stdout

    @staticmethod
    def _date_env(author: str, day: float) -> dict[str, str]:
        date = f"{T0 + int(day * DAY)} +0000"
        email = f"{author.lower()}@example.com"
        return {"GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": email,
                "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_NAME": author,
                "GIT_COMMITTER_EMAIL": email, "GIT_COMMITTER_DATE": date}

    def commit(self, msg: str, files: dict[str, str | bytes | None] | None = None,
               author: str = "Ana", day: float = 0) -> str:
        for name, content in (files or {}).items():
            p = self.path / name
            if content is None:
                p.unlink()
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                p.write_bytes(content)
            else:
                p.write_text(content)
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", msg, env=self._date_env(author, day))
        return self._git("rev-parse", "HEAD").strip()

    def mv(self, old: str, new: str) -> None:
        (self.path / new).parent.mkdir(parents=True, exist_ok=True)
        self._git("mv", old, new)

    def checkout(self, branch: str, create: bool = False) -> None:
        self._git("checkout", "-q", *(["-b"] if create else []), branch)

    def orphan(self, branch: str) -> None:
        self._git("checkout", "-q", "--orphan", branch)
        self._git("rm", "-rf", "-q", "--ignore-unmatch", ".")

    def merge(self, branch: str, msg: str, day: float, author: str = "Ana",
              unrelated: bool = False) -> str:
        args = ["merge", "-q", "--no-ff", "-m", msg]
        if unrelated:
            args.append("--allow-unrelated-histories")
        self._git(*args, branch, env=self._date_env(author, day))
        return self._git("rev-parse", "HEAD").strip()

    def tag(self, name: str, annotated: bool = False) -> None:
        if annotated:
            self._git("tag", "-a", name, "-m", name, env=self._date_env("Ana", 0))
        else:
            self._git("tag", name)
