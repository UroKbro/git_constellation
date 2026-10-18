"""SQLite cache of parsed git history, with incremental append."""
import hashlib
import os
import sqlite3
import subprocess
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path

from constellation.git_reader import _git, head_sha, iter_commits
from constellation.types import Commit, FileChange

_SCHEMA = "1"
_PROGRESS_EVERY = 500

_DDL = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE commits(idx INTEGER PRIMARY KEY, sha TEXT, parents TEXT, author TEXT,
                     email TEXT, ts INTEGER, subject TEXT);
CREATE TABLE changes(commit_idx INTEGER, status TEXT, path TEXT, old_path TEXT,
                     added INTEGER, deleted INTEGER, binary INTEGER);
CREATE INDEX changes_commit ON changes(commit_idx);
"""


def cache_dir() -> Path:
    env = os.environ.get("GIT_CONSTELLATION_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "git_constellation"


def _db_path(root: Path, ref: str) -> Path:
    key = hashlib.sha1(f"{root.resolve()}\0{ref}".encode()).hexdigest()[:16]
    return cache_dir() / f"{key}.db"


def _is_ancestor(root: Path, old: str, new: str) -> bool:
    r = subprocess.run(_git(root, "merge-base", "--is-ancestor", old, new),
                       capture_output=True)
    return r.returncode == 0


def _read_all(con: sqlite3.Connection) -> list[Commit]:
    by_idx: dict[int, list[FileChange]] = {}
    for idx, status, path, old, added, deleted, binary in con.execute(
            "SELECT commit_idx, status, path, old_path, added, deleted, binary "
            "FROM changes ORDER BY rowid"):
        by_idx.setdefault(idx, []).append(
            FileChange(status, path, old, added, deleted, bool(binary)))
    return [Commit(sha, tuple(parents.split()), author, email, ts, subject,
                   tuple(by_idx.get(idx, ())))
            for idx, sha, parents, author, email, ts, subject in con.execute(
                "SELECT idx, sha, parents, author, email, ts, subject "
                "FROM commits ORDER BY idx")]


def _append(con: sqlite3.Connection, start: int, commits: Iterable[Commit]) -> int:
    idx = start
    for c in commits:
        con.execute("INSERT INTO commits VALUES (?,?,?,?,?,?,?)",
                    (idx, c.sha, " ".join(c.parents), c.author, c.email, c.ts, c.subject))
        con.executemany("INSERT INTO changes VALUES (?,?,?,?,?,?,?)",
                        [(idx, ch.status, ch.path, ch.old_path, ch.added, ch.deleted,
                          int(ch.binary)) for ch in c.changes])
        idx += 1
    return idx


def _counted(commits: Iterable[Commit], progress: Callable[[int], None] | None):
    n = 0
    for c in commits:
        n += 1
        if progress and n % _PROGRESS_EVERY == 0:
            progress(n)
        yield c


def _rebuild(path: Path, root: Path, ref: str, head: str,
             progress: Callable[[int], None] | None) -> list[Commit]:
    path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    commits = list(_counted(iter_commits(root, ref), progress))
    with closing(sqlite3.connect(path)) as con, con:
        con.executescript(_DDL)
        con.executemany("INSERT INTO meta VALUES (?,?)",
                        [("head", head), ("schema", _SCHEMA)])
        _append(con, 0, commits)
    if progress:
        progress(len(commits))
    return commits


def _try_cached(path: Path, root: Path, ref: str, head: str,
                progress: Callable[[int], None] | None) -> list[Commit] | None:
    if not path.exists():
        return None
    try:
        with closing(sqlite3.connect(path)) as con, con:
            meta = dict(con.execute("SELECT key, value FROM meta"))
            if meta.get("schema") != _SCHEMA or "head" not in meta:
                return None
            stored = meta["head"]
            if stored != head:
                if not _is_ancestor(root, stored, head):
                    return None
                n = con.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
                new = list(_counted(iter_commits(root, ref, exclude=stored), progress))
                _append(con, n, new)
                con.execute("UPDATE meta SET value=? WHERE key='head'", (head,))
            commits = _read_all(con)
    except sqlite3.DatabaseError:
        return None
    if progress:
        progress(len(commits))
    return commits


def load_commits(root: Path, ref: str = "HEAD",
                 progress: Callable[[int], None] | None = None,
                 use_cache: bool = True) -> list[Commit]:
    head = head_sha(root, ref)
    if head is None:
        return []
    if not use_cache:
        commits = list(_counted(iter_commits(root, ref), progress))
        if progress:
            progress(len(commits))
        return commits
    path = _db_path(root, ref)
    cached = _try_cached(path, root, ref, head, progress)
    if cached is not None:
        return cached
    return _rebuild(path, root, ref, head, progress)
