"""SQLite cache of parsed git history, with incremental append."""
import hashlib
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path

from constellation.git_reader import _git, head_sha, iter_commits
from constellation.types import Commit, FileChange

_SCHEMA = "2"
_PROGRESS_EVERY = 500

_DDL = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE commits(idx INTEGER PRIMARY KEY, sha TEXT, parents TEXT, author TEXT,
                     email TEXT, ts INTEGER, subject TEXT);
CREATE TABLE changes(commit_idx INTEGER, status TEXT, path TEXT, old_path TEXT,
                     added INTEGER, deleted INTEGER, binary INTEGER,
                     is_merge INTEGER);
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


def _rev_list(root: Path, ref: str) -> list[str]:
    r = subprocess.run(_git(root, "rev-list", "--reverse", "--date-order", ref),
                       capture_output=True, check=True)
    return r.stdout.decode().split()


def _read_all(con: sqlite3.Connection) -> list[Commit]:
    by_idx: dict[tuple[int, int], list[FileChange]] = {}
    for idx, status, path, old, added, deleted, binary, is_merge in con.execute(
            "SELECT commit_idx, status, path, old_path, added, deleted, binary, "
            "is_merge FROM changes ORDER BY rowid"):
        by_idx.setdefault((idx, is_merge), []).append(
            FileChange(status, path, old, added, deleted, bool(binary)))
    return [Commit(sha, tuple(parents.split()), author, email, ts, subject,
                   tuple(by_idx.get((idx, 0), ())), tuple(by_idx.get((idx, 1), ())))
            for idx, sha, parents, author, email, ts, subject in con.execute(
                "SELECT idx, sha, parents, author, email, ts, subject "
                "FROM commits ORDER BY idx")]


def _append(con: sqlite3.Connection, start: int, commits: Iterable[Commit]) -> int:
    idx = start
    for c in commits:
        con.execute("INSERT INTO commits VALUES (?,?,?,?,?,?,?)",
                    (idx, c.sha, " ".join(c.parents), c.author, c.email, c.ts, c.subject))
        con.executemany("INSERT INTO changes VALUES (?,?,?,?,?,?,?,?)",
                        [(idx, ch.status, ch.path, ch.old_path, ch.added, ch.deleted,
                          int(ch.binary), is_merge)
                         for is_merge, chs in ((0, c.changes), (1, c.merge_changes))
                         for ch in chs])
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
                new = list(_counted(iter_commits(root, ref, exclude=stored), progress))
                combined = _read_all(con) + new
                order = _rev_list(root, ref)
                by_sha = {c.sha: c for c in combined}
                if len(by_sha) != len(combined) or set(order) != set(by_sha):
                    return None
                commits = [by_sha[sha] for sha in order]
                con.execute("DELETE FROM commits")
                con.execute("DELETE FROM changes")
                _append(con, 0, commits)
                con.execute("UPDATE meta SET value=? WHERE key='head'", (head,))
            else:
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
    if use_cache:
        path = _db_path(root, ref)
        try:
            cached = _try_cached(path, root, ref, head, progress)
            if cached is not None:
                return cached
            return _rebuild(path, root, ref, head, progress)
        except (OSError, sqlite3.Error) as e:  # read-only, disk full…: never break the run
            print(f"warning: cache unavailable ({e}); reading history without cache",
                  file=sys.stderr)
    commits = list(_counted(iter_commits(root, ref), progress))
    if progress:
        progress(len(commits))
    return commits
