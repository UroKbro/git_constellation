"""RepoHistory: the core data structure consumed by metrics and the renderer."""
from dataclasses import dataclass

import numpy as np

from constellation.types import Commit, Tag


def _dir_of(path: str) -> str:
    return path.rpartition("/")[0]


@dataclass
class FileTrack:
    path: str
    paths: list[tuple[int, str]]
    event_commit: np.ndarray
    event_added: np.ndarray
    event_deleted: np.ndarray
    event_author: np.ndarray
    born: int
    died: int
    binary: bool


class _Builder:
    """Mutable per-file state while replaying commits."""
    __slots__ = ("paths", "ec", "ea", "ed", "eu", "born", "died", "binary")

    def __init__(self, path: str, born: int):
        self.paths = [(born, path)]
        self.ec: list[int] = []
        self.ea: list[int] = []
        self.ed: list[int] = []
        self.eu: list[int] = []
        self.born = born
        self.died = -1
        self.binary = False


class RepoHistory:
    def __init__(self):
        self.commits: list[Commit] = []
        self.commit_ts = np.zeros(0, dtype=np.int64)
        self.authors: list[str] = []
        self.commit_author = np.zeros(0, dtype=np.int32)
        self.files: list[FileTrack] = []
        self.file_dir = np.zeros(0, dtype=np.int32)
        self.dirs: list[str] = [""]
        self.dir_parent = np.full(1, -1, dtype=np.int32)
        self.tags: list[Tag] = []
        self.lineage = np.zeros(0, dtype=np.int32)
        self.head_loc: dict[str, int] | None = None
        self._alive_paths: dict[str, int] = {}

    @classmethod
    def build(cls, commits: list[Commit], tags: list[Tag] = (),
              head_loc: dict[str, int] | None = None) -> "RepoHistory":
        h = cls()
        h.commits = list(commits)
        h.tags = list(tags)
        h.head_loc = head_loc
        n = len(h.commits)
        h.commit_ts = np.array([cm.ts for cm in h.commits], dtype=np.int64)

        author_ids: dict[str, int] = {}
        ca = np.zeros(n, dtype=np.int32)
        builders: list[_Builder] = []
        alive: dict[str, int] = {}

        def new_file(path: str, i: int) -> int:
            builders.append(_Builder(path, i))
            alive[path] = len(builders) - 1
            return len(builders) - 1

        for i, cm in enumerate(h.commits):
            aid = author_ids.setdefault(cm.author, len(author_ids))
            ca[i] = aid
            for ch in cm.changes:
                st = ch.status
                if st == "R":
                    fid = alive.pop(ch.old_path, None) if ch.old_path else None
                    if fid is None:
                        fid = new_file(ch.path, i)
                    else:
                        alive[ch.path] = fid
                        builders[fid].paths.append((i, ch.path))
                elif st == "A":
                    fid = alive.get(ch.path)
                    if fid is None:
                        fid = new_file(ch.path, i)
                else:  # M, T, D
                    fid = alive.get(ch.path)
                    if fid is None:
                        fid = new_file(ch.path, i)
                b = builders[fid]
                b.ec.append(i)
                b.ea.append(ch.added)
                b.ed.append(ch.deleted)
                b.eu.append(aid)
                b.binary = ch.binary
                if st == "D":
                    b.died = i
                    del alive[ch.path]

        h.authors = list(author_ids)
        h.commit_author = ca
        h._alive_paths = alive

        dir_ids: dict[str, int] = {"": 0}
        dirs = [""]
        parents = [-1]

        def dir_id(d: str) -> int:
            got = dir_ids.get(d)
            if got is not None:
                return got
            pid = dir_id(_dir_of(d))
            dir_ids[d] = len(dirs)
            dirs.append(d)
            parents.append(pid)
            return dir_ids[d]

        i32 = np.int32
        for b in builders:
            for _, p in b.paths:
                dir_id(_dir_of(p))
            h.files.append(FileTrack(
                path=b.paths[-1][1], paths=b.paths,
                event_commit=np.array(b.ec, dtype=i32),
                event_added=np.array(b.ea, dtype=i32),
                event_deleted=np.array(b.ed, dtype=i32),
                event_author=np.array(b.eu, dtype=i32),
                born=b.born, died=b.died, binary=b.binary))
        h.file_dir = np.array([dir_ids[_dir_of(f.path)] for f in h.files], dtype=i32)
        h.dirs = dirs
        h.dir_parent = np.array(parents, dtype=i32)
        h.lineage = np.zeros(n, dtype=i32)
        return h

    def files_alive_at(self, i: int) -> np.ndarray:
        out = np.zeros(len(self.files), dtype=bool)
        for k, f in enumerate(self.files):
            out[k] = f.born <= i and (f.died == -1 or f.died > i)
        return out

    def loc_at(self, i: int) -> np.ndarray:
        alive = self.files_alive_at(i)
        out = np.zeros(len(self.files), dtype=np.int64)
        use_head = self.head_loc is not None and i == len(self.commits) - 1
        for k in np.flatnonzero(alive):
            f = self.files[k]
            if use_head and f.path in self.head_loc:
                out[k] = self.head_loc[f.path]
                continue
            m = int(np.searchsorted(f.event_commit, i, side="right"))
            total = int(f.event_added[:m].sum(dtype=np.int64)
                        - f.event_deleted[:m].sum(dtype=np.int64))
            out[k] = max(total, 0)
        return out

    def index_at_time(self, ts: int) -> int:
        return int(np.searchsorted(self.commit_ts, ts, side="right")) - 1

    def file_id(self, path: str) -> int | None:
        return self._alive_paths.get(path)
