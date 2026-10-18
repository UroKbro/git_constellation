"""RepoHistory: the core data structure consumed by metrics and the renderer."""
from dataclasses import dataclass

import numpy as np

from constellation.types import Commit, FileChange, Tag


def assign_lineage(commits: list[Commit]) -> np.ndarray:
    """Mainline (first-parent chain from the last commit) is 0; commits brought in
    by the k-th mainline merge's non-first parents get k; unreachable ones get 0."""
    n = len(commits)
    out = np.zeros(n, dtype=np.int32)
    if n == 0:
        return out
    idx = {cm.sha: i for i, cm in enumerate(commits)}
    parents = [[idx[p] for p in cm.parents if p in idx] for cm in commits]
    assigned = np.zeros(n, dtype=bool)
    i = n - 1
    while i != -1 and not assigned[i]:
        assigned[i] = True
        i = parents[i][0] if parents[i] and commits[i].parents[0] in idx else -1
    mainline_merges = [i for i in range(n) if assigned[i] and len(parents[i]) > 1]
    k = 0
    for m in mainline_merges:
        k += 1
        stack = [p for p in parents[m][1:] if not assigned[p]]
        while stack:
            j = stack.pop()
            if assigned[j]:
                continue
            assigned[j] = True
            out[j] = k
            stack.extend(p for p in parents[j] if not assigned[p])
    return out


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    @property
    def empty(self) -> bool:
        return self.start > self.end

    @property
    def size(self) -> int:
        return max(0, self.end - self.start + 1)


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
    __slots__ = ("paths", "ec", "ea", "ed", "eu", "born", "died", "binary", "phantom")

    def __init__(self, path: str, born: int):
        self.phantom = False
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
        self._ts_runmax = np.zeros(0, dtype=np.int64)
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
        self._born = np.zeros(0, dtype=np.int64)
        self._died = np.zeros(0, dtype=np.int64)
        self._ev_file = np.zeros(0, dtype=np.int64)
        self._ev_commit = np.zeros(0, dtype=np.int64)
        self._ev_delta = np.zeros(0, dtype=np.int64)
        self._head: np.ndarray | None = None

    @classmethod
    def build(cls, commits: list[Commit], tags: list[Tag] = (),
              head_loc: dict[str, int] | None = None) -> "RepoHistory":
        h = cls()
        h.commits = list(commits)
        h.tags = list(tags)
        h.head_loc = head_loc
        n = len(h.commits)
        h.commit_ts = np.array([cm.ts for cm in h.commits], dtype=np.int64)
        h._ts_runmax = np.maximum.accumulate(h.commit_ts)

        author_ids: dict[str, int] = {}
        ca = np.zeros(n, dtype=np.int32)
        builders: list[_Builder] = []
        alive: dict[str, int] = {}
        erased: set[int] = set()
        resorted: set[int] = set()

        def fold(fid: int, mc: tuple[FileChange, ...]) -> None:
            """Merge a dying phantom's events into the file it really belongs to."""
            b = builders[fid]
            path = b.paths[-1][1]
            tgt = next((alive[ch.path] for ch in mc if ch.status == "R"
                        and ch.old_path == path and ch.path in alive), None)
            if tgt is None:
                tgt = next((j for j in range(fid - 1, -1, -1)
                            if j not in erased and builders[j].died != -1
                            and builders[j].paths[-1][1] == path), None)
            if tgt is None:
                return
            t = builders[tgt]
            t.ec += b.ec
            t.ea += b.ea
            t.ed += b.ed
            t.eu += b.eu
            resorted.add(tgt)
            erased.add(fid)

        def new_file(path: str, i: int) -> int:
            builders.append(_Builder(path, i))
            alive[path] = len(builders) - 1
            return len(builders) - 1

        def reconcile(mc: tuple[FileChange, ...], i: int) -> None:
            """Move the alive set toward a merge's tree: the new side of every
            per-parent diff is that tree. Idempotent; records no events."""
            present = {ch.path for ch in mc if ch.status != "D"}
            for ch in mc:  # renames whose target is missing keep their identity
                if (ch.status == "R" and ch.path not in alive
                        and ch.old_path in alive and ch.old_path not in present):
                    fid = alive.pop(ch.old_path)
                    alive[ch.path] = fid
                    builders[fid].paths.append((i, ch.path))
            for ch in mc:  # paths absent from the merge tree die
                gone = ch.path if ch.status == "D" else (
                    ch.old_path if ch.status == "R" else None)
                if gone is not None and gone not in present and gone in alive:
                    fid = alive.pop(gone)
                    builders[fid].died = i
                    if builders[fid].phantom:
                        fold(fid, mc)
            for ch in mc:  # paths in the merge tree but not alive are born
                if ch.status != "D" and ch.path not in alive:
                    builders[new_file(ch.path, i)].binary = ch.binary

        for i, cm in enumerate(h.commits):
            aid = author_ids.setdefault(cm.author, len(author_ids))
            ca[i] = aid
            if cm.merge_changes:
                reconcile(cm.merge_changes, i)
            for ch in cm.changes:
                st = ch.status
                if st == "R":
                    # A rename onto a still-alive path replaces that file.
                    victim = alive.get(ch.path)
                    if victim is not None:
                        builders[victim].died = i
                        del alive[ch.path]
                    fid = alive.pop(ch.old_path, None) if ch.old_path else None
                    if fid is None:
                        fid = new_file(ch.path, i)
                    else:
                        alive[ch.path] = fid
                        builders[fid].paths.append((i, ch.path))
                elif st == "A":
                    # Deliberate: "A" on an already-alive path merges into the
                    # existing id (only dead/unknown paths get a new id).
                    fid = alive.get(ch.path)
                    if fid is None:
                        fid = new_file(ch.path, i)
                else:  # M, T, D
                    fid = alive.get(ch.path)
                    if fid is None:
                        fid = new_file(ch.path, i)
                        builders[fid].phantom = True
                b = builders[fid]
                b.ec.append(i)
                b.ea.append(ch.added)
                b.ed.append(ch.deleted)
                b.eu.append(aid)
                b.binary = ch.binary
                if st == "D":
                    b.died = i
                    del alive[ch.path]

        remap: dict[int, int] = {}
        for j in range(len(builders)):
            if j not in erased:
                remap[j] = len(remap)
        for j in resorted:
            b = builders[j]
            o = np.argsort(b.ec, kind="stable")
            b.ec, b.ea, b.ed, b.eu = ([getattr(b, k)[x] for x in o]
                                      for k in ("ec", "ea", "ed", "eu"))
        builders = [b for j, b in enumerate(builders) if j not in erased]
        alive = {p: remap[f] for p, f in alive.items()}

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
        h.lineage = assign_lineage(h.commits)
        h._born = np.array([f.born for f in h.files], dtype=np.int64)
        h._died = np.array([f.died for f in h.files], dtype=np.int64)
        ev_c = [f.event_commit for f in h.files]
        if ev_c:
            ids = np.repeat(np.arange(len(h.files), dtype=np.int64),
                            [len(x) for x in ev_c])
            com = np.concatenate(ev_c).astype(np.int64)
            dl = np.concatenate([f.event_added.astype(np.int64) - f.event_deleted
                                 for f in h.files])
            order = np.argsort(com, kind="stable")
            h._ev_file, h._ev_commit, h._ev_delta = ids[order], com[order], dl[order]
        if head_loc is not None:
            h._head = np.array([head_loc.get(f.path, -1) for f in h.files],
                               dtype=np.int64)
        return h

    def files_alive_at(self, i: int) -> np.ndarray:
        return (self._born <= i) & ((self._died == -1) | (self._died > i))

    def loc_at(self, i: int) -> np.ndarray:
        n = len(self.files)
        k = int(np.searchsorted(self._ev_commit, i, side="right"))
        out = np.bincount(self._ev_file[:k], weights=self._ev_delta[:k],
                          minlength=n).astype(np.int64)
        np.maximum(out, 0, out=out)
        if self._head is not None and i == len(self.commits) - 1:
            have = self._head >= 0
            out[have] = self._head[have]
        out[~self.files_alive_at(i)] = 0
        return out

    def index_at_time(self, ts: int) -> int:
        return int(np.searchsorted(self._ts_runmax, ts, side="right")) - 1

    def window(self, since: int | None = None, until: int | None = None) -> Window:
        start = 0 if since is None else int(
            np.searchsorted(self._ts_runmax, since, side="left"))
        end = len(self.commits) - 1 if until is None else self.index_at_time(until)
        return Window(start, end)

    def file_id(self, path: str) -> int | None:
        return self._alive_paths.get(path)
