"""Deterministic 3D layout: directories are stars, file placements are planets."""
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from constellation.metrics import CouplingPair
from constellation.model import RepoHistory

_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))
_STEP = 0.05
_K_HOME = 1.0
_K_PAIR = 0.3


def _fib(n: int) -> np.ndarray:
    """n roughly even unit vectors on a sphere."""
    if n == 0:
        return np.zeros((0, 3))
    i = np.arange(n)
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    t = _GOLDEN * i
    return np.stack([r * np.cos(t), y, r * np.sin(t)], axis=1)


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


@dataclass
class Layout:
    dir_pos: np.ndarray          # (n_dirs, 3) float32
    dir_radius: np.ndarray       # (n_dirs,) float32, orbit radius of the dir's files
    place_file: np.ndarray       # (n_places,) int32
    place_dir: np.ndarray        # (n_places,) int32
    place_pos: np.ndarray        # (n_places, 3) float32
    file_place_start: np.ndarray  # (n_files,) int32
    place_commit: np.ndarray     # (n_places,) int32, commit where the placement starts

    def place_at(self, file_id: int, commit: int) -> int:
        """Placement index active at `commit` (first one before birth)."""
        s = int(self.file_place_start[file_id])
        e = (int(self.file_place_start[file_id + 1])
             if file_id + 1 < len(self.file_place_start) else len(self.place_file))
        k = int(np.searchsorted(self.place_commit[s:e], commit, side="right")) - 1
        return s + max(k, 0)

    def bounds(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Bounding sphere (AABB midpoint, max distance) of placements plus stars."""
        pp = self.place_pos if mask is None else self.place_pos[mask]
        pts = np.concatenate([pp, self.dir_pos]).astype(np.float64)
        if len(pp) == 0 and len(self.dir_pos) == 0:
            return np.zeros(3), 10.0
        if mask is not None and len(pp) == 0:
            return np.zeros(3), 10.0
        centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
        return centre, float(np.linalg.norm(pts - centre, axis=1).max())


def compute_layout(h: RepoHistory, pairs: Sequence[CouplingPair] = (),
                   seed: int = 0, iterations: int = 60) -> Layout:
    rng = np.random.default_rng(seed)
    n_dirs = len(h.dirs)
    dir_id = {d: i for i, d in enumerate(h.dirs)}

    pf, pd, pc, starts = [], [], [], []
    for fid, f in enumerate(h.files):
        starts.append(len(pf))
        for commit, path in f.paths:
            pf.append(fid)
            pd.append(dir_id[path.rpartition("/")[0]])
            pc.append(commit)
    place_file = np.array(pf, dtype=np.int32)
    place_dir = np.array(pd, dtype=np.int32)
    place_commit = np.array(pc, dtype=np.int32)
    file_place_start = np.array(starts, dtype=np.int32)
    n_places = len(pf)

    parent = np.asarray(h.dir_parent, dtype=np.int64)
    depth = np.zeros(n_dirs, dtype=np.int64)
    for d in range(n_dirs):
        p, k = parent[d], 0
        while p >= 0:
            p, k = parent[p], k + 1
        depth[d] = k

    own = np.bincount(place_dir, minlength=n_dirs)
    subtree = own.astype(np.int64)
    for d in np.argsort(-depth, kind="stable"):
        if parent[d] >= 0:
            subtree[parent[d]] += subtree[d]

    # 1. stars
    dir_pos = np.zeros((n_dirs, 3))
    children: list[list[int]] = [[] for _ in range(n_dirs)]
    for d in range(n_dirs):
        if parent[d] >= 0:
            children[parent[d]].append(d)
    for p in np.argsort(depth, kind="stable"):
        ch = sorted(children[p], key=lambda c: (-subtree[c], c))
        if not ch:
            continue
        dirs3 = _fib(len(ch)) @ _rotation(rng).T
        dist = 8.0 + 2.5 * np.sqrt(subtree[ch])
        dir_pos[ch] = dir_pos[p] + dirs3 * dist[:, None]

    # 2. placements
    dir_radius = 2.0 + 0.35 * np.sqrt(own)
    home = np.zeros((n_places, 3))
    if n_places:
        order = np.argsort(place_dir, kind="stable")
        bounds = np.searchsorted(place_dir[order], np.arange(n_dirs + 1))
        for d in range(n_dirs):
            idx = order[bounds[d]:bounds[d + 1]]
            if len(idx) == 0:
                continue
            r = dir_radius[d]
            pos = (_fib(len(idx)) @ _rotation(rng).T) * r
            pos += rng.normal(size=pos.shape) * (0.15 * r / np.sqrt(3))
            home[idx] = dir_pos[d] + pos

    # 3. relaxation
    pos = home.copy()
    limit = 1.5 * dir_radius[place_dir]
    star = dir_pos[place_dir]
    if n_places and pairs and iterations > 0:
        last = np.array([(starts[i + 1] if i + 1 < len(starts) else n_places) - 1
                         for i in range(len(starts))], dtype=np.int64)
        a = last[[p.a for p in pairs]]
        b = last[[p.b for p in pairs]]
        w = np.array([p.strength for p in pairs]) * _K_PAIR
        keep = a != b
        a, b, w = a[keep], b[keep], w[keep]
    else:
        a = b = np.zeros(0, dtype=np.int64)
        w = np.zeros(0)
    for _ in range(iterations if n_places else 0):
        force = _K_HOME * (home - pos)
        if len(a):
            pull = (pos[b] - pos[a]) * w[:, None]
            for ax in range(3):
                force[:, ax] += (np.bincount(a, pull[:, ax], n_places)
                                 - np.bincount(b, pull[:, ax], n_places))
        pos += _STEP * force
        off = pos - star
        norm = np.linalg.norm(off, axis=1)
        scale = np.where(norm > limit, limit / np.maximum(norm, 1e-12), 1.0)
        pos = star + off * scale[:, None]

    return Layout(dir_pos.astype(np.float32), dir_radius.astype(np.float32),
                  place_file, place_dir, pos.astype(np.float32),
                  file_place_start, place_commit)
