"""Directory collapse for very large repos: deep, crowded dirs become one planet."""
from dataclasses import dataclass, field

import numpy as np

from constellation.model import RepoHistory, Window

MAX_FILES = 10_000
MIN_DEPTH = 3     # collapse only dirs strictly deeper than this
MIN_SUBTREE = 200  # ... whose subtree holds more files than this


@dataclass
class CollapsePlan:
    member_to_group: np.ndarray  # (n_files,) int32, -1 = not collapsed
    group_dir: list[int]
    expanded: set[int] = field(default_factory=set)

    def expand(self, group: int) -> None:
        self.expanded.add(group)


def in_window(h: RepoHistory, w: Window) -> np.ndarray:
    """Files alive at some commit of the window."""
    if w.empty:
        return np.zeros(len(h.files), dtype=bool)
    return (h._born <= w.end) & ((h._died == -1) | (h._died > w.start))


def dir_depth(h: RepoHistory) -> np.ndarray:
    """Path components per dir ('' = 0, 'a/b' = 2)."""
    return np.array([0 if d == "" else d.count("/") + 1 for d in h.dirs], dtype=np.int64)


def plan(h: RepoHistory, w: Window) -> CollapsePlan:
    """Decided once from the union of files ever alive in the window. A file
    belongs to the shallowest qualifying ancestor of its final directory."""
    n = len(h.files)
    m2g = np.full(n, -1, dtype=np.int32)
    union = in_window(h, w)
    if int(union.sum()) <= MAX_FILES:
        return CollapsePlan(m2g, [])
    parent = h.dir_parent
    depth = dir_depth(h)
    sub = np.bincount(h.file_dir[union], minlength=len(h.dirs)).astype(np.int64)
    for d in np.argsort(-depth, kind="stable"):
        if parent[d] >= 0:
            sub[parent[d]] += sub[d]
    qualifies = (depth > MIN_DEPTH) & (sub > MIN_SUBTREE)
    owner = np.full(len(h.dirs), -1, dtype=np.int64)  # shallowest qualifying ancestor
    groups: list[int] = []
    for d in np.argsort(depth, kind="stable"):
        p = parent[d]
        if p >= 0 and owner[p] >= 0:
            owner[d] = owner[p]
        elif qualifies[d]:
            owner[d] = len(groups)
            groups.append(int(d))
    m2g[union] = owner[h.file_dir[union]]
    return CollapsePlan(m2g, groups)
