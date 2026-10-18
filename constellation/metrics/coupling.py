"""Coupling: files that change together, via an incremental pair-count index."""
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import ceil

import numpy as np

from constellation.metrics.activity import change_counts

MAX_COMMIT_FILES = 50
MIN_SHARED = 5
MIN_STRENGTH = 0.5
MAX_SNAPSHOTS = 20


@dataclass(frozen=True)
class CouplingPair:
    a: int
    b: int
    shared: int
    strength: float


class CouplingIndex:
    def __init__(self, h, snapshot_every: int = 500):
        self.h = h
        n = len(h.commits)
        self._nf = max(1, len(h.files))
        self.snapshot_every = max(snapshot_every, ceil(n / MAX_SNAPSHOTS), 1)
        snapshot_every = self.snapshot_every
        self._pairs: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        ev_c, ev_f = h._ev_commit, h._ev_file
        cuts = np.flatnonzero(np.diff(ev_c)) + 1 if len(ev_c) else np.zeros(0, dtype=np.int64)
        for fs, cs in zip(np.split(ev_f, cuts), np.split(ev_c, cuts)):
            if not len(fs):
                continue
            files = sorted(set(fs.tolist()))
            if 2 <= len(files) <= MAX_COMMIT_FILES:
                self._pairs[int(cs[0])] = list(combinations(files, 2))
        self._snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        running: Counter = Counter()
        for i in range(n):
            running.update(self._pairs[i])
            if i % snapshot_every == 0:
                self._snapshots[i] = self._encode(running)

    def _encode(self, counter: Counter) -> tuple[np.ndarray, np.ndarray]:
        keys = np.fromiter((a * self._nf + b for a, b in counter),
                           dtype=np.int64, count=len(counter))
        cnt = np.fromiter(counter.values(), dtype=np.int32, count=len(counter))
        order = np.argsort(keys)
        return keys[order], cnt[order]

    def _decode(self, snap: tuple[np.ndarray, np.ndarray]) -> Counter:
        keys, cnt = snap
        a, b = np.divmod(keys, self._nf)
        return Counter(dict(zip(zip(a.tolist(), b.tolist()), cnt.tolist())))

    def shared_counts(self, upto: int) -> Counter:
        upto = min(upto, len(self.h.commits) - 1)
        if upto < 0:
            return Counter()
        base = upto // self.snapshot_every * self.snapshot_every
        out = self._decode(self._snapshots[base])
        for i in range(base + 1, upto + 1):
            out.update(self._pairs[i])
        return out

    def pairs(self, start: int, playhead: int, min_shared: int = MIN_SHARED,
              min_strength: float = MIN_STRENGTH) -> list[CouplingPair]:
        if start > playhead:
            return []
        shared = self.shared_counts(playhead)
        shared.subtract(self.shared_counts(start - 1))
        counts = change_counts(self.h, start, playhead)
        out = []
        for (a, b), s in shared.items():
            if s < min_shared:
                continue
            denom = min(counts[a], counts[b])
            strength = s / denom if denom else 0.0
            if strength >= min_strength:
                out.append(CouplingPair(a, b, s, strength))
        out.sort(key=lambda p: (p.strength, p.shared), reverse=True)
        return out
