"""Ownership metrics: who owns each file, and bus-factor risk."""
from dataclasses import dataclass

import numpy as np

from constellation.metrics.activity import change_counts

BUS_SHARE = 0.8
BUS_MIN_CHANGES = 10
BUS_SILENT_DAYS = 180


@dataclass
class Ownership:
    dominant: np.ndarray  # int32 author id, -1 if no events
    share: np.ndarray  # float64 0..1
    top3: list[list[tuple[int, float]]]  # per file (author id, share), descending


def _events(h):
    """Flat per-event (file, commit, churn, author) arrays."""
    tracks = h.files
    lens = np.fromiter((len(f.event_commit) for f in tracks), dtype=np.int64, count=len(tracks))
    if lens.sum() == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z
    fid = np.repeat(np.arange(len(tracks), dtype=np.int64), lens)
    commit = np.concatenate([f.event_commit for f in tracks]).astype(np.int64)
    churn = np.concatenate([np.asarray(f.event_added, dtype=np.int64) + f.event_deleted
                            for f in tracks])
    author = np.concatenate([f.event_author for f in tracks]).astype(np.int64)
    return fid, commit, churn, author


def ownership(h, start: int, playhead: int) -> Ownership:
    n = len(h.files)
    dominant = np.full(n, -1, dtype=np.int32)
    share = np.zeros(n, dtype=np.float64)
    top3: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    if start > playhead or n == 0:
        return Ownership(dominant, share, top3)
    fid, commit, churn, author = _events(h)
    m = (commit >= start) & (commit <= playhead)
    fid, churn, author = fid[m], churn[m], author[m]
    if len(fid) == 0:
        return Ownership(dominant, share, top3)
    file_churn = np.bincount(fid, weights=churn, minlength=n)
    w = np.where(file_churn[fid] > 0, churn, 1).astype(np.float64)
    na = len(h.authors)
    keys, inv = np.unique(fid * na + author, return_inverse=True)
    sums = np.bincount(inv, weights=w)
    kf, ka = keys // na, keys % na
    total = np.bincount(fid, weights=w, minlength=n)
    frac = sums / total[kf]
    order = np.lexsort((ka, -sums, kf))  # file, then sum desc, then author id asc
    kf, ka, frac = kf[order], ka[order], frac[order]
    first = np.concatenate(([True], kf[1:] != kf[:-1]))
    dominant[kf[first]] = ka[first]
    share[kf[first]] = frac[first]
    bounds = np.flatnonzero(first).tolist() + [len(kf)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        top3[int(kf[a])] = [(int(ka[j]), float(frac[j])) for j in range(a, min(b, a + 3))]
    return Ownership(dominant, share, top3)


def bus_factor_risk(h, start: int, playhead: int) -> np.ndarray:
    n = len(h.files)
    if start > playhead or n == 0:
        return np.zeros(n, dtype=bool)
    o = ownership(h, start, playhead)
    ts = h.commit_ts[:playhead + 1]
    recent = (ts > h.commit_ts[playhead] - BUS_SILENT_DAYS * 86400) & (ts <= h.commit_ts[playhead])
    active = np.zeros(len(h.authors) + 1, dtype=bool)  # last slot absorbs dominant == -1
    active[h.commit_author[:playhead + 1][recent]] = True
    silent = ~active[o.dominant]
    return (h.files_alive_at(playhead) & (o.share >= BUS_SHARE)
            & (change_counts(h, start, playhead) >= BUS_MIN_CHANGES) & silent & (o.dominant >= 0))
