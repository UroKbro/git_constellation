"""Activity metrics: change counts, hotspot heat, fossil age."""
import numpy as np

FOSSIL_DAYS = 365
HALF_LIFE_DAYS = 180
_DAY = 86400.0


def _range(h, start: int, playhead: int) -> tuple[int, int]:
    lo = int(np.searchsorted(h._ev_commit, start, side="left"))
    hi = int(np.searchsorted(h._ev_commit, playhead, side="right"))
    return lo, max(lo, hi)


def change_counts(h, start: int, playhead: int) -> np.ndarray:
    n = len(h.files)
    if start > playhead:
        return np.zeros(n, dtype=np.int64)
    lo, hi = _range(h, start, playhead)
    return np.bincount(h._ev_file[lo:hi], minlength=n).astype(np.int64)


def hotspot_raw(h, start: int, playhead: int) -> np.ndarray:
    n = len(h.files)
    if start > playhead or n == 0:
        return np.zeros(n, dtype=np.float64)
    lo, hi = _range(h, start, playhead)
    age = np.maximum(0.0, (h.commit_ts[playhead] - h.commit_ts[h._ev_commit[lo:hi]]) / _DAY)
    weights = np.bincount(h._ev_file[lo:hi], weights=0.5 ** (age / HALF_LIFE_DAYS),
                          minlength=n)
    binary = np.fromiter((f.binary for f in h.files), dtype=bool, count=n)
    mask = h.files_alive_at(playhead) & ~binary
    return np.where(mask, weights * np.log2(1 + h.loc_at(playhead)), 0.0)


def hotspots(h, w, playhead: int) -> np.ndarray:
    if w.empty:
        return np.zeros(len(h.files), dtype=np.float64)
    peak = hotspot_raw(h, w.start, w.end).max(initial=0.0)
    if peak <= 0:
        return np.zeros(len(h.files), dtype=np.float64)
    return np.clip(hotspot_raw(h, w.start, playhead) / peak, 0.0, 1.0)


def fossil_age_days(h, playhead: int) -> np.ndarray:
    n = len(h.files)
    k = int(np.searchsorted(h._ev_commit, playhead, side="right"))
    last = np.zeros(n, dtype=np.int64)
    np.maximum.at(last, h._ev_file[:k], h._ev_commit[:k])
    age = np.maximum(0.0, (h.commit_ts[playhead] - h.commit_ts[last]) / _DAY)
    return np.where(h.files_alive_at(playhead), age, np.nan)
