"""Plain-language findings derived from the metrics."""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from constellation.metrics import (FOSSIL_DAYS, CouplingIndex, bus_factor_risk,
                                   change_counts, fossil_age_days, hotspots,
                                   ownership)
from constellation.model import RepoHistory, Window

MAX_PER_KIND = 10


@dataclass(frozen=True)
class Finding:
    kind: str
    title: str
    detail: str
    file_ids: tuple[int, ...]


def _ranked(ids: np.ndarray, primary: np.ndarray, h: RepoHistory) -> list[int]:
    """File ids ordered by primary descending, then path ascending."""
    return sorted(ids.tolist(), key=lambda i: (-primary[i], h.files[i].path))[:MAX_PER_KIND]


def findings(h: RepoHistory, w: Window, playhead: int,
             coupling: CouplingIndex | None = None) -> list[Finding]:
    if not h.commits:
        return []
    out = [_summary(h, w, playhead)]
    if w.empty:
        return out
    path = lambda i: h.files[i].path  # noqa: E731
    changes = change_counts(h, w.start, playhead)
    own = ownership(h, w.start, playhead)
    loc = h.loc_at(playhead)

    score = hotspots(h, w, playhead)
    for i in _ranked(np.flatnonzero(score > 0), score, h):
        author = h.authors[own.dominant[i]]
        out.append(Finding(
            "hotspot", f"{path(i)} is a hotspot",
            f"Changed {changes[i]} times, {loc[i]} lines, mostly by {author} "
            f"({round(own.share[i] * 100)}%).", (i,)))

    if coupling is None:
        coupling = CouplingIndex(h)
    shown = 0
    alive = h.files_alive_at(playhead)
    for p in coupling.pairs(w.start, playhead):
        if h.file_dir[p.a] == h.file_dir[p.b] or not (alive[p.a] and alive[p.b]):
            continue
        out.append(Finding(
            "coupling", f"{path(p.a)} ↔ {path(p.b)}",
            f"Change together {round(p.strength * 100)}% of the time "
            f"({p.shared} commits) across different folders.", (p.a, p.b)))
        shown += 1
        if shown == MAX_PER_KIND:
            break

    risk = bus_factor_risk(h, w.start, playhead)
    now = int(h.commit_ts[playhead])
    in_range = h.commit_author[:playhead + 1]
    for i in _ranked(np.flatnonzero(risk), changes, h):
        dom = int(own.dominant[i])
        last = int(np.flatnonzero(in_range == dom).max())
        days = max(0, (now - int(h.commit_ts[last])) // 86400)
        out.append(Finding(
            "bus_factor", f"{path(i)} depends on {h.authors[dom]}",
            f"{h.authors[dom]} wrote {round(own.share[i] * 100)}% of it and "
            f"hasn't committed in {days} days.", (i,)))

    age = fossil_age_days(h, playhead)
    with np.errstate(invalid="ignore"):
        old = (age > FOSSIL_DAYS) & (loc > 0)
    for i in _ranked(np.flatnonzero(old), loc, h):
        out.append(Finding("fossil", f"{path(i)} is a fossil",
                           f"Untouched for {int(age[i])} days ({loc[i]} lines).", (i,)))
    return out


def _summary(h: RepoHistory, w: Window, playhead: int) -> Finding:
    n_files = int(h.files_alive_at(playhead).sum())
    if w.empty:
        return Finding("summary", f"0 commits by 0 authors, {n_files} files",
                       "No commits in the selected range.", ())
    n_authors = len(set(h.commit_author[w.start:w.end + 1].tolist()))
    first = datetime.fromtimestamp(int(h.commit_ts[w.start]), timezone.utc)
    last = datetime.fromtimestamp(int(h.commit_ts[w.end]), timezone.utc)
    return Finding("summary",
                   f"{w.size} commits by {n_authors} authors, {n_files} files",
                   f"From {first:%Y-%m-%d} to {last:%Y-%m-%d}.", ())
