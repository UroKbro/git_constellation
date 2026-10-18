"""Panel and tooltip text for files, commits and collapsed directories."""
from datetime import datetime

import numpy as np

from constellation.insights import _plural
from constellation.metrics import (CouplingIndex, change_counts, hotspots,
                                   ownership)
from constellation.model import RepoHistory, Window

MAX_COUPLED = 5


def _day(ts: int) -> str:
    return f"{datetime.fromtimestamp(ts):%Y-%m-%d}"


def file_details(h: RepoHistory, w: Window, playhead: int, file_id: int,
                 coupling: CouplingIndex | None) -> list[str]:
    f = h.files[file_id]
    n = int(change_counts(h, w.start, playhead)[file_id])
    lines = [f.path, f"{int(h.loc_at(playhead)[file_id])} lines · {_plural(n, 'change')}"]
    top = ownership(h, w.start, playhead).top3[file_id]
    lines.append("Authors: " + (", ".join(
        f"{h.authors[a]} {round(s * 100)}%" for a, s in top) or "none"))
    ev = f.event_commit[f.event_commit <= playhead]
    first, last = (int(ev[0]), int(ev[-1])) if len(ev) else (f.born, f.born)
    lines.append(f"First touched {_day(h.commits[first].ts)} · "
                 f"last touched {_day(h.commits[last].ts)}")
    lines.append(f"Hotspot {float(hotspots(h, w, playhead)[file_id]):.2f}")
    pairs = coupling.pairs(w.start, playhead, min_shared=2, min_strength=0.0) if coupling else []
    pairs = [p for p in pairs if file_id in (p.a, p.b)]
    if not pairs:
        lines.append("Coupled with: none")
    else:
        lines.append("Coupled with:")
        for p in pairs[:MAX_COUPLED]:
            other = p.b if p.a == file_id else p.a
            lines.append(f"  {h.files[other].path} ({round(p.strength * 100)}%)")
    return lines


def commit_details(h: RepoHistory, index: int) -> list[str]:
    c = h.commits[index]
    when = datetime.fromtimestamp(c.ts)
    lines = [f"{c.sha[:10]} · {c.author} · {when:%Y-%m-%d %H:%M}", c.subject]
    if len(c.parents) > 1:
        lines.append(f"merge of {len(c.parents)} parents")
    else:
        added = sum(x.added for x in c.changes)
        deleted = sum(x.deleted for x in c.changes)
        lines.append(f"{_plural(len(c.changes), 'file')} · +{added} −{deleted}")
    return lines


def tooltip(h: RepoHistory, playhead: int, file_id: int) -> str:
    return f"{h.files[file_id].path} — {int(h.loc_at(playhead)[file_id])} lines"


def aggregate_tooltip(h: RepoHistory, plan, group: int, playhead: int) -> str:
    alive = h.files_alive_at(playhead)
    n = int(np.count_nonzero((plan.member_to_group == group) & alive))
    return f"{h.dirs[plan.group_dir[group]]}/ — {_plural(n, 'file')} (double-click to expand)"
