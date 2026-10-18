"""Background loader: parse history, build model, coupling, layout and collapse plan."""
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from constellation.cache import load_commits
from constellation.collapse import CollapsePlan, plan
from constellation.git_reader import (NotARepoError, UnknownRefError, head_loc,
                                      head_sha, read_tags, repo_root)
from constellation.layout import Layout, compute_layout
from constellation.metrics import CouplingIndex
from constellation.model import RepoHistory, Window


@dataclass
class Loaded:
    root: Path
    h: RepoHistory
    window: Window
    layout: Layout
    coupling: CouplingIndex
    plan: CollapsePlan


def _load(q: queue.Queue, path: Path, ref: str, since: int | None,
          until: int | None, use_cache: bool) -> None:
    try:
        root = repo_root(path)
        head = head_sha(root, ref)
    except NotARepoError:
        q.put(("error", f"error: {path} is not inside a git repository"))
        return
    except UnknownRefError:
        q.put(("error", f"error: unknown branch or ref '{ref}'"))
        return
    if head is None:
        q.put(("done", None))
        return
    commits = load_commits(
        root, ref, use_cache=use_cache,
        progress=lambda n: q.put(("progress", f"Reading history… {n} commits", None)))
    h = RepoHistory.build(commits, read_tags(root), head_loc(root, ref))
    w = h.window(since, until)
    coupling = CouplingIndex(h)
    q.put(("progress", f"Laying out {len(h.files)} files…", None))
    pairs = () if w.empty else coupling.pairs(w.start, w.end)
    layout = compute_layout(h, pairs)
    q.put(("done", Loaded(root, h, w, layout, coupling, plan(h, w))))


def _worker(q: queue.Queue, *args) -> None:
    try:
        _load(q, *args)
    except FileNotFoundError as e:
        q.put(("error", "error: git not found on PATH" if e.filename == "git"
               else f"error: {e}"))
    except (subprocess.CalledProcessError, OSError) as e:
        q.put(("error", f"error: {e}"))
    except Exception as e:  # noqa: BLE001 - surface anything to the UI
        q.put(("error", f"error: {e}"))


def start_loading(root: Path, ref: str, since: int | None, until: int | None,
                  use_cache: bool) -> queue.Queue:
    """Start the loader daemon thread; messages arrive on the returned queue."""
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_worker, args=(q, root, ref, since, until, use_cache),
                     daemon=True).start()
    return q
