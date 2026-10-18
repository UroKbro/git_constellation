"""Command line entry: python -m constellation PATH [--report out.html | --summary]."""
import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from constellation.cache import load_commits
from constellation.git_reader import (NotARepoError, UnknownRefError, head_loc,
                                      head_sha, read_tags, repo_root)
from constellation.insights import findings
from constellation.model import RepoHistory
from constellation.report import render_report


def _date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="constellation",
                                 description="Visualize git history as a constellation.")
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--branch", default="HEAD")
    ap.add_argument("--since", type=_date, metavar="YYYY-MM-DD")
    ap.add_argument("--until", type=_date, metavar="YYYY-MM-DD")
    ap.add_argument("--report", metavar="PATH")
    ap.add_argument("--summary", action="store_true",
                    help="print findings as text instead of opening the galaxy window")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--screenshot", metavar="PATH", help=argparse.SUPPRESS)
    ap.add_argument("--at", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--display", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--bench", type=float, metavar="SECONDS", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    try:
        return _run(args)
    except FileNotFoundError as e:
        if e.filename != "git":
            print(f"error: {e}", file=sys.stderr)
        else:
            print("error: git not found on PATH", file=sys.stderr)
        return 1
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    try:
        root = repo_root(Path(args.path))
        head = head_sha(root, args.branch)
    except NotARepoError:
        print(f"error: {args.path} is not inside a git repository", file=sys.stderr)
        return 1
    except UnknownRefError:
        print(f"error: unknown branch or ref '{args.branch}'", file=sys.stderr)
        return 1
    since = None if args.since is None else int(args.since.timestamp())
    until = None if args.until is None else int(
        (args.until + timedelta(days=1)).timestamp()) - 1
    if head is None:
        print("No commits yet.")
    if not (args.report or args.summary):
        from constellation import app  # the only path that imports pygame
        return app.run(root, args.branch, since, until, not args.no_cache,
                       f"git-constellation — {root.name}", bench=args.bench,
                       screenshot=args.screenshot, at=args.at, display=args.display)
    if head is None:
        return 0

    commits = load_commits(
        root, args.branch, use_cache=not args.no_cache,
        progress=lambda n: print(f"Reading history… {n} commits", file=sys.stderr))
    h = RepoHistory.build(commits, read_tags(root), head_loc(root, args.branch))
    w = h.window(since, until)
    playhead = w.end if not w.empty else len(h.commits) - 1
    items = findings(h, w, playhead)

    if args.report:
        Path(args.report).write_text(render_report(root.name, h, w, items), encoding="utf-8")
        print(f"Report written to {args.report}")
    else:
        for f in items:
            print(f.title)
            print(f"  {f.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
