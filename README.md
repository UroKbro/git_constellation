# Git-Constellation

Turns a local Git repository's history into a picture of its health:
**hotspots** (big files that keep changing), **hidden dependencies** (files in
different folders that always change together), **bus-factor risks** (files
only one quiet author understands) and **fossils** (code nobody has touched in
over a year).

Today it produces a plain-text summary and a self-contained HTML report. The 3D
galaxy window — directories as stars, files as planets, commits as pulses of
light, with history replay — is being built in Phase 2 (see [Status](#status)).

Strictly read-only: it never writes to the repository it analyses.

## Requirements

- Python 3.14
- `git` on your `PATH`
- macOS (Apple Silicon) is the primary target; Linux and Windows should work

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

This installs the `git-constellation` command (equivalently `python -m constellation`).

## Usage

```sh
git-constellation [PATH]                          # PATH defaults to the current directory
git-constellation PATH --report health.html       # write the HTML report
git-constellation PATH --branch main --since 2024-01-01 --until 2024-12-31
git-constellation PATH --no-cache                 # re-read history, skip the cache
```

| Option | Meaning |
|---|---|
| `PATH` | Any directory inside a git repository |
| `--branch REF` | Branch, tag or commit to analyse (default `HEAD`) |
| `--since` / `--until YYYY-MM-DD` | Limit the analysis to a date range (local time, inclusive). Full history is still read so line counts stay correct |
| `--report PATH` | Write a single-file HTML report (no external resources; light and dark mode) |
| `--no-cache` | Don't read or write the history cache |

History is cached in SQLite under `~/.cache/git_constellation/` (override with
`GIT_CONSTELLATION_CACHE=/some/dir`). The first run on a large repo takes
seconds (Django, 35k commits: ~13 s); later runs reuse the cache and only parse
new commits (~1 s). If the cache can't be written, it warns and carries on
without it.

## What the findings mean

All metrics are computed from history up to the end of the selected range.
Merge commits are not counted as changes.

| Finding | Rule |
|---|---|
| Hotspot | Recency-weighted change count (half-life 180 days) × log₂(1 + lines). Scores are normalised to the busiest file; binary files are excluded |
| Hidden dependency | Two files in different folders changed together in ≥ 5 commits and in ≥ 50% of the commits touching the less-changed one. Commits touching more than 50 files (formatting sweeps, vendoring) are ignored |
| Bus-factor risk | One author wrote ≥ 80% of a file's churn, the file has ≥ 10 changes, and that author hasn't committed anything in the last 180 days |
| Fossil | A file with content that hasn't been touched for more than 365 days |

Author names are merged using the repository's `.mailmap`.

## Status

| Phase | Scope | State |
|---|---|---|
| 1. Foundations | git reading, cache, model, metrics, text summary, HTML report | Done (on `main`) |
| 2. Galaxy + History | window, camera, layout, planets/stars, bloom, timeline playback with pulses, hover/click details | In progress (`phase2-galaxy`) |
| 3. Hotspots lens | heat colouring, labels, insight panel — first shippable version | Planned |
| 4. Remaining lenses | coupling, ownership, fossils lenses; search; author filter | Planned |
| 5. Story features | guided tour, compare mode | Planned |
| 6. Export & live | screenshot, video recording, `--watch` | Planned |

When Phase 2 lands, running `git-constellation PATH` will open the galaxy
window, and the text summary will move to `--summary`.

The full design is in [`docs/design.md`](docs/design.md); per-phase
implementation plans are in [`docs/superpowers/plans/`](docs/superpowers/plans/).

## Development

```sh
.venv/bin/pytest                  # fast suite
.venv/bin/pytest -m slow          # performance and wheel-install tests
.venv/bin/pytest -m "slow or not slow"   # everything
```

Tests build small throwaway git repositories in temporary directories, so they
need `git` but no network. OpenGL tests skip automatically when no GL context
is available.

```
constellation/
  __main__.py     CLI
  git_reader.py   git plumbing → commit records
  cache.py        SQLite history cache with incremental append
  model.py        RepoHistory: file identity, renames, merges, state at any commit
  metrics/        hotspots, coupling, ownership / bus factor, fossils (numpy)
  insights.py     metrics → plain-language findings
  report.py       HTML report
  layout.py, camera.py, picking.py, render/   galaxy window (Phase 2, in progress)
tests/            pytest suite
docs/             design and implementation plans
```
