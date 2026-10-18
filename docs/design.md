# Git-Constellation — Design

Date: 2026-10-17
Status: draft, awaiting approval

## 1. Intent

Git-Constellation is a Python desktop app that turns a local Git repository's
history into a living 3D galaxy. Directories are star systems, files are
planets, commits are pulses of light. It must be **beautiful enough to watch**
and **useful enough to open on a real repo**.

**Purpose: see the health and story of a codebase at a glance.** The target
moments are: joining an unfamiliar project, inheriting an old one, or asking
"why does my repo feel fragile?". Every visual property encodes real data —
nothing is decoration only.

The user handed most product decisions to the builder; choices below marked
*(builder's call)* were made on their behalf and may be revisited.

## 2. Goals, non-goals, anti-goals

**Goals**
- Open any local repo with one command and see a meaningful galaxy within
  seconds for small repos, and under ~60 s first-run for large ones
  (e.g. 50k commits / 10k files), near-instant on re-open via cache.
- Replay history as an animation (the "History" lens).
- Answer "where is the risk?" visually and in plain language (Hotspots first,
  then Coupling, Ownership, Fossils).
- Smooth interaction: 60 fps target on an Apple Silicon Mac at 10k planets.

**Non-goals**
- No remote data (GitHub/GitLab APIs, PRs, issues).
- No writing to the repo — strictly read-only.
- No static code analysis (complexity, linting); insight comes from history
  plus current line counts only.
- Not a `git log` / diff viewer replacement.

**Anti-goals** (failure even if it technically works)
- Pretty but meaningless: if a user can't name one real finding about their
  repo after 2 minutes, it failed.
- Unreadable soup: on large repos the galaxy must stay legible (clustering,
  dimming, labels for the important things only), not a hairball.
- Sluggish: a freezing UI while parsing is unacceptable — parse in a
  background thread with a progress display.

## 3. Constraints

- Python 3.14 (user's installed version), macOS Apple Silicon primary target.
  Should also run on Linux/Windows with no code changes where libs allow.
- Lives in `~/Python_Projects/git_constellation/`.
- Dependencies *(builder's call)*: `pygame-ce` (window, input, fonts),
  `moderngl` (GPU rendering, OpenGL 3.3 core — max available on macOS is 4.1,
  so nothing newer), `numpy`. Dev: `pytest`. All have cp314 arm64 wheels
  (verified). Requires the `git` CLI on PATH.
- Git access *(builder's call)*: **plumbing via `subprocess`**, not GitPython.
  A single `git log --numstat` stream is an order of magnitude faster than
  per-commit GitPython objects and adds no dependency.

## 4. User experience

### Launch
```
python -m constellation [PATH]        # PATH defaults to cwd
python -m constellation PATH --branch main --since 2024-01-01
python -m constellation PATH --report out.html   # headless, no window
```
Not a git repo → clear error message and exit code 1. Empty repo → window
opens with a centered "No commits yet" message.

### Window layout
- Full-window 3D scene, dark background with a faint starfield.
- **Bottom: timeline bar** — commit density histogram over time, a playhead,
  play/pause, speed (0.25×–64×), tag markers. Click/drag to scrub.
- **Right: insight panel** (toggle with `I`) — plain-language findings for the
  active lens, plus details of the selected file/commit.
- **Top-left: lens indicator** and search box (`/` to focus).
- Startup: progress bar ("Reading history… 12,340 commits") while parsing.

### The galaxy (shared by all lenses)
- **Star systems = directories.** A glowing core per directory; nested
  directories orbit their parent. Root is the galactic center.
- **Planets = files** existing at the playhead time. Radius ∝ sqrt(current
  line count), clamped. Base color = language family (by extension).
- **Placement:** hierarchical (planet stays near its directory's star) plus
  a co-change attraction so coupled files drift toward each other.
- **Commit comets:** at the playhead, each commit appears as a comet that
  fires pulses to every touched file. Pulse brightness ∝ lines changed.
  Added files flash in; deleted files collapse and fade; renames slide.
- **Branches:** commits laid on the timeline are colored by first-parent
  lineage; merge commits render a braid effect where two lineage colors
  meet. *(builder's call: show branch structure as color/braid on comets and
  the timeline, not as separate 3D geometry — keeps the scene legible.)*
- Bloom/glow post-processing on emissive elements.

### Lenses (keys `1`–`5`, all recolor the same galaxy)
1. **History** — the default. Language colors, replay animation.
2. **Hotspots** — planets heat from cool blue to white-hot red by hotspot
   score (§6). Top 10 get always-visible labels. Insight panel lists them
   with reasons.
3. **Coupling** — faint luminous lines between file pairs with high
   co-change; thickness ∝ strength. Selecting a file shows only its links.
   Panel lists strongest cross-directory pairs ("hidden dependencies").
4. **Ownership** — each top author gets a color (top 12, rest grey);
   planet color = dominant author. Warning halo when bus-factor risk (§6).
5. **Fossils** — color by age since last touch; untouched > 1 year fade to
   grey moons. Panel lists largest stale areas.

Lenses compute against the history **up to the playhead**, so scrubbing
shows how risk evolved.

### Interaction
- Mouse: left-drag orbit, right-drag pan, wheel zoom. Double-click a planet
  or star to fly to it.
- Hover: tooltip with path + key metric for the active lens.
- Click planet → panel shows: path, lines, total changes, top 3 authors with
  %, first/last touched, hotspot score, top 5 coupled files.
- Click comet → hash, author, date, message, diffstat.
- `/` search: fuzzy match over file paths, author names, commit messages.
  Matching objects stay lit, everything else dims to 15%; Enter flies to the
  best match. Esc clears.
- `A` author filter: cycle/choose an author; only their pulses and files
  stay lit.
- `T` guided tour (§5). `C` compare mode (§5). `Space` play/pause,
  `←/→` step one commit, `Home/End` jump to start/end.
- `F12` screenshot (PNG to `~/Pictures/git_constellation/` or cwd fallback).

## 5. Features beyond the core

- **Guided tour (`T`):** an automatic camera flight with captions:
  first commit → busiest month → largest refactor (commit with most files
  touched) → top 3 hotspots → most-coupled cross-directory pair → newest
  active area. ~2 minutes; any input cancels.
- **Compare (`C`):** pick two points (tags or timeline clicks). Planets are
  colored green (grew/new), red (shrank/deleted), white (unchanged); panel
  summarizes counts and biggest movers.
- **Export:**
  - Screenshot (F12).
  - Video recording (`R` toggles): frames streamed to `ffmpeg` if found on
    PATH, else a PNG frame sequence folder. Fixed 30 fps capture.
  - `--report out.html`: headless self-contained HTML health report — top
    hotspots, coupling pairs, bus-factor risks, fossils, summary stats.
- **Live mode (`--watch`):** polls `HEAD` every 2 s; new commits are
  appended incrementally and replayed as fresh pulses.

## 6. Metrics (definitions the builder must use)

All computed per file over commits in the selected range up to the
playhead. Merge commits are excluded from change counts (their changes are
already counted on the branch) *(builder's call)*.

- **changes(f):** number of non-merge commits touching f.
- **churn(f):** sum of added + deleted lines.
- **loc(f):** current line count from `git ls-tree`/blob size at the
  playhead commit (count newlines; binary files → 0 and excluded from
  hotspots). Exact LOC is computed for HEAD; for historical playheads, LOC
  is approximated by cumulative added − deleted from numstat.
- **hotspot(f) = recency_weighted_changes(f) × log2(1 + loc(f))**, where each
  change is weighted by `0.5 ** (age_days / 180)` (half-life 6 months)
  relative to the playhead. Normalized to 0–1 by the max in the repo.
- **coupling(a, b) = shared(a,b) / min(changes(a), changes(b))**, only for
  pairs with shared ≥ 5 and computed from commits touching ≤ 50 files
  (giant commits like formatting sweeps are ignored). Show pairs ≥ 0.5.
- **ownership:** per-file share of churn by author. Author identity is
  normalized via `.mailmap` (git does this with `%aN/%aE`).
- **bus-factor risk:** dominant author share ≥ 80%, file changes ≥ 10, and
  that author has no commits in the last 180 days of the range.
- **fossil age:** days from last touch to the playhead.

### 6a. Settled decisions (builder's calls, after design review)

1. **Lens liveness.** Per-file change events are stored as sorted numpy
   arrays (timestamps, churn, author ids). Hotspots, Ownership and Fossils
   recompute with vectorized `searchsorted` at up to 4 Hz during playback
   and on scrub-end (budget < 100 ms at 10k files, worker thread).
   Coupling uses a sparse co-change matrix built incrementally per commit,
   snapshotted every 500 commits (playhead = nearest snapshot + delta), and
   refreshes only on pause/scrub-end.
2. **`--since/--until`.** Full history is always parsed (and cached); the
   flags only window the timeline and metrics, so LOC stays correct.
3. **Which commits.** All commits reachable from `--branch` (default HEAD),
   ordered `git log --reverse --date-order` (parents before children).
   "Step" = next commit in that order. Lineage color: the first-parent chain
   is the mainline color; commits reachable only through a merge's second
   parent get a color per merge. Unmerged branches are out of scope.
4. **Normalization.** Hotspot score is normalized against the max score at
   the **end of the selected range**, clamped to 1, so early history is
   honestly cooler and risk visibly grows. Bus-factor "last 180 days" and
   fossil age are measured back from the **playhead**.
5. **Collapse.** Directory collapse is decided once from the union of all
   files ever existing in range, and stays fixed for the session.
6. **Performance target.** Base M1 on Retina. Bloom runs at half
   resolution and drops to quarter if frame time exceeds 20 ms.

## 7. Architecture

```
git_constellation/
  pyproject.toml
  README.md
  constellation/
    __main__.py        CLI (argparse) → app or report
    git_reader.py      subprocess plumbing → Commit/FileChange records
    cache.py           SQLite cache keyed by repo path + HEAD sha
    model.py           RepoHistory: commits, file timelines, dir tree,
                       state-at-time queries
    metrics.py         hotspot, coupling, ownership, fossils (pure, numpy)
    layout.py          hierarchical + force-directed 3D layout (numpy)
    insights.py        metric results → plain-language findings
    report.py          HTML report
    app.py             main loop, state, input dispatch
    camera.py          orbit camera, fly-to tweening
    timeline.py        playback clock, commit cursor
    render/
      renderer.py      moderngl context, passes, bloom
      shaders/*.glsl   planets (instanced sphere impostors), stars,
                       pulses (particles), lines, bloom, starfield
      ui.py            2D UI drawn with pygame onto a Surface, uploaded
                       as a texture overlay (re-upload only when dirty)
    lenses.py          lens → per-planet color/size/halo arrays
    tour.py, compare.py, recorder.py, watcher.py
  tests/
```

**Data flow:** `git_reader` streams
`git log --reverse --find-renames --numstat
--format=<sentinel>%H%x1f%P%x1f%aN%x1f%aE%x1f%at%x1f%s` plus
`git for-each-ref refs/tags`
→ `cache` → `model`. Metrics/layout are computed from the model off the
render thread; renderer only consumes numpy arrays.

**Rendering approach:** planets as instanced billboard sphere impostors
(cheap, scales to 10k+); pulses as a GPU particle buffer; bloom via
downsample/blur passes on an HDR framebuffer. Text labels via pygame fonts
rendered into the UI overlay, projected from 3D positions.

**Scale handling:** if > 10,000 files exist at the playhead, directories
beyond depth 3 with > 200 files collapse into a single aggregate planet
(expandable on double-click) *(builder's call)*. Layout is computed once for
the union of all files ever existing (so planets don't jump during replay),
with incremental relaxation for live-mode additions.

**Threading:** parsing and metric computation run in a worker thread;
results handed to the main loop via a queue. OpenGL calls only on the main
thread.

## 8. Build phases

1. **Foundations** — git_reader, cache, model, metrics (all with tests);
   `--report` HTML works end-to-end headless. Proves the data is right.
2. **Galaxy + History** — window, camera, layout, planets/stars, bloom,
   timeline playback with pulses, hover/click details.
3. **Hotspots** — lens, labels, insight panel. *First shippable version
   (History + Hotspots).*
4. **Remaining lenses** — Coupling, Ownership, Fossils; search; author
   filter.
5. **Story features** — guided tour, compare mode.
6. **Export & live** — screenshot, video recording, `--watch`.

## 9. Testing

- Unit tests build tiny throwaway git repos in `tmp_path` via `git` CLI
  (known commits, renames, deletes, merges, binary files, multiple
  authors) and assert parsed records and metric values exactly.
- Layout tests: determinism (seeded), no NaNs, siblings closer to their
  directory star than to unrelated stars.
- Rendering: a smoke test that creates a standalone (headless) moderngl
  context and renders one frame if a GL context is available; skipped
  otherwise.
- Manual acceptance: run against this machine's own repos and a large
  public repo clone (e.g. `cpython` shallow history) for performance.

## 10. Left to the builder

Exact visual styling (palette, glow intensity, easing curves), shader
details, UI typography, keyboard shortcuts beyond those listed, cache file
location (suggest `~/.cache/git_constellation/`), and internal module
boundaries where the above is only a sketch.
