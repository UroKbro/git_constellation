# Git-Constellation Phase 2 (Galaxy + History) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python -m constellation PATH` opens a window showing the repo as a 3D galaxy (directories = stars, files = planets) and replays its history: a timeline you can play, scrub and step, commit comets firing pulses at touched files, bloom, and hover/click details. `--report` and the text summary keep working.

**Architecture:** Everything that can be pure numpy is pure and unit-tested: `layout` (positions), `collapse` (aggregate planets), `timeline` (playback clock), `scene` (per-frame planet/star/pulse arrays and animations), `camera`, `picking`, `details` (panel text). `loader` runs parsing + layout + coupling in a worker thread and posts progress to a queue. `render/` owns all OpenGL (moderngl): one HDR framebuffer, instanced impostors, particles, bloom, and a pygame-drawn UI overlay texture. `app` is the main loop that glues them; GL runs only on the main thread.

**Tech Stack:** Python 3.14, numpy, pygame-ce 2.5.8+, moderngl 5.13+, pytest. Apple M4 Pro dev machine (driver reports "4.1 Metal").

**Director amendments (2026-10-18):** framing via `Layout.bounds`, comets as a pickable batch, panel authors by churn, `--screenshot` for visual verification, work-vs-frame bench timing, per-playhead scene caching — folded into Tasks 3, 4, 5, 7, 8, 9, 12 and Acceptance.

**Spec:** `docs/design.md`. §4 (UX: window layout, galaxy, interaction), §6a (settled decisions 3, 5, 6), §7 (architecture, scale handling, threading) and §8 item 2 bind this plan. Phase 1 plan: `docs/superpowers/plans/2026-10-17-phase1-foundations.md`.

## Global Constraints

- Branch `phase2-galaxy` off `main` (455454a). Implementers **do not commit**; the coordinator runs the full suite and commits each task.
- Runtime deps become `numpy>=2.2`, `pygame-ce>=2.5.8`, `moderngl>=5.13` (pyproject `dependencies`). No other new deps.
- OpenGL: request **3.3 core, forward-compatible** via `pygame.display.gl_set_attribute` (`GL_CONTEXT_MAJOR_VERSION=3`, `GL_CONTEXT_MINOR_VERSION=3`, `GL_CONTEXT_PROFILE_MASK=GL_CONTEXT_PROFILE_CORE`, `GL_CONTEXT_FLAGS=GL_CONTEXT_FORWARD_COMPATIBLE_FLAG`) **before** `set_mode(..., OPENGL | DOUBLEBUF | RESIZABLE)`. Shaders are `#version 330 core`. No GL features above 4.1.
- Retina: every viewport, FBO and pixel read uses the **drawable size** (`ctx.screen.size`), while mouse coordinates and `pygame.display.get_window_size()` are in window points — convert with `scale = drawable_w / window_w`.
- GL calls only on the main thread. Parsing, `RepoHistory.build`, layout and `CouplingIndex` run in the loader worker thread.
- Scene/timeline/camera/picking/layout/collapse/details modules must not import pygame or moderngl (pure numpy, headless-testable).
- All playhead arguments remain commit indexes into `RepoHistory.commits`.
- Playback: 1× = 4 commits/s; speeds are powers of two from 0.25× to 64× (`[0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]`), default 1×.
- Planet radius = `clip(0.15 * sqrt(loc), 0.25, 3.0)` world units (loc 0 → 0.25).
- Bloom at **half** resolution; auto-switch to **quarter** when the rolling mean frame time over the last 60 frames exceeds **20 ms**; never switch back during a session.
- Collapse (§7, §6a.5): decided once at load from the union of files ever alive in the window: if that union exceeds **10,000** files, every directory at depth > 3 whose subtree holds > 200 files becomes one aggregate planet. Double-click an aggregate expands it (members shown at their layout positions) for the rest of the session.
- Pulses: at most **20,000** live particles; when full, the oldest are dropped. Pulse life 1.2 s wall time; brightness `= 0.4 + 0.6 * min(1, log2(1 + added + deleted) / 10)`.
- Animations (wall time): add = flash-in 0.6 s (scale 0→1, emissive boost 3→1); delete = collapse-and-fade 0.8 s; rename = slide between placements 0.6 s, ease in-out cubic. Fly-to = 0.8 s ease in-out cubic.
- Copy (exact): progress `"Reading history… {n} commits"`, then `"Laying out {n} files…"`; empty repo `"No commits yet"` centred in the window; lens indicator `"1 History"`; help line in the panel footer `"Space play · ←/→ step · Home/End · I panel · F3 fps"`.
- Text summary stays available: new flag `--summary` prints the Phase 1 findings to stdout and exits without a window; the line `"Galaxy window arrives in Phase 2 — use --report for HTML."` is removed.

## Review Focus

1. **Window opened on a repo with 0 commits, or a `--since` that selects 0 commits** → the window still opens; empty repo shows "No commits yet", empty window shows the galaxy at the last commit with a paused, empty timeline. No exception, no division by zero in histogram/normalisation. Tests in Task 6 (timeline with 0 commits / empty window) and Task 11 (`--summary` / empty repo CLI).
2. **Window resized or moved between a Retina and a non-Retina display** → FBOs are rebuilt at the new drawable size, picking stays aligned with the cursor. Test in Task 9 (renderer `resize` rebuilds to the requested size) and Task 5 (point→pixel scale used by picking).
3. **Scrubbing backwards / jumping with Home/End** → no pulses or flash animations are spawned for a backwards move or a jump of more than 256 commits; planet set equals `files_alive_at(playhead)` immediately. Test in Task 7.
4. **Files moved by a rename across directories, deleted and re-added, or never alive inside the window** → each placement has a finite position; files never alive in the window are not drawn and not pickable. Tests in Tasks 3 and 7.
5. **Huge single commit (e.g. 5,000 files vendored in one go) during 64× playback** → particle cap holds at 20,000, frame loop keeps running. Test in Task 7.

## File map

| File | Responsibility | Task |
|------|----------------|------|
| `constellation/model.py` | phantom-file fold at merge reconciliation | 1 |
| `pyproject.toml`, `constellation/render/__init__.py`, `constellation/render/gl.py`, `tests/conftest.py` (`gl_ctx` fixture) | deps, context creation | 2 |
| `constellation/layout.py` | placements + positions | 3 |
| `constellation/camera.py`, `constellation/picking.py` | orbit camera, ray picking | 4 |
| `constellation/app.py`, `constellation/render/renderer.py`, `constellation/render/shaders/points.{vert,frag}`, `constellation/__main__.py` | vertical slice | 5 |
| `constellation/timeline.py` | playback clock | 6 |
| `constellation/palette.py`, `constellation/collapse.py`, `constellation/scene.py` | per-frame arrays, animations, pulses | 7 |
| `constellation/details.py` | panel/tooltip text | 8 |
| `constellation/render/renderer.py`, `constellation/render/shaders/*.glsl` | full renderer + bloom | 9 |
| `constellation/render/ui.py` | 2D overlay | 10 |
| `constellation/loader.py` | worker thread | 11 |
| `constellation/app.py`, `constellation/__main__.py` | integration, `--bench` | 12 |

## Execution waves

Tasks in the same wave touch disjoint files and run in parallel.

| Wave | Tasks | Notes |
|------|-------|-------|
| A | 1, 2, 3, 4 | 2 owns `tests/conftest.py` edits this wave |
| B | 5 | vertical slice — needs 2, 3, 4; manual check by coordinator: window shows Django dots, orbit works |
| C | 6, 7, 8, 11 | pure modules + loader |
| D | 9, 10 | renderer.py (9) and ui.py (10) are disjoint |
| E | 12 | integration |
| F | acceptance (coordinator) | Django 60 fps, `--bench` numbers, 0/0 invariant, full suite |

---

### Task 1: Fold phantom files at merge reconciliation

A side-branch commit that edits a path the mainline already renamed or deleted (date order replays the side edit later) currently creates a *phantom* file — a new id born from an `M`/`T`/`D` event on a path that is not alive — which lives until the merge reconciles it away (ghost planet). Fold it.

**Files:** Modify `constellation/model.py` (`RepoHistory.build`, `reconcile`). Test: `tests/test_merge_reconcile.py`.

**Interfaces:**
- Consumes: Phase 1 `RepoHistory.build`, the `zombie_after_dir_rename`, `modify_delete_kept` repo scenarios in `tests/test_merge_reconcile.py`.
- Produces: no API change. Phantoms are tracked inside `build`.

**Rules:**
- A builder is a *phantom* when it was created by an `M`, `T` or `D` change (not `A`/`R`) on a path not in `alive`.
- In `reconcile`, when a phantom dies because its path is absent from the merge tree:
  - if the merge changes contain `R old_path=<phantom path> -> path=P` and `P` is alive, append the phantom's events to `alive[P]`'s file;
  - otherwise, if an earlier (dead) builder had the same path, append them to the most recent such builder;
  - then **erase** the phantom: it never appears in `h.files`. File ids are compacted after the replay (remap `alive`/`_alive_paths`).
- Phantoms that survive the merge (path present in the merge tree) are left as they are.
- After folding, each file's event arrays stay sorted by commit index.

- [ ] **Step 1: Write the failing tests** in `tests/test_merge_reconcile.py`

```python
def test_phantom_after_rename_folds_into_target(repo):
    zombie_after_dir_rename(repo)
    h = _build(repo)
    assert [f.path for f in h.files] == ["new/x"]
    edit = next(i for i, cm in enumerate(h.commits) if cm.subject == "edit")
    assert edit in h.files[0].event_commit.tolist()
    assert list(h.files[0].event_commit) == sorted(h.files[0].event_commit)
    for i in range(len(h.commits)):
        assert h.files_alive_at(i).sum() == 1   # never a ghost old/x


def test_phantom_after_delete_folds_into_dead_original(repo):
    repo.commit("base", {"g.py": "1\n", "k.py": "k\n"}, day=0)
    repo.checkout("side", create=True)
    repo.commit("edit", {"g.py": "1\n2\n"}, day=3)
    repo.checkout("main")
    repo.commit("del", {"g.py": None}, day=2)
    repo.merge("side", "merge", day=4, resolve={"g.py": None})
    h = _build(repo)
    assert sorted(f.path for f in h.files) == ["g.py", "k.py"]
    g = h.files[[f.path for f in h.files].index("g.py")]
    edit = next(i for i, cm in enumerate(h.commits) if cm.subject == "edit")
    assert edit in g.event_commit.tolist()
    assert _alive(h) == {"k.py"}


def test_surviving_phantom_kept(repo):
    modify_delete_kept(repo)          # existing scenario: merge keeps y.py
    assert "y.py" in _alive(_build(repo))
```

- [ ] **Step 2:** `pytest tests/test_merge_reconcile.py -v` → the two new fold tests FAIL.
- [ ] **Step 3:** Implement the rules above in `RepoHistory.build`.
- [ ] **Step 4:** `pytest -q` → all green (98 + 3).

---

### Task 2: Dependencies and GL context helpers

**Files:** Modify `pyproject.toml`. Create `constellation/render/__init__.py` (empty), `constellation/render/gl.py`. Modify `tests/conftest.py`. Test: `tests/test_render_gl.py`.

**Interfaces — Produces:**
- `configure_gl_attributes() -> None` — sets the four pygame GL attributes in Global Constraints. Must be called after `pygame.init()` and before `set_mode`.
- `open_window(size: tuple[int, int], title: str) -> tuple[pygame.Surface, moderngl.Context]` — `configure_gl_attributes()`, `set_mode(size, OPENGL|DOUBLEBUF|RESIZABLE)`, `moderngl.create_context()`, enables blending. Title `"Git-Constellation — {repo}"` is the caller's job.
- `standalone_context() -> moderngl.Context | None` — `moderngl.create_standalone_context(require=330)`, `None` on any exception.
- `drawable_scale(ctx, window_size) -> float` — `ctx.screen.size[0] / window_size[0]` (1.0 if width is 0).
- conftest fixture `gl_ctx` — yields `standalone_context()` or calls `pytest.skip("no OpenGL context")`.

- [ ] **Step 1:** Add `pygame-ce>=2.5.8`, `moderngl>=5.13` to `dependencies`; `.venv/bin/pip install -e '.[dev]'`.
- [ ] **Step 2: Tests**

```python
def test_standalone_context_is_33_core(gl_ctx):
    assert gl_ctx.version_code >= 330

def test_render_to_f16_fbo(gl_ctx):
    tex = gl_ctx.texture((8, 8), 4, dtype="f2")
    fbo = gl_ctx.framebuffer([tex]); fbo.use(); fbo.clear(1.0, 0.5, 0.25, 1.0)
    px = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype=np.float16)
    assert np.allclose(px[:4], [1.0, 0.5, 0.25, 1.0])

def test_configure_gl_attributes_sets_core_profile(monkeypatch):
    calls = {}
    monkeypatch.setattr(pygame.display, "gl_set_attribute", lambda k, v: calls.__setitem__(k, v))
    configure_gl_attributes()
    assert calls[pygame.GL_CONTEXT_MAJOR_VERSION] == 3
    assert calls[pygame.GL_CONTEXT_MINOR_VERSION] == 3
    assert calls[pygame.GL_CONTEXT_PROFILE_MASK] == pygame.GL_CONTEXT_PROFILE_CORE
    assert calls[pygame.GL_CONTEXT_FLAGS] == pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG
```

- [ ] **Step 3:** Implement `gl.py`. **Step 4:** `pytest -q` green (the GL tests skip cleanly when `standalone_context()` returns None).

---

### Task 3: Layout

**Files:** Create `constellation/layout.py`. Test: `tests/test_layout.py`.

**Interfaces:**
- Consumes: `RepoHistory` (`files[i].paths`, `dirs`, `dir_parent`), optional `list[CouplingPair]`.
- Produces:
  - `@dataclass Layout`: `dir_pos: (n_dirs,3) float32`, `dir_radius: (n_dirs,) float32` (orbit radius of that dir's files), `place_file: (n_places,) int32`, `place_dir: (n_places,) int32`, `place_pos: (n_places,3) float32`, `file_place_start: (n_files,) int32` (first placement index of file i; placements of file i are contiguous, one per entry of `files[i].paths`, in that order).
  - `Layout.place_at(file_id: int, commit: int) -> int` — placement index active at commit (last `paths` entry with start ≤ commit; first one if commit is before birth).
  - `compute_layout(h: RepoHistory, pairs: Sequence[CouplingPair] = (), seed: int = 0, iterations: int = 60) -> Layout`.
  - `Layout.bounds(mask: np.ndarray | None = None) -> tuple[np.ndarray(3), float]` — centre and radius of the bounding sphere (centre = midpoint of the AABB, radius = max distance to it) over `place_pos[mask]` plus `dir_pos`; with an empty selection returns `(zeros(3), 10.0)`.

**Algorithm (decided):**
1. Directory stars: root at origin. Children of a dir are placed on a sphere around the parent at distance `dist(child) = 8 + 2.5 * sqrt(subtree_files(child))` (subtree_files = placements in the child's subtree), directions from a Fibonacci sphere rotated by a seeded random rotation per parent, larger subtrees first.
2. Placements: each `(file, path)` sits on a Fibonacci sphere of radius `dir_radius[d] = 2 + 0.35 * sqrt(n_places_in_d)` around its directory star, jittered by `0.15 * dir_radius` (seeded).
3. Relaxation (`iterations` steps, numpy, step 0.05): spring to the home position (k=1.0) + attraction between the *final* placements of each coupled pair, weight `strength` (k=0.3), capped so a placement never leaves `1.5 * dir_radius` of its own star.
- Seeded `np.random.default_rng(seed)`; no dependence on dict/set iteration order other than insertion order.

- [ ] **Step 1: Tests** (build histories with `tests/history_helpers.py`)
  - `test_deterministic`: two calls with the same seed → `np.array_equal` on every array; different seed → `place_pos` differs.
  - `test_finite`: no NaN/inf in any array for a history with renames across dirs, a delete + re-add, and an empty-dir-only root.
  - `test_siblings_nearer_own_star`: for every placement, distance to own `dir_pos` < distance to every other dir star at the same depth.
  - `test_rename_has_two_placements`: file renamed `a/x → b/x` has 2 contiguous placements, `place_dir` = [a, b]; `place_at` returns the first before the rename commit and the second from it on.
  - `test_coupling_pulls_pair_closer`: with a pair of strength 1.0 between files in different dirs, their final placements are closer than with `pairs=()`, and each still within `1.5 * dir_radius` of its star.
  - `test_bounds_contains_everything`: every placement and star lies within `radius` (+1e-4) of `center`; empty mask → `(0, 10.0)`.
  - `@pytest.mark.slow test_layout_10k_under_2s`: synthetic 10,000 files in 400 dirs, 3 levels → `compute_layout` < 2.0 s.
- [ ] **Step 2–4:** fail → implement → `pytest tests/test_layout.py -v` green.

---

### Task 4: Camera and picking

**Files:** Create `constellation/camera.py`, `constellation/picking.py`. Tests: `tests/test_camera.py`, `tests/test_picking.py`.

**Interfaces — Produces:**
- `class OrbitCamera(target=(0,0,0), distance=60.0, yaw=0.6, pitch=0.35, fov_deg=50.0)`:
  - `orbit(dx_px, dy_px)` (0.005 rad/px; pitch clamped to ±1.5), `pan(dx_px, dy_px, viewport_h)` (moves target in the view plane, scaled by distance), `zoom(wheel_steps)` (distance × 0.9^steps, clamped [2, 5000]).
  - `frame(center, radius)` — no animation: target = center, distance = `radius / tan(fov/2) * 1.1` (initial framing used by the app).
  - `fly_to(target, distance, now)` and `update(now)` — eased (cubic in-out, 0.8 s) interpolation of target and distance; `flying -> bool`.
  - `eye() -> np.ndarray(3)`, `view() -> (4,4) float32`, `proj(aspect) -> (4,4) float32` (near 0.1, far 20000), matrices **column-major-ready** (`.T.copy()` is what the renderer uploads — document in docstring).
  - `ray(px, py, viewport_w, viewport_h) -> (origin(3), direction(3) unit)`; pixel origin top-left.
- `pick(origin, direction, centers (n,3), radii (n,), mask (n,) bool) -> int | None` — nearest positive hit among masked spheres; vectorised.

- [ ] **Tests:** after `frame(c, r)` every point sampled on the sphere (c, r) projects inside NDC [-1, 1]² at aspect 1.0 and 1.6; ray through the viewport centre equals the camera forward direction; a point projected with `proj @ view` to NDC and back via `ray` hits it (round trip to 1e-4); `zoom(+10)` then `zoom(-10)` restores distance; pitch clamp; `fly_to` reaches target exactly at `now+0.8` and is halfway (±1e-6, cubic midpoint 0.5) at `now+0.4`; `pick` returns the nearer of two spheres on the ray, ignores masked-out and behind-origin spheres, returns `None` on miss; `pick` over 100k spheres < 5 ms (slow).

---

### Task 5: Vertical slice — window with real files as dots

**Files:** Create `constellation/app.py`, `constellation/render/renderer.py`, `constellation/render/shaders/points.vert`, `points.frag`. Modify `constellation/__main__.py`. Test: `tests/test_renderer.py`, `tests/test_cli.py`.

**Interfaces:**
- Consumes: Task 2 `open_window`, `standalone_context`; Task 3 `compute_layout`; Task 4 `OrbitCamera`.
- Produces:
  - `class Renderer(ctx: moderngl.Context, size: tuple[int,int])` with `resize(size)`, `set_points(pos (n,3) f4, color (n,4) f4, size_px (n,) f4)`, `render(view, proj) -> None` (draws into `ctx.screen` here; Task 9 replaces internals), `read_pixels() -> np.ndarray (h,w,4) uint8` (tests).
  - `app.run(h: RepoHistory, layout: Layout, title: str, w: Window) -> int` — loop at vsync/60 fps cap: left-drag orbit, right-drag pan, wheel zoom, Esc/close quits; returns 0.
  - Hidden flag `--screenshot PATH` with `--at FRACTION` (both `help=argparse.SUPPRESS`; `--at` default 1.0): load, frame the galaxy, (Task 12: play from `window.start` to the fraction at 64× and) render 30 frames so pulses and bloom settle, read the drawable framebuffer (`ctx.screen.read(components=3)`, flipped) and save it as PNG via `pygame.image.save`, exit 0. The window may be created with `pygame.HIDDEN`.
  - `__main__`: no `--report`/`--summary` → loads synchronously (Task 12 swaps in the loader), `compute_layout(h)`, `app.run(...)`. `--summary` → Phase 1 text output (Global Constraints). Empty repo: print `"No commits yet."` and, in window mode, `app.run_empty(title)` shows the centred message (Task 10 provides text rendering; here a blank window is acceptable).
- Window mode is imported lazily (`from constellation import app` inside the branch) so `--report`/`--summary` never import pygame.

- [ ] **Tests:** `tests/test_packaging.py` runs the installed script with `--summary` (it must never open a window; the wheel install now pulls pygame-ce and moderngl). Renderer smoke (uses `gl_ctx`, renders into an offscreen FBO of 64×64 standing in for `ctx.screen` — add `Renderer(ctx, size, target=None)` where `target=None` means `ctx.screen`): one point at the origin, camera looking at it → centre pixel non-black, corner pixel black. CLI: existing tests that ran without `--report` add `--summary` (`test_plain_text_findings` asserts the "Galaxy window…" line is **absent**); `test_cli_window_mode_calls_app` monkeypatches `constellation.app.run` to record its args and asserts it got a `RepoHistory` with the repo's files and returned 0.
  - App starts with `camera.frame(*layout.bounds(alive_at_window_end))`.
  - `@pytest.mark.slow test_screenshot_flag` — on a 3-commit repo, `--screenshot out.png` exits 0 and `pygame.image.load(out.png)` has size == the drawable size (non-empty, not all black); skip when no display/GL.
- [ ] **Coordinator check:** `--screenshot` of the Django clone and of this repo into the scratchpad (paths sent to the director): ~9.5k dots clustered by directory, whole galaxy in frame; then orbit/pan/zoom by hand on Retina.

---

### Task 6: Timeline (playback clock)

**Files:** Create `constellation/timeline.py`. Test: `tests/test_timeline.py`.

**Interfaces — Produces:** `class Timeline(window: Window, commit_ts: np.ndarray)`:
- state: `cursor: float` (commit index, starts at `window.end`, paused), `playing: bool`, `speed: float`.
- `toggle()`, `faster()`, `slower()` (move along the speed list, clamped), `step(n: int)` (pauses, moves cursor by n commits within the window), `home()`, `end()`, `scrub(fraction: float)` (0..1 → window index, by **time** using `commit_ts` running max), `update(dt: float) -> tuple[int, int]` returning `(prev_index, new_index)` of the integer playhead; pressing play at the window end restarts from `window.start`; reaching the end pauses.
- `playhead -> int`; `histogram(bins: int) -> np.ndarray[int]` commit counts per equal-time bin over the window; `fraction_of(index) -> float`.
- Empty window: `playhead == window.end` clamped to `len-1` (≥ 0), `update` returns `(p, p)`, histogram is zeros, `toggle` does nothing.

- [ ] **Tests:** 1× for 1.0 s advances 4 commits; 64× clamps at window end and pauses; `faster` from 64 stays 64, `slower` from 0.25 stays 0.25; `step(-1)` at start stays; `scrub(0.5)` on commits at days 0,1,2,10 lands on the last commit with ts ≤ day 5; histogram sums to window size; empty window (Review Focus 1).

---

### Task 7: Palette, collapse and scene state

**Files:** Create `constellation/palette.py`, `constellation/collapse.py`, `constellation/scene.py`. Tests: `tests/test_palette.py`, `tests/test_collapse.py`, `tests/test_scene.py`.

**Interfaces — Produces:**
- `palette.language_color(path: str) -> tuple[float,float,float]` — by extension family: Python `#3572A5`, JS/TS `#F1E05A`, web (html/css/scss/vue/svelte) `#E34C26`, C/C++/ObjC `#6E7FD2`, Go `#00ADD8`, Rust `#DEA584`, Java/Kotlin/Scala `#B07219`, Ruby `#CC342D`, shell `#89E051`, docs (md/rst/txt) `#C9D1D9`, data/config (json/yaml/toml/ini/xml/csv) `#8B949E`, images/binary `#6E5494`, other `#9DA5B4`. `palette.lineage_color(k: int)` — 0 = `#7FD4FF`, k ≥ 1 cycle through 8 fixed hues (`#FF9E64 #C3E88D #F07178 #C792EA #FFCB6B #89DDFF #F78C6C #82AAFF`).
- `collapse.CollapsePlan(member_to_group: (n_files,) int32 (-1 = not collapsed), group_dir: list[int], expanded: set[int])`; `collapse.plan(h, w) -> CollapsePlan` per Global Constraints; `CollapsePlan.expand(group)`.
- `scene.PlanetBatch(pos, radius, color (n,4: rgb + emissive), ids (n,) int32 — file id, or -(group+1) for aggregates)`, `scene.StarBatch(pos, radius, color)`, `scene.PulseBatch(start, end, color, born, life)` (all float32 numpy).
- `class Scene(h, layout, plan: CollapsePlan, window: Window)`:
  - `advance(prev: int, cur: int, now: float)` — when `0 < cur - prev <= 256`: for each commit in `(prev, cur]` spawn a comet at the touched files' star centroid offset by 4 units outward, and one pulse per touched file (merge commits: `merge_changes` are not pulsed; their comet alternates `lineage[i]` and the second parent's lineage color = braid); register add/delete/rename animations. Otherwise (backwards, jump) clear pulses and animations.
  - `planets(playhead: int, now: float) -> PlanetBatch` — alive files at playhead (aggregates replace collapsed members unless expanded; aggregate radius from summed loc, colour = most common language), radius per Global Constraints, animation-adjusted, plus deleting files still fading.
  - `stars(playhead) -> StarBatch` — dirs with ≥ 1 alive file in their subtree; radius `0.6 + 0.12 * sqrt(alive_in_subtree)`.
  - `pulses(now) -> PulseBatch` — live particles (cap 20,000, oldest dropped).
  - `scene.CometBatch(pos, color (n,4), color2 (n,4) — second braid colour, equals `color` for non-merges, commit (n,) int32, born, life)`; `comets(now) -> CometBatch` — one comet per commit spawned by `advance`, life 1.5 s.
  - Per-frame budget: `planets/stars` cache the playhead-dependent arrays (`files_alive_at`, `loc_at`, aggregates, star subtree counts) keyed by the integer playhead and only recompute when it changes; animations are applied as an overlay on the cached arrays.
- [ ] **Tests:** colours for `a.py`, `b.tsx`, `README.md`, `Makefile`; collapse: 10,001 files with `a/b/c/d/` holding 300 files → that dir collapses, `a/b/` (depth 2) does not, ≤ 10,000 files → nothing collapses; scene: `planets(i).ids` equals `flatnonzero(files_alive_at(i))` when nothing animates (`now` far after), radius for loc 0/100/10⁶ = 0.25/1.5/3.0; `advance(0, 3)` spawns pulses equal to the touched-file count of commits 1..3; `advance(5, 2)` and `advance(0, 1000)` spawn none and clear existing; a commit touching 25,000 files leaves exactly 20,000 pulses; deleted file appears with fading alpha at `now+0.4` and is gone at `now+0.9`; renamed file's position at `now+0.3` is between its two placements; files never alive in the window never appear; `advance(0, 3)` yields 3 comets whose `commit` is `[1, 2, 3]`, a merge comet has `color2 != color`; `@pytest.mark.slow test_scene_frame_budget` — synthetic 10k files / 50k commits, 64× advance (64 commits per call), `planets+stars+pulses+comets` mean of 100 calls < 4 ms.

---

### Task 8: Details text

**Files:** Create `constellation/details.py`. Test: `tests/test_details.py`.

**Interfaces — Produces:**
- `file_details(h, w, playhead, file_id, coupling: CouplingIndex | None) -> list[str]` — lines, exact format:
  `path`; `"{loc} lines · {n} changes"`; `"Authors: Ana 70%, Bo 20%, Cy 10%"` (top 3 by **churn share** from `metrics.ownership(h, w.start, playhead).top3`, rounded — the same numbers the Phase 1 findings/report print for that file); `"First touched {YYYY-MM-DD} · last touched {YYYY-MM-DD}"` (local time); `"Hotspot {score:.2f}"`; then `"Coupled with:"` + up to 5 `"  {path} ({pct}%)"` from `coupling.pairs(w.start, playhead, min_shared=2, min_strength=0.0)` sorted by strength — the line `"Coupled with: none"` when empty.
- `commit_details(h, index) -> list[str]` — `"{sha[:10]} · {author} · {YYYY-MM-DD HH:MM}"`, the subject, `"{n} files · +{added} −{deleted}"` (merge: `"merge of {k} parents"` instead of the diffstat).
- `tooltip(h, playhead, file_id) -> str` — `"{path} — {loc} lines"`; aggregate: `"{dir}/ — {n} files (double-click to expand)"` via `aggregate_tooltip(h, plan, group, playhead)`.
- [ ] **Tests:** exact strings on a small RepoBuilder repo with three authors and a coupled pair; merge commit line; pluralisation reuses `insights._plural` (`1 change`, `1 file`).

---

### Task 9: Full renderer (impostors, stars, starfield, pulses, bloom)

**Files:** Modify `constellation/render/renderer.py`. Create `constellation/render/shaders/{planet,star,starfield,pulse,bloom_down,blur,composite,overlay}.{vert,frag}` (vertex shaders may be shared). Test: `tests/test_renderer.py`.

**Interfaces:**
- Consumes: Task 7 batches.
- Produces (replaces Task 5's `set_points`):
  - `Renderer(ctx, size, target=None)`, `resize(size)` rebuilds HDR (`f2`, RGBA) scene FBO + depth, bloom chain at `bloom_scale` (0.5 or 0.25).
  - `bloom_scale: float` property; `set_bloom_scale(s)`.
  - `draw(view, proj, planets: PlanetBatch, stars: StarBatch, pulses: PulseBatch, now: float, overlay: tuple[bytes, (w,h)] | None)` — passes in order: starfield (2,000 seeded points, fixed), stars (additive billboards), planets (instanced camera-facing quads, ray-sphere impostor in the fragment shader with depth write, lambert from the nearest star + emissive), pulses (additive points, fade by age), comets (`comets: CometBatch` argument after `pulses`: bright head + short tail along the direction away from the galaxy centre, tail alternating `color`/`color2` for merges = braid), bloom (threshold 1.0 → downsample → 2× separable 9-tap gaussian → composite with tonemap `1 - exp(-x)`), overlay texture alpha-blended last. Instance buffers are orphaned and rewritten only when the batch object changes (`id()`), else reused.
- [ ] **Tests (gl_ctx, offscreen 128×128 target):** one planet at origin → centre pixel brighter than a pixel 40 px away and corners black; with bloom a bright planet lights pixels just outside its disc (halo > 0) while `planets` of emissive 0 does not; `resize((64,48))` → `read_pixels().shape == (48, 64, 4)`; `set_bloom_scale(0.25)` rebuilds chain at 32×32 for 128×128; empty batches render without error.

---

### Task 10: UI overlay

**Files:** Create `constellation/render/ui.py`. Test: `tests/test_ui.py` (sets `SDL_VIDEODRIVER=dummy`, `pygame.font.init()`).

**Interfaces — Produces:** `class Overlay(size: tuple[int,int], scale: float)` drawing on a `pygame.Surface(SRCALPHA)` at drawable resolution, `dirty` flag, `to_bytes() -> tuple[bytes, (w,h)]` (RGBA, flipped for GL; clears `dirty`).
- `draw_timeline(hist, playhead_frac, playing, speed, tag_fracs, lineage_colors_per_bin)` — bar at the bottom, 64 points tall; histogram bars, playhead line, ▶/❚❚, `"{speed}×"`, tag ticks.
- `timeline_hit(x_pt, y_pt) -> float | None` — fraction for clicks inside the bar's track (points), else None.
- `draw_panel(lines: list[str], visible: bool)` — right side, 360 pt wide, footer help line (Global Constraints copy).
- `draw_tooltip(text, x_pt, y_pt)`, `draw_lens("1 History")`, `draw_progress(text, fraction | None)` (centred bar), `draw_message("No commits yet")`, `draw_fps(fps: float | None)`.
- Fonts: `pygame.font.SysFont("Menlo,Helvetica", ...)` sized ×`scale`.
- [ ] **Tests:** `timeline_hit` at the bar's left/right track edges → 0.0/1.0, above the bar → None; `to_bytes` length = w·h·4 and clears `dirty`; drawing the panel marks dirty; a long path in a tooltip is truncated with `…` to fit 600 pt.

---

### Task 11: Loader worker

**Files:** Create `constellation/loader.py`. Test: `tests/test_loader.py`.

**Interfaces — Produces:**
- `@dataclass Loaded(root: Path, h: RepoHistory, window: Window, layout: Layout, coupling: CouplingIndex, plan: CollapsePlan)`.
- `start_loading(root: Path, ref: str, since: int|None, until: int|None, use_cache: bool) -> queue.Queue` — daemon thread; posts `("progress", text, fraction|None)` (the two Global Constraints strings; fraction None while counting commits), then `("done", Loaded)` or `("error", message)` with the same messages `__main__` prints for OSError/CalledProcessError.
- Layout uses `coupling.pairs(w.start, w.end)` as attraction pairs.
- [ ] **Tests:** on a 3-commit RepoBuilder repo the queue yields ≥ 1 progress then `done` with `len(h.commits) == 3` within 10 s; a path inside a non-git dir yields `("error", ...)`; `Loaded.layout.place_pos` finite.

---

### Task 12: App integration and `--bench`

**Files:** Modify `constellation/app.py`, `constellation/__main__.py`. Test: `tests/test_cli.py`, `tests/test_app_logic.py`.

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `app.run(root, ref, since, until, use_cache, title, bench: float | None = None) -> int` — opens the window immediately, shows progress from `start_loading`, then the galaxy at `window.end`, paused. Errors from the loader → printed to stderr, exit 1.
  - Input (design §4 Interaction, Phase 2 subset): left-drag orbit, right-drag pan, wheel zoom; hover → tooltip (picking against `Scene.planets` using drawable scale); click planet → `file_details` in panel; click comet → `commit_details` (picking tests live comets first, sphere radius 1.5, then planets); double-click planet/star → `fly_to`; double-click aggregate → `plan.expand`; `Space`, `←/→`, `Home/End`, `+`/`-` speed, `I` panel, `F3` fps, Esc quit; click/drag on timeline → `scrub`. Keys from later phases (`1`–`5` other than `1`, `/`, `A`, `T`, `C`, `F12`) are ignored.
  - Resize → `Renderer.resize(ctx.screen.size)`, overlay rebuilt.
  - Bloom auto-switch per Global Constraints.
  - Hidden flag `--bench SECONDS` (argparse `help=argparse.SUPPRESS`): after load, play at 64× from `window.start` for SECONDS at half bloom, then the same at quarter bloom; measure **work** time (frame start → `ctx.finish()` before `flip`) and **frame** wall time, print `"bench half: work mean {ms:.1f} p95 {ms:.1f} ms · frame mean {ms:.1f} p95 {ms:.1f} ms"` and the same for `quarter`; exit 0. Request vsync off (`pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)`) but do not rely on it — macOS may force vsync, which is why work time is reported.
  - `--screenshot`/`--at` now play from `window.start` to the fraction at 64× before the 30 settle frames.
  - Pure helpers in `app.py` for testability: `InputState` mapping pygame-free events (`("key", name)`, `("click", x, y)`, `("dblclick", x, y)`) to actions — tested without a window.
- [ ] **Tests:** a comet and a planet on the same ray → the comet is picked and resolves to its commit index; `test_app_logic.py` drives `InputState` → `Timeline`/`OrbitCamera` actions (Space toggles, → steps and pauses, `+` ×2 speed, click inside timeline scrubs, `I` toggles panel); CLI window-mode test from Task 5 updated to the new `run` signature.

---

### Acceptance (coordinator, Wave F)

- `pytest -q` (incl. slow) green; Phase 1 tests unchanged except the `--summary` additions.
- Django clone: model invariant still 0 extra / 0 missing vs `git ls-tree -r HEAD`; window opens with progress, galaxy legible.
- `python -m constellation <django> --bench 20` → **work** mean ≤ 16.7 ms at half bloom on Retina; record half and quarter work/frame numbers in the ledger and report.
- Screenshots (scratchpad, paths sent to the director): Django overview, Django `--at 0.6` with pulses visible, this repo, and one after expanding an aggregate. Report Django's aggregate count (collapse likely triggers: ever-alive union > 10k).
- Manual: scrub start→end, step, Home/End, hover/click a planet and cross-check its panel line counts and authors against `git log --follow --numstat -- <file>`; resize across displays.
