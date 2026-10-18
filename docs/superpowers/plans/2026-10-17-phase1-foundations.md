# Git-Constellation Phase 1 (Foundations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless, fully tested data layer — read any local repo's history, cache it, model it, compute every lens metric, and emit plain-language findings as a terminal summary and an HTML report.

**Architecture:** `git_reader` streams one `git log --raw --numstat -z` call into frozen dataclasses; `cache` persists them in SQLite keyed by repo + ref; `model.RepoHistory` turns commits into per-file numpy event arrays with stable file ids across renames; `metrics/*` are pure functions over `RepoHistory`; `insights` turns metric results into findings; `report` and `__main__` present them.

**Tech Stack:** Python 3.14, numpy, sqlite3 (stdlib), git CLI via subprocess, pytest.

**Spec:** `docs/design.md` (§6 metrics, §6a settled decisions, §7 architecture are binding for this plan).

## Global Constraints

- Project root: `~/Python_Projects/git_constellation/`; package `constellation/`; tests `tests/`.
- Python `>=3.14`; runtime deps for this phase: `numpy` only. Dev: `pytest`. (`pygame-ce`, `moderngl` are added in Phase 2.)
- Strictly read-only toward the analysed repo. Every git call is `git -C <root> -c core.quotepath=off ...`.
- Decode git output as UTF-8 with `errors="replace"`.
- Merge commits contribute no file changes (git log omits merge diffs by default — keep it that way; do not pass `-m`/`--cc`).
- Hotspot half-life 180 days; coupling: ignore commits touching > 50 files, require shared ≥ 5, report strength ≥ 0.5; bus-factor: dominant share ≥ 0.8, changes ≥ 10, dominant author silent for the last 180 days before the playhead; fossils: untouched > 365 days.
- Hotspot normalization: divide by max raw score at the **end of the window**, clamp to 1.
- All playhead/window arguments are **commit indexes** into `RepoHistory.commits` (inclusive).
- Cache dir default `~/.cache/git_constellation/`; overridable by the `GIT_CONSTELLATION_CACHE` env var (tests set it to `tmp_path`).

## Review Focus

1. Paths with spaces, unicode, or quotes → parsed verbatim (`-z` + `core.quotepath=off`). Test in Task 2.
2. A file deleted and later re-created at the same path → a **new** file id; history of the old one is not merged in. Test in Task 3.
3. Running from a subdirectory of the repo → analyses the whole repo, paths relative to the top. Test in Task 2.
4. Shallow clone (parents missing from history) → no crash; lineage treats missing parents as absent. Test in Task 4.
5. Two unrelated histories merged (`--allow-unrelated-histories`, two root commits) → both roots parsed, lineage assigns the second root's chain to the merge's lineage. Test in Task 4.

## Execution waves (for parallel dispatch)

Tasks in the same wave touch disjoint files and may run in parallel. Implementers **do not commit**; the coordinator runs the full suite and commits each task after the wave.

| Wave | Tasks |
|------|-------|
| A | 1 |
| B | 2, 3 |
| C | 4, 5, 6, 7, 8 |
| D | 9 |
| E | 10 |

---

### Task 1: Scaffold, shared types, test repo builder

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`, `constellation/__init__.py`, `constellation/types.py`, `tests/conftest.py`, `tests/repo_builder.py`, `tests/history_helpers.py`, `tests/test_repo_builder.py`

**Interfaces:**
- Produces (`constellation/types.py`, all `@dataclass(frozen=True, slots=True)`):
  - `FileChange(status: str, path: str, old_path: str | None, added: int, deleted: int, binary: bool)` — `status` is git's letter: `"A" | "M" | "D" | "R" | "T"`; `old_path` set only for `"R"`; binary → `added == deleted == 0`, `binary=True`.
  - `Commit(sha: str, parents: tuple[str, ...], author: str, email: str, ts: int, subject: str, changes: tuple[FileChange, ...])`
  - `Tag(name: str, sha: str)` — `sha` is the peeled commit sha.
- Produces (`tests/repo_builder.py`): `T0 = 1_700_000_000`, `DAY = 86_400`, and
  `class RepoBuilder(path: Path)` — inits a repo on branch `main` with isolated config (env `GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM=1`, `commit.gpgsign=false`). Methods:
  - `commit(msg: str, files: dict[str, str | bytes | None] = {}, author: str = "Ana", day: float = 0) -> str` — writes each file (`None` deletes it), `git add -A`, commits with author/committer name `author`, email `f"{author.lower()}@example.com"`, both dates `T0 + day*DAY`; `--allow-empty`. Returns sha.
  - `mv(old: str, new: str) -> None` (`git mv`, staged for next commit).
  - `checkout(branch: str, create: bool = False) -> None`; `orphan(branch: str) -> None` (`checkout --orphan` + `rm -rf` the index/worktree).
  - `merge(branch: str, msg: str, day: float, author: str = "Ana", unrelated: bool = False) -> str` (`--no-ff`).
  - `tag(name: str, annotated: bool = False) -> None`.
  - `path: Path` attribute.
- Produces (`tests/history_helpers.py`, for tests that build histories without git):
  `c(i: int, changes: list[FileChange], author: str = "Ana", parents: tuple[str, ...] | None = None, day: float | None = None) -> Commit` — `sha=f"s{i}"`, `email=f"{author.lower()}@example.com"`, `ts = T0 + int((i if day is None else day) * DAY)`, `subject=f"c{i}"`, parents default `(f"s{i-1}",)` or `()` when `i == 0`;
  `A(path, added, binary=False)`, `M(path, added, deleted)`, `D(path, deleted, binary=False)`, `R(old, new, added, deleted)` → `FileChange` with statuses `"A"/"M"/"D"/"R"` (binary forces added/deleted to 0).
- Produces (`tests/conftest.py`): fixture `repo(tmp_path) -> RepoBuilder` at `tmp_path / "repo"`; autouse fixture setting `GIT_CONSTELLATION_CACHE` to `tmp_path / "cache"`.

- [ ] **Step 1:** `git init` the project folder (`~/Python_Projects/git_constellation`), write `pyproject.toml` (setuptools, name `git-constellation`, `requires-python = ">=3.14"`, deps `["numpy>=2.2"]`, optional `dev = ["pytest>=8"]`, script `git-constellation = "constellation.__main__:main"`, `[tool.pytest.ini_options] testpaths = ["tests"]`), `.gitignore` (`__pycache__/`, `.venv/`, `*.egg-info/`, `.pytest_cache/`), a short `README.md`. Create `.venv` and `pip install -e '.[dev]'`.
- [ ] **Step 2: Write the failing test** `tests/test_repo_builder.py`:

```python
def test_builder_makes_commits_with_fixed_dates(repo):
    sha = repo.commit("first", {"a.py": "x\n"}, author="Bo", day=2)
    out = subprocess.run(["git", "-C", repo.path, "log", "-1", "--format=%H %an %ae %at"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == [sha, "Bo", "bo@example.com", str(T0 + 2 * DAY)]

def test_builder_delete_and_merge(repo):
    repo.commit("a", {"a.py": "1\n"})
    repo.checkout("side", create=True)
    repo.commit("b", {"b.py": "2\n"}, day=1)
    repo.checkout("main")
    m = repo.merge("side", "merge", day=2)
    repo.commit("rm", {"a.py": None}, day=3)
    assert not (repo.path / "a.py").exists()
    parents = subprocess.run(["git", "-C", repo.path, "log", "-1", "--format=%P", m],
                             capture_output=True, text=True, check=True).stdout.split()
    assert len(parents) == 2
```

- [ ] **Step 3:** Run `pytest tests/test_repo_builder.py -v` → FAIL (no `repo` fixture).
- [ ] **Step 4:** Implement `types.py`, `repo_builder.py`, `conftest.py`.
- [ ] **Step 5:** Run `pytest -v` → PASS.
- [ ] **Step 6:** Commit `feat: scaffold project, shared types, test repo builder`.

---

### Task 2: git_reader

**Files:**
- Create: `constellation/git_reader.py`, `tests/test_git_reader.py`

**Interfaces:**
- Consumes: `FileChange`, `Commit`, `Tag` (Task 1).
- Produces:
  - `class NotARepoError(Exception)`, `class UnknownRefError(Exception)`
  - `repo_root(path: Path) -> Path` — `rev-parse --show-toplevel`; raises `NotARepoError` (also when `path` doesn't exist).
  - `head_sha(root: Path, ref: str = "HEAD") -> str | None` — `rev-parse --verify --quiet <ref>^{commit}`; `None` if the repo has no commits; `UnknownRefError` if it has commits but `ref` doesn't resolve.
  - `parse_log(data: bytes) -> Iterator[Commit]` — pure parser (format below).
  - `iter_commits(root: Path, ref: str = "HEAD", exclude: str | None = None) -> Iterator[Commit]` — streams `git log --reverse --date-order --find-renames --raw --numstat -z --format=%x1e%H%x1f%P%x1f%aN%x1f%aE%x1f%at%x1f%s <ref | exclude..ref>` via `Popen`, yielding parsed commits in order (oldest first). Reading the whole stdout then calling `parse_log` is acceptable.
  - `read_tags(root: Path) -> list[Tag]` — `for-each-ref refs/tags --format=%(refname:short)%09%(objectname)%09%(*objectname)`; use `*objectname` when non-empty (annotated).
  - `head_loc(root: Path, ref: str) -> dict[str, int]` — every blob in `ls-tree -r -z <ref>`, contents via one `cat-file --batch` process; binary (a `\0` in the first 8000 bytes) → 0; else `count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)`.

**Log byte format** (verified against git 2.56): each record is `\x1e` + six `\x1f`-separated header fields + `\0`. If the commit has changes, `\n` follows, then N raw entries, then N numstat entries in the same order:
- raw: `:<srcmode> <dstmode> <srcsha> <dstsha> <STATUS>\0<path>\0`, and for `R###` a second `<newpath>\0` after the old path. Status letter = first char of the last space-separated field.
- numstat: `<added>\t<deleted>\t<path>\0`, or for renames `<added>\t<deleted>\t\0<old>\0<new>\0`; binary uses `-\t-`.
- Merge commits: header only. Root commit: empty parents field.

- [ ] **Step 1: Write failing tests** `tests/test_git_reader.py`:

```python
def test_parses_add_modify_rename_delete_binary(repo):
    repo.commit("one", {"sp ace.py": "a\nb\n", "bin.dat": b"\x00\x01"}, day=0)
    repo.mv("sp ace.py", "renamed ü.py")
    repo.commit("two", {"renamed ü.py": "a\nb\nc\n"}, author="Bo", day=1)
    repo.commit("three", {"bin.dat": None}, day=2)
    commits = list(iter_commits(repo.path))
    assert [c.subject for c in commits] == ["one", "two", "three"]
    assert commits[0].parents == ()
    assert commits[1].author == "Bo" and commits[1].ts == T0 + DAY
    assert set(commits[0].changes) == {
        FileChange("A", "bin.dat", None, 0, 0, True),
        FileChange("A", "sp ace.py", None, 2, 0, False)}
    assert commits[1].changes == (FileChange("R", "renamed ü.py", "sp ace.py", 1, 0, False),)
    assert commits[2].changes == (FileChange("D", "bin.dat", None, 0, 0, True),)

def test_merge_commit_has_parents_and_no_changes(repo): ...
    # main a → side b → merge; assert merge.parents has 2 shas, merge.changes == ()

def test_exclude_returns_only_new_commits(repo): ...
    # 3 commits; iter_commits(root, exclude=<sha of 1st>) → subjects of 2nd, 3rd

def test_repo_root_from_subdirectory(repo):
    repo.commit("one", {"pkg/sub/x.py": "1\n"})
    assert repo_root(repo.path / "pkg" / "sub") == repo.path.resolve()

def test_not_a_repo(tmp_path):
    with pytest.raises(NotARepoError): repo_root(tmp_path)
    with pytest.raises(NotARepoError): repo_root(tmp_path / "missing")

def test_empty_repo_and_unknown_ref(repo):
    assert head_sha(repo.path) is None
    repo.commit("one", {"a": "1\n"})
    with pytest.raises(UnknownRefError): head_sha(repo.path, "nope")

def test_tags_are_peeled(repo):
    s = repo.commit("one", {"a": "1\n"}); repo.tag("v1"); repo.tag("v2", annotated=True)
    assert sorted(read_tags(repo.path)) == [Tag("v1", s), Tag("v2", s)]

def test_head_loc(repo):
    repo.commit("one", {"a.py": "1\n2\n", "b.py": "1\n2", "e.py": "", "bin": b"\x00z"})
    assert head_loc(repo.path, "HEAD") == {"a.py": 2, "b.py": 2, "e.py": 0, "bin": 0}
```

- [ ] **Step 2:** Run → FAIL (import error).
- [ ] **Step 3:** Implement `git_reader.py` per the interfaces and byte format.
- [ ] **Step 4:** Run `pytest tests/test_git_reader.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: git_reader streams history via plumbing`.

---

### Task 3: model — file identity, events, LOC, directories

**Files:**
- Create: `constellation/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `Commit`, `FileChange`, `Tag` (Task 1); tests use `tests/history_helpers.py` (Task 1).
- Produces:

```python
@dataclass
class FileTrack:
    path: str                  # latest path of this file id
    paths: list[tuple[int, str]]  # (commit index, path) — every path it has had, ascending
    event_commit: np.ndarray   # int32, ascending commit indexes of changes
    event_added: np.ndarray    # int32
    event_deleted: np.ndarray  # int32
    event_author: np.ndarray   # int32 author ids
    born: int                  # commit index of the "A"
    died: int                  # commit index of the "D", -1 if alive at the end
    binary: bool               # last change was binary

class RepoHistory:
    commits: list[Commit]
    commit_ts: np.ndarray      # int64
    authors: list[str]         # author id → name (first-seen order)
    commit_author: np.ndarray  # int32 author id per commit
    files: list[FileTrack]     # file id → track (creation order)
    file_dir: np.ndarray       # int32 dir id per file (dir of latest path)
    dirs: list[str]            # dir id → path; dirs[0] == "" (root)
    dir_parent: np.ndarray     # int32, dir_parent[0] == -1
    tags: list[Tag]
    lineage: np.ndarray        # int32 per commit (filled by Task 4)

    @classmethod
    def build(cls, commits: list[Commit], tags: list[Tag] = (),
              head_loc: dict[str, int] | None = None) -> "RepoHistory"
    def files_alive_at(self, i: int) -> np.ndarray          # bool[len(files)]
    def loc_at(self, i: int) -> np.ndarray                  # int64[len(files)], 0 when not alive
    def index_at_time(self, ts: int) -> int                 # last commit index with ts <= ts; -1 if none
    def file_id(self, path: str) -> int | None              # id of the file currently (at the end) at path
```

- Rules: `"R"` moves the existing id to the new path (append to `paths`); `"A"` on a path whose id is dead (or unknown) creates a **new** id; `"D"` sets `died`. `"M"`/`"T"`/`"R"` on an unknown path (shallow clone) creates an id born at that commit. Every change appends an event. `loc_at(i)`: if `i == len(commits)-1` and `head_loc` given, use it for alive files (by latest path); otherwise cumulative `added - deleted` up to `i`, clipped at 0. Directories include every ancestor of every path (`"a/b/c.py"` → `""`, `"a"`, `"a/b"`).
- `build` sets `lineage` to zeros; Task 4 replaces that.

- [ ] **Step 1: Write failing tests** (`tests/test_model.py`):

```python
def test_rename_keeps_identity_and_events():
    h = RepoHistory.build([c(0, [A("a.py", 3)]), c(1, [R("a.py", "b/a.py", 1, 0)])])
    assert len(h.files) == 1 and h.files[0].path == "b/a.py"
    assert h.files[0].event_commit.tolist() == [0, 1]
    assert h.file_id("b/a.py") == 0 and h.file_id("a.py") is None

def test_path_reuse_after_delete_is_new_file():
    h = RepoHistory.build([c(0, [A("x", 2)]), c(1, [D("x", 2)]), c(2, [A("x", 5)])])
    assert len(h.files) == 2
    assert (h.files[0].born, h.files[0].died) == (0, 1)
    assert h.files_alive_at(1).tolist() == [False, False]
    assert h.files_alive_at(2).tolist() == [False, True]

def test_loc_cumulative_and_head_exact():
    h = RepoHistory.build([c(0, [A("a", 10)]), c(1, [M("a", 2, 5)])], head_loc={"a": 99})
    assert h.loc_at(0).tolist() == [10]
    assert h.loc_at(1).tolist() == [99]

def test_dirs_and_authors():
    h = RepoHistory.build([c(0, [A("a/b/c.py", 1)], author="Ana"), c(1, [A("d.py", 1)], author="Bo")])
    assert h.dirs[0] == "" and set(h.dirs) == {"", "a", "a/b"}
    assert h.dirs[h.file_dir[0]] == "a/b" and h.dirs[h.file_dir[1]] == ""
    assert h.dir_parent[h.dirs.index("a/b")] == h.dirs.index("a")
    assert h.authors == ["Ana", "Bo"] and h.commit_author.tolist() == [0, 1]

def test_index_at_time():
    h = RepoHistory.build([c(0, []), c(1, []), c(2, [])])
    assert h.index_at_time(T0 - 1) == -1
    assert h.index_at_time(T0 + DAY + 5) == 1

def test_modify_unknown_path_creates_file():   # shallow clone case
    h = RepoHistory.build([c(0, [M("old.py", 1, 0)], parents=("missing",))])
    assert h.files[0].born == 0
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `model.py`. Build per-file event lists in Python, convert to numpy once at the end.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: RepoHistory model with stable file ids`.

---

### Task 4: model — lineage and window

**Files:**
- Modify: `constellation/model.py` (add `assign_lineage` and `Window`; call `assign_lineage` from `build`)
- Test: `tests/test_lineage.py`

**Interfaces:**
- Consumes: `RepoHistory` (Task 3).
- Produces:
  - `assign_lineage(commits: list[Commit]) -> np.ndarray` (int32): walk first parents from the last commit → those commits get `0`. Then for each mainline merge in ascending index order, numbered `k = 1, 2, …`, every not-yet-assigned commit reachable through its non-first parents (following all parents, staying within unassigned commits) gets `k`. Commits still unassigned (unreachable) get `0`. Parents not present in `commits` are ignored.
  - `@dataclass(frozen=True) class Window: start: int; end: int` and `RepoHistory.window(since: int | None = None, until: int | None = None) -> Window` — timestamps → inclusive commit index range (`start` = first index with ts ≥ since, `end` = last with ts ≤ until; defaults 0 and last). No matching commit → `start = len(commits)` / `end = -1` as applicable. Property `Window.empty -> bool` (`start > end`); `Window.size -> int` (`max(0, end - start + 1)`). Empty history → `Window(0, -1)`. Metrics called with `start > playhead` return all-zero results.

- [ ] **Step 1: Write failing tests**:

```python
def test_feature_branch_gets_merge_lineage(repo):
    repo.commit("m0", {"a": "1\n"}, day=0)
    repo.checkout("f", create=True); repo.commit("f1", {"b": "1\n"}, day=1); repo.commit("f2", {"b": "2\n"}, day=2)
    repo.checkout("main"); repo.commit("m1", {"a": "2\n"}, day=3)
    repo.merge("f", "merge f", day=4)
    h = RepoHistory.build(list(iter_commits(repo.path)))
    by = {c.subject: int(l) for c, l in zip(h.commits, h.lineage)}
    assert by == {"m0": 0, "f1": 1, "f2": 1, "m1": 0, "merge f": 0}

def test_unrelated_histories(repo):
    repo.commit("a0", {"a": "1\n"}, day=0)
    repo.orphan("other"); repo.commit("o0", {"o": "1\n"}, day=1)
    repo.checkout("main"); repo.merge("other", "join", day=2, unrelated=True)
    h = RepoHistory.build(list(iter_commits(repo.path)))
    assert sum(1 for c in h.commits if not c.parents) == 2
    assert {c.subject: int(l) for c, l in zip(h.commits, h.lineage)}["o0"] == 1

def test_shallow_clone_does_not_crash(repo, tmp_path):
    for d in range(5): repo.commit(f"c{d}", {"a": f"{d}\n"}, day=d)
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "2", f"file://{repo.path}", shallow], check=True)
    h = RepoHistory.build(list(iter_commits(shallow)))
    assert len(h.commits) == 2 and h.lineage.tolist() == [0, 0]

def test_window():
    h = RepoHistory.build([c(i, []) for i in range(5)])
    assert h.window() == Window(0, 4)
    assert h.window(since=T0 + DAY, until=T0 + 3 * DAY) == Window(1, 3)
```

- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: lineage assignment and time windows`.

---

### Task 5: cache

**Files:**
- Create: `constellation/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: `repo_root`, `head_sha`, `iter_commits` (Task 2); `Commit`, `FileChange` (Task 1).
- Produces:
  - `cache_dir() -> Path` — `$GIT_CONSTELLATION_CACHE` or `~/.cache/git_constellation`.
  - `load_commits(root: Path, ref: str = "HEAD", progress: Callable[[int], None] | None = None, use_cache: bool = True) -> list[Commit]`
    - DB file: `cache_dir() / f"{sha1(f'{root.resolve()}\0{ref}').hexdigest()[:16]}.db"`.
    - Tables: `meta(key TEXT PRIMARY KEY, value TEXT)` holding `head` and `schema` (`"1"`); `commits(idx INTEGER PRIMARY KEY, sha, parents, author, email, ts, subject)`; `changes(commit_idx, status, path, old_path, added, deleted, binary)`.
    - Stored head == current head → load from DB. Stored head is an ancestor of current (`git merge-base --is-ancestor`) → `iter_commits(root, ref, exclude=stored_head)` (`exclude` passed as a keyword), append, update head. Otherwise (or schema mismatch, or corrupt DB) → rebuild from scratch.
    - Empty repo → `[]`, nothing written.
    - `progress(n)` called every 500 commits parsed, and once at the end with the total.

- [ ] **Step 1: Write failing tests**:

```python
def test_round_trip_equals_fresh_parse(repo):
    repo.commit("one", {"a b.py": "1\n", "bin": b"\x00"}); repo.mv("a b.py", "c.py"); repo.commit("two", day=1)
    first = load_commits(repo.path)
    assert first == list(iter_commits(repo.path))
    assert load_commits(repo.path) == first          # served from cache

def test_incremental_append(repo, monkeypatch):
    repo.commit("one", {"a": "1\n"}); load_commits(repo.path)
    repo.commit("two", {"a": "2\n"}, day=1)
    calls = []
    real = cache.iter_commits
    monkeypatch.setattr(cache, "iter_commits", lambda *a, **k: (calls.append(k), real(*a, **k))[1])
    assert [c.subject for c in load_commits(repo.path)] == ["one", "two"]
    assert calls[0].get("exclude") is not None

def test_rewritten_history_rebuilds(repo):
    repo.commit("one", {"a": "1\n"}); repo.commit("two", {"a": "2\n"}, day=1); load_commits(repo.path)
    subprocess.run(["git", "-C", repo.path, "reset", "-q", "--hard", "HEAD~1"], check=True)
    repo.commit("other", {"a": "3\n"}, day=2)
    assert [c.subject for c in load_commits(repo.path)] == ["one", "other"]

def test_corrupt_db_rebuilds(repo):
    repo.commit("one", {"a": "1\n"}); load_commits(repo.path)
    for f in cache_dir().glob("*.db"): f.write_bytes(b"garbage")
    assert len(load_commits(repo.path)) == 1

def test_empty_repo(repo):
    assert load_commits(repo.path) == []
```

- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: sqlite history cache with incremental append`.

---

### Task 6: metrics — hotspots, change counts, fossils

**Files:**
- Create: `constellation/metrics/__init__.py` (re-exports), `constellation/metrics/activity.py`, `tests/test_metrics_activity.py`

**Interfaces:**
- Consumes: `RepoHistory`, `Window` (Tasks 3–4).
- Produces (all arrays length `len(h.files)`):
  - `change_counts(h, start: int, playhead: int) -> np.ndarray` (int64) — events with `start <= commit <= playhead`.
  - `hotspot_raw(h, start: int, playhead: int) -> np.ndarray` (float64) — for alive, non-binary files: `Σ 0.5 ** (age_days / 180) × log2(1 + loc_at(playhead))` over events in range, `age_days = (ts[playhead] - ts[event]) / 86400`. Others 0.
  - `hotspots(h, w: Window, playhead: int) -> np.ndarray` — `hotspot_raw(h, w.start, playhead) / max(hotspot_raw(h, w.start, w.end))`, clamped to [0, 1]; all zeros if that max is 0.
  - `fossil_age_days(h, playhead: int) -> np.ndarray` (float64) — days from last event ≤ playhead to `ts[playhead]`; `nan` for files not alive.
  - `FOSSIL_DAYS = 365`, `HALF_LIFE_DAYS = 180`.
- Use `np.searchsorted` on each file's `event_commit`; a per-file Python loop over 10k files is acceptable for this phase if it stays under the 100 ms budget in the perf test.

- [ ] **Step 1: Write failing tests**:

```python
def test_hotspot_raw_exact():
    # file "a" with 3 lines at playhead; events at day 0 and day 180; playhead = day 180
    h = RepoHistory.build([c(0, [A("a", 2)], day=0), c(1, [M("a", 1, 0)], day=180)])
    assert hotspot_raw(h, 0, 1).tolist() == pytest.approx([1.5 * 2.0])   # (0.5 + 1) * log2(1+3)

def test_hotspots_normalized_to_window_end_and_binary_excluded():
    h = RepoHistory.build([c(0, [A("a", 1), A("img", 0, binary=True)]), c(1, [M("a", 1, 0)]), c(2, [M("a", 1, 0)])])
    w = h.window()
    assert hotspots(h, w, 2)[0] == 1.0
    assert 0 < hotspots(h, w, 0)[0] < 1.0      # early history is cooler
    assert hotspots(h, w, 2)[1] == 0.0

def test_change_counts_respect_window():
    h = RepoHistory.build([c(i, [M("a", 1, 0)] if i else [A("a", 1)]) for i in range(4)])
    assert change_counts(h, 1, 2).tolist() == [2]

def test_fossil_age():
    h = RepoHistory.build([c(0, [A("a", 1), A("b", 1)], day=0), c(1, [M("b", 1, 0)], day=400), c(2, [D("b", 2)], day=401)])
    ages = fossil_age_days(h, 1)
    assert ages[0] == 400 and ages[1] == 0
    assert np.isnan(fossil_age_days(h, 2)[1])

def test_perf_10k_files_under_100ms(): ...
    # synthetic: 10_000 files, 50_000 commits each touching 3 random files (seeded);
    # time.perf_counter around hotspots(...) at the last commit → < 0.1 s (mark @pytest.mark.slow)
```

- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS. Register the `slow` marker in `pyproject.toml`.
- [ ] **Step 5:** Commit `feat: hotspot, change-count and fossil metrics`.

---

### Task 7: metrics — ownership and bus factor

**Files:**
- Create: `constellation/metrics/ownership.py`, `tests/test_metrics_ownership.py`; Modify: `constellation/metrics/__init__.py` (re-export)

**Interfaces:**
- Consumes: `RepoHistory` (Task 3); `change_counts` (Task 6).
- Produces:
  - `@dataclass class Ownership: dominant: np.ndarray  # int32 author id, -1 if no events; share: np.ndarray  # float64 0..1; top3: list[list[tuple[int, float]]]  # per file (author id, share), descending`
  - `ownership(h, start: int, playhead: int) -> Ownership` — share of `added + deleted` per author within range; if a file's churn in range is 0 but it has events, share by event count instead. Ties → lowest author id.
  - `bus_factor_risk(h, start: int, playhead: int) -> np.ndarray` (bool) — alive at playhead AND `share >= 0.8` AND `change_counts >= 10` AND the dominant author has **no commits at all** (any file) with `ts[playhead] - 180*86400 < ts <= ts[playhead]`.
  - Constants `BUS_SHARE = 0.8`, `BUS_MIN_CHANGES = 10`, `BUS_SILENT_DAYS = 180`.

- [ ] **Step 1: Write failing tests**:

```python
def test_share_by_churn():
    h = RepoHistory.build([c(0, [A("f", 8)], author="Ana"), c(1, [M("f", 2, 0)], author="Bo")])
    o = ownership(h, 0, 1)
    assert h.authors[o.dominant[0]] == "Ana" and o.share[0] == pytest.approx(0.8)
    assert [(h.authors[a], s) for a, s in o.top3[0]] == [("Ana", pytest.approx(0.8)), ("Bo", pytest.approx(0.2))]

def test_bus_factor_risk_when_owner_went_quiet():
    cs = [c(i, [A("f", 1)] if i == 0 else [M("f", 1, 0)], author="Ana", day=i) for i in range(10)]
    cs.append(c(10, [A("g", 1)], author="Bo", day=400))
    h = RepoHistory.build(cs)
    assert bus_factor_risk(h, 0, 10).tolist() == [True, False]
    assert bus_factor_risk(h, 0, 9).tolist()[0] is False    # Ana still active at day 9

def test_no_risk_below_10_changes(): ...   # same as above with 9 commits → False
```

- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: ownership and bus-factor metrics`.

---

### Task 8: metrics — coupling index

**Files:**
- Create: `constellation/metrics/coupling.py`, `tests/test_metrics_coupling.py`; Modify: `constellation/metrics/__init__.py` (re-export)

**Interfaces:**
- Consumes: `RepoHistory` (Task 3); `change_counts` (Task 6).
- Produces:
  - `@dataclass(frozen=True) class CouplingPair: a: int; b: int; shared: int; strength: float` (`a < b`, file ids)
  - `class CouplingIndex: __init__(self, h: RepoHistory, snapshot_every: int = 500)`; `shared_counts(self, upto: int) -> Counter[tuple[int, int]]` (pairs co-changed in commits `0..upto`); `pairs(self, start: int, playhead: int, min_shared: int = 5, min_strength: float = 0.5) -> list[CouplingPair]` sorted by `(strength, shared)` descending.
  - Rules: only commits with `2 <= files touched <= 50` contribute pairs; `shared(range) = shared_counts(playhead) - shared_counts(start - 1)`; `strength = shared / min(change_counts(a), change_counts(b))` over the same range (change counts include giant commits). Snapshots (Counter copies) are taken at commit indexes that are multiples of `snapshot_every`; `shared_counts(i)` = nearest snapshot ≤ i plus replay.
  - Constants `MAX_COMMIT_FILES = 50`, `MIN_SHARED = 5`, `MIN_STRENGTH = 0.5`.

- [ ] **Step 1: Write failing tests**:

```python
def test_strength_and_giant_commit_ignored():
    cs = [c(0, [A("a", 1), A("b", 1)])]
    cs += [c(i, [M("a", 1, 0), M("b", 1, 0)]) for i in range(1, 5)]      # shared = 5 incl. creation
    cs.append(c(5, [M("a", 1, 0)]))                                     # a alone
    cs.append(c(6, [M("a", 1, 0), M("b", 1, 0)] + [A(f"z{k}", 1) for k in range(49)]))  # 51 files
    h = RepoHistory.build(cs)
    (p,) = CouplingIndex(h).pairs(0, 6)
    assert (p.a, p.b, p.shared) == (0, 1, 5)
    assert p.strength == pytest.approx(5 / 6)       # min(changes a=7, b=6)

def test_snapshots_match_brute_force():
    # seeded random history: 40 files, 1200 commits of 1–6 files
    idx = CouplingIndex(h, snapshot_every=100)
    for upto in (0, 99, 100, 101, 777, 1199):
        assert idx.shared_counts(upto) == brute_force_shared(h, upto)

def test_window_subtracts_earlier_pairs(): ...
    # 5 shared commits at 0..4, window start=3 → shared 2 → no pair returned at min_shared=5
```

- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: incremental coupling index with snapshots`.

---

### Task 9: insights

**Files:**
- Create: `constellation/insights.py`, `tests/test_insights.py`

**Interfaces:**
- Consumes: everything from `constellation.metrics` (Tasks 6–8), `RepoHistory`, `Window`.
- Produces:
  - `@dataclass(frozen=True) class Finding: kind: str; title: str; detail: str; file_ids: tuple[int, ...]` — `kind ∈ {"summary", "hotspot", "coupling", "bus_factor", "fossil"}`.
  - `findings(h, w: Window, playhead: int, coupling: CouplingIndex | None = None) -> list[Finding]` — order: 1 summary, ≤10 hotspots (score > 0, descending), ≤10 coupling pairs whose files are in **different directories**, all bus-factor risks (≤10, by change count), ≤10 fossils (age > 365 days, by LOC descending, LOC > 0).
  - Exact copy (format strings; `{pct}` is an integer percent, `{days}` an integer):
    - summary title: `"{n_commits} commits by {n_authors} authors, {n_files} files"`; detail: `"From {first:%Y-%m-%d} to {last:%Y-%m-%d}."`
    - hotspot: title `"{path} is a hotspot"`, detail `"Changed {changes} times, {loc} lines, mostly by {author} ({pct}%)."`
    - coupling: title `"{path_a} ↔ {path_b}"`, detail `"Change together {pct}% of the time ({shared} commits) across different folders."`
    - bus_factor: title `"{path} depends on {author}"`, detail `"{author} wrote {pct}% of it and hasn't committed in {days} days."`
    - fossil: title `"{path} is a fossil"`, detail `"Untouched for {days} days ({loc} lines)."`
  - Empty history → `[]`.

- [ ] **Step 1: Write failing tests** — build one small history covering each kind (reuse helpers) and assert the exact `title`/`detail` strings for one finding of each kind, the ordering of kinds, the 10-item caps (12 hot files → 10 hotspot findings), and that same-directory coupling pairs are excluded.
- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: plain-language insights`.

---

### Task 10: report and CLI

**Files:**
- Create: `constellation/report.py`, `constellation/__main__.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `repo_root`, `head_sha`, `read_tags`, `head_loc`, `NotARepoError`, `UnknownRefError` (Task 2); `load_commits` (Task 5); `RepoHistory` (Tasks 3–4); `findings`, `CouplingIndex` (Tasks 8–9).
- Produces:
  - `render_report(repo_name: str, h: RepoHistory, w: Window, items: list[Finding]) -> str` — one self-contained HTML document (inline CSS, no external requests, light/dark via `prefers-color-scheme`), sections in finding-kind order, all text HTML-escaped.
  - `main(argv: list[str] | None = None) -> int` — args: `path` (default `.`), `--branch` (default `HEAD`), `--since`/`--until` (`YYYY-MM-DD`, local midnight), `--report PATH`, `--no-cache`. Behaviour:
    - Not a repo → stderr `error: {path} is not inside a git repository`, return 1. Unknown ref → `error: unknown branch or ref '{ref}'`, return 1. Bad date → argparse error (exit 2).
    - Empty repo → stdout `No commits yet.`, return 0.
    - Progress to stderr: `Reading history… {n} commits` (via `load_commits` progress).
    - With `--report`: write HTML, print `Report written to {path}`. Without: print findings as plain text (`title` line, indented `detail` line), then `Galaxy window arrives in Phase 2 — use --report for HTML.`
    - Playhead = `w.end` if the window is non-empty, else the last commit; the summary's `n_commits` is `w.size`.
  - `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 1: Write failing tests** (`tests/test_cli.py`, call `main([...])` with `capsys`):
  - not-a-repo → returns 1, stderr contains `is not inside a git repository`
  - empty repo → `No commits yet.`
  - small repo with a hot file → stdout contains `is a hotspot`
  - `--report out.html` → file exists, starts with `<!doctype html>`, contains the hotspot title, contains no `http://`/`https://` resource URLs
  - path with `<script>` in a filename is escaped in the HTML
  - `--since` later than all commits → summary reports `0 commits`, returns 0
- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5: End-to-end check.** Run `python -m constellation ~/Python_Projects/git_constellation` (the project's own repo by now) and `--report` into the scratchpad; open the HTML and confirm findings are sane. Run the full suite including `-m slow`.
- [ ] **Step 6:** Commit `feat: HTML report and CLI`.
