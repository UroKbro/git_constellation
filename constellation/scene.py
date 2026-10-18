"""Per-frame scene state: planets, stars, pulses, comets and replay animations.

`now` is wall-clock seconds; animations are keyed on the `now` at which
`advance` saw the event. Pure numpy, no GL."""
from dataclasses import dataclass

import numpy as np

from constellation.collapse import CollapsePlan, in_window
from constellation.layout import Layout
from constellation.model import RepoHistory, Window
from constellation.palette import language_color, lineage_color

ADD_S = 0.6
DEL_S = 0.8
REN_S = 0.6
PULSE_LIFE = 1.2
COMET_LIFE = 1.5
MAX_PULSES = 20_000
MAX_STEP = 256
COMET_OFFSET = 4.0
STAR_RGB = (1.0, 0.92, 0.78)

_F = np.float32


def planet_radius(loc) -> np.ndarray:
    return np.clip(0.15 * np.sqrt(np.asarray(loc, dtype=np.float64)), 0.25, 3.0)


def pulse_brightness(churn) -> np.ndarray:
    c = np.asarray(churn, dtype=np.float64)
    return 0.4 + 0.6 * np.minimum(1.0, np.log2(1.0 + c) / 10.0)


def ease_in_out_cubic(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 4 * t ** 3, 1 - (-2 * t + 2) ** 3 / 2)


@dataclass
class PlanetBatch:
    pos: np.ndarray     # (n, 3)
    radius: np.ndarray  # (n,)
    color: np.ndarray   # (n, 4) rgb + emissive factor
    ids: np.ndarray     # (n,) int32 file id, or -(group + 1) for aggregates


@dataclass
class StarBatch:
    pos: np.ndarray
    radius: np.ndarray
    color: np.ndarray   # (n, 3)


@dataclass
class PulseBatch:
    start: np.ndarray   # (n, 3)
    end: np.ndarray     # (n, 3)
    color: np.ndarray   # (n, 4) rgb + brightness
    born: np.ndarray
    life: np.ndarray


@dataclass
class CometBatch:
    pos: np.ndarray     # (n, 3)
    color: np.ndarray   # (n, 4)
    color2: np.ndarray  # (n, 4) second braid colour; equals `color` for non-merges
    commit: np.ndarray  # (n,) int32
    born: np.ndarray
    life: np.ndarray


_PULSE_KEYS = ("start", "end", "color", "born")
_COMET_KEYS = ("pos", "color", "color2", "commit", "born")


def _empty_pulses() -> dict:
    return {"start": np.zeros((0, 3)), "end": np.zeros((0, 3)),
            "color": np.zeros((0, 4)), "born": np.zeros(0)}


def _empty_comets() -> dict:
    return {"pos": np.zeros((0, 3)), "color": np.zeros((0, 4)),
            "color2": np.zeros((0, 4)), "commit": np.zeros(0, dtype=np.int32),
            "born": np.zeros(0)}


def _prune(store: dict, life: float, now: float) -> None:
    keep = store["born"] + life > now
    if not keep.all():
        for k in store:
            store[k] = store[k][keep]


class Scene:
    def __init__(self, h: RepoHistory, layout: Layout, plan: CollapsePlan,
                 window: Window):
        self.h, self.layout, self.plan, self.window = h, layout, plan, window
        n = self.n_files = len(h.files)
        nc = len(h.commits)
        self.in_window = in_window(h, window)
        self.file_rgb = np.array([language_color(f.path) for f in h.files],
                                 dtype=np.float64).reshape(n, 3)
        fams: dict[tuple, int] = {}  # language family = distinct palette colour
        self._fam = np.array([fams.setdefault(tuple(rgb), len(fams)) for rgb in self.file_rgb],
                             dtype=np.int64)
        self._fam_rgb = np.array(list(fams), dtype=np.float64).reshape(-1, 3)

        # per-event churn aligned with h._ev_* (same stable sort as the model)
        if n:
            com = np.concatenate([f.event_commit for f in h.files]).astype(np.int64)
            churn = np.concatenate([f.event_added.astype(np.int64) + f.event_deleted
                                    for f in h.files])
            self._ev_churn = churn[np.argsort(com, kind="stable")]
        else:
            self._ev_churn = np.zeros(0, dtype=np.int64)

        # placements: (file, start commit) keys, sorted, for vectorised place_at
        starts = np.asarray(layout.file_place_start, dtype=np.int64)
        n_places = len(layout.place_pos)
        counts = np.diff(np.append(starts, n_places))
        place_commit = np.asarray(layout.place_commit, dtype=np.int64)
        place_file = np.repeat(np.arange(n, dtype=np.int64), counts)
        self._B = nc + 2
        self._starts = starts
        self._pkey = place_file * self._B + place_commit
        first = np.zeros(n_places, dtype=bool)
        first[starts[counts > 0]] = True
        later = np.flatnonzero(~first)  # rename placements
        o = np.argsort(place_commit[later], kind="stable")
        self._ren_place = later[o]
        self._ren_commit = place_commit[later][o]
        self._place_file = place_file
        self._place_pos = np.asarray(layout.place_pos, dtype=np.float64)
        self._dir_pos = np.asarray(layout.dir_pos, dtype=np.float64)
        self._place_dir = np.asarray(layout.place_dir, dtype=np.int64)

        born, died = h._born, h._died
        self._born_order = np.argsort(born, kind="stable")
        self._born_sorted = born[self._born_order]
        dead = np.flatnonzero(died >= 0)
        self._died_order = dead[np.argsort(died[dead], kind="stable")]
        self._died_sorted = died[self._died_order]

        # commit colours: lineage and the second parent's lineage (braid)
        lin = np.array([lineage_color(int(k)) for k in range(int(h.lineage.max(initial=0)) + 1)])
        self._c1 = lin[h.lineage] if nc else np.zeros((0, 3))
        sha = {cm.sha: i for i, cm in enumerate(h.commits)}
        l2 = h.lineage.copy()
        for i, cm in enumerate(h.commits):
            if len(cm.parents) > 1 and cm.parents[1] in sha:
                l2[i] = h.lineage[sha[cm.parents[1]]]
        self._c2 = lin[l2] if nc else np.zeros((0, 3))

        # dirs: depth levels for subtree sums, owning collapse group per dir
        parent = np.asarray(h.dir_parent, dtype=np.int64)
        self._dir_parent = parent
        n_dirs = len(h.dirs)
        depth = np.array([0 if d == "" else d.count("/") + 1 for d in h.dirs])
        self._levels = [np.flatnonzero(depth == k) for k in range(int(depth.max(initial=0)), 0, -1)]
        gid = {d: g for g, d in enumerate(plan.group_dir)}
        self._dir_group = np.full(n_dirs, -1, dtype=np.int64)
        for d in np.argsort(depth, kind="stable"):
            p = parent[d]
            self._dir_group[d] = (self._dir_group[p] if p >= 0 and self._dir_group[p] >= 0
                                  else gid.get(int(d), -1))
        self._group_pos = self._dir_pos[np.array(plan.group_dir, dtype=np.int64)]

        # animation state, per file
        self._add_t = np.full(n, -np.inf)
        self._ren_t = np.full(n, -np.inf)
        self._ren_from = np.full(n, -1, dtype=np.int64)
        self._ren_to = np.full(n, -1, dtype=np.int64)
        self._del_t = np.full(n, -np.inf)
        # delete fade: radius from loc just before death, last placement before death
        ev_died = died[h._ev_file]
        before = (ev_died >= 0) & (h._ev_commit < ev_died)
        loc = np.maximum(0, np.bincount(h._ev_file[before], weights=h._ev_delta[before],
                                        minlength=n))
        self._del_r = planet_radius(loc)
        self._del_place = np.zeros(n, dtype=np.int64)
        if len(dead):
            self._del_place[dead] = self._place(dead, np.maximum(died[dead] - 1, born[dead]))
        self._path_files: dict[str, list[int]] | None = None
        self._dying = np.zeros(0, dtype=np.int64)
        self._anim_until = -np.inf
        self._pulses = _empty_pulses()
        self._comets = _empty_comets()
        self._key = None
        self._base: dict = {}

    # ---- helpers -----------------------------------------------------------
    def _place(self, files: np.ndarray, commits) -> np.ndarray:
        """Vectorised Layout.place_at."""
        files = np.asarray(files, dtype=np.int64)
        k = np.searchsorted(self._pkey, files * self._B + commits, side="right") - 1
        return np.maximum(k, self._starts[files]) if len(files) else k

    def _hidden(self, groups: np.ndarray) -> np.ndarray:
        """Group >= 0 and not expanded."""
        out = groups >= 0
        if self.plan.expanded and out.any():
            out &= ~np.isin(groups, list(self.plan.expanded))
        return out

    def _clear(self) -> None:
        self._pulses = _empty_pulses()
        self._comets = _empty_comets()
        for a in (self._add_t, self._ren_t, self._del_t):
            a.fill(-np.inf)
        self._dying = np.zeros(0, dtype=np.int64)
        self._anim_until = -np.inf

    # ---- events ------------------------------------------------------------
    def advance(self, prev: int, cur: int, now: float) -> None:
        """Spawn comets/pulses/animations for commits (prev, cur]; a backwards
        step or a jump of more than 256 commits clears them. prev == cur is a no-op."""
        if prev == cur:
            return
        if not 0 < cur - prev <= MAX_STEP:
            self._clear()
            return
        win = self.in_window
        self._spawn(prev, cur, now)

        lo = np.searchsorted(self._born_sorted, prev + 1, side="left")
        hi = np.searchsorted(self._born_sorted, cur, side="right")
        f = self._born_order[lo:hi]
        f = f[win[f]]
        self._add_t[f] = now

        lo = np.searchsorted(self._ren_commit, prev + 1, side="left")
        hi = np.searchsorted(self._ren_commit, cur, side="right")
        p = self._ren_place[lo:hi]
        f = self._place_file[p]
        ok = win[f]
        p, f = p[ok], f[ok]
        self._ren_t[f] = now          # later placements overwrite earlier ones
        self._ren_to[f] = p
        self._ren_from[f] = p - 1

        lo = np.searchsorted(self._died_sorted, prev + 1, side="left")
        hi = np.searchsorted(self._died_sorted, cur, side="right")
        f = self._died_order[lo:hi]
        f = f[win[f]]
        self._del_t[f] = now
        if len(f):
            self._dying = np.union1d(self._dying, f)
        self._anim_until = max(self._anim_until, now + max(ADD_S, DEL_S, REN_S))

    def _merge_stars(self, i: int) -> np.ndarray:
        """Star positions of the alive files at the paths a merge-only commit's
        merge_changes leave in its tree."""
        h = self.h
        if self._path_files is None:
            self._path_files = {}
            for fid, f in enumerate(h.files):
                for _, p in f.paths:
                    self._path_files.setdefault(p, []).append(fid)
        hit: set[int] = set()
        for ch in h.commits[i].merge_changes:
            if ch.status == "D":
                continue
            for fid in self._path_files.get(ch.path, ()):
                f = h.files[fid]
                if not (self.in_window[fid] and f.born <= i and (f.died == -1 or f.died > i)):
                    continue
                if next(p for c, p in reversed(f.paths) if c <= i) == ch.path:
                    hit.add(fid)
        if not hit:
            return np.zeros((0, 3))
        ids = np.fromiter(hit, dtype=np.int64)
        return self._dir_pos[self._place_dir[self._place(ids, i)]]

    def _spawn(self, prev: int, cur: int, now: float) -> None:
        h, n = self.h, self.n_files
        lo = int(np.searchsorted(h._ev_commit, prev + 1, side="left"))
        hi = int(np.searchsorted(h._ev_commit, cur, side="right"))
        ef, ec, ch = h._ev_file[lo:hi], h._ev_commit[lo:hi], self._ev_churn[lo:hi]
        d = h._died[ef]
        ok = self.in_window[ef] & (h._born[ef] <= ec) & ((d == -1) | (d >= ec))
        key, inv = np.unique(ec[ok] * max(n, 1) + ef[ok], return_inverse=True)
        churn = np.bincount(inv, weights=ch[ok], minlength=len(key))
        pf, pcm = key % max(n, 1), key // max(n, 1)
        places = self._place(pf, pcm)

        m = cur - prev
        slot = pcm - (prev + 1)
        star = self._dir_pos[self._place_dir[places]] if len(places) else np.zeros((0, 3))
        cnt = np.bincount(slot, minlength=m)
        cen = np.stack([np.bincount(slot, weights=star[:, a], minlength=m) for a in range(3)],
                       axis=1) / np.maximum(cnt, 1)[:, None]
        for k in np.flatnonzero(cnt == 0):
            i = prev + 1 + int(k)
            if i < len(h.commits) and not h.commits[i].changes and h.commits[i].merge_changes:
                stars = self._merge_stars(i)
                if len(stars):
                    cen[k] = stars.mean(axis=0)
        norm = np.linalg.norm(cen, axis=1)
        out = np.where(norm[:, None] > 1e-9, cen / np.maximum(norm, 1e-12)[:, None],
                       np.array([0.0, 1.0, 0.0]))
        pos = cen + COMET_OFFSET * out
        commits = np.arange(prev + 1, cur + 1)
        valid = commits < len(h.commits)
        commits, pos = commits[valid], pos[valid]
        one = np.ones((len(commits), 1))
        c1 = np.hstack([self._c1[commits], one])
        new_c = {"pos": pos, "color": c1, "color2": np.hstack([self._c2[commits], one]),
                 "commit": commits.astype(np.int32), "born": np.full(len(commits), now)}
        for k in _COMET_KEYS:
            self._comets[k] = np.concatenate([self._comets[k], new_c[k]])

        end = self._place_pos[places] if len(places) else np.zeros((0, 3))
        g = self.plan.member_to_group[pf]
        hid = self._hidden(g)
        if hid.any():
            end = end.copy()
            end[hid] = self._group_pos[g[hid]]
        new_p = {"start": pos[slot] if len(slot) else np.zeros((0, 3)), "end": end,
                 "color": np.hstack([self._c1[pcm].reshape(-1, 3),
                                     pulse_brightness(churn)[:, None]]),
                 "born": np.full(len(key), now)}
        for k in _PULSE_KEYS:
            self._pulses[k] = np.concatenate([self._pulses[k], new_p[k]])[-MAX_PULSES:]

    # ---- per-frame batches -------------------------------------------------
    def _ensure(self, playhead: int) -> dict:
        key = (playhead, frozenset(self.plan.expanded))
        if key == self._key:
            return self._base
        h = self.h
        alive = h.files_alive_at(playhead) & self.in_window
        loc = h.loc_at(playhead) if self.n_files else np.zeros(0, dtype=np.int64)
        ids = np.flatnonzero(alive)
        places = self._place(ids, playhead)
        grp = self.plan.member_to_group[ids]
        hid = self._hidden(grp)
        shown, sp = ids[~hid], places[~hid]

        hg, hf = grp[hid].astype(np.int64), ids[hid]
        n_g = len(self.plan.group_dir)
        present = np.flatnonzero(np.bincount(hg, minlength=n_g)) if len(hg) else np.zeros(0, np.int64)
        sum_loc = np.bincount(hg, weights=loc[hf], minlength=n_g)
        n_e = max(len(self._fam_rgb), 1)
        fam_ct = np.bincount(hg * n_e + self._fam[hf], minlength=n_g * n_e).reshape(n_g, n_e)
        agg_rgb = self._fam_rgb[fam_ct[present].argmax(axis=1)] if len(present) else np.zeros((0, 3))

        nd = len(h.dirs)
        sub = np.bincount(self._place_dir[places], minlength=nd).astype(np.int64)
        for lvl in self._levels:
            np.add.at(sub, self._dir_parent[lvl], sub[lvl])
        dg = self._dir_group
        vis = (sub > 0) & ~self._hidden(dg)

        self._base = {
            "alive": alive, "nfile": len(shown), "places": sp,
            "ids": np.concatenate([shown, -(present + 1)]).astype(np.int32),
            "pos": np.concatenate([self._place_pos[sp], self._group_pos[present]]),
            "radius": np.concatenate([planet_radius(loc[shown]),
                                      2 * planet_radius(sum_loc[present])]),
            "rgb": np.concatenate([self.file_rgb[shown], agg_rgb]),
            "star_pos": self._dir_pos[vis].astype(_F),
            "star_radius": (0.6 + 0.12 * np.sqrt(sub[vis])).astype(_F),
        }
        self._base["star_color"] = np.tile(np.array(STAR_RGB, dtype=_F), (int(vis.sum()), 1))
        self._key = key
        return self._base

    def planets(self, playhead: int, now: float) -> PlanetBatch:
        b = self._ensure(playhead)
        pos, radius = b["pos"], b["radius"]
        color = np.hstack([b["rgb"], np.ones((len(radius), 1))])
        ids = b["ids"]
        if now < self._anim_until:
            pos, radius = pos.copy(), radius.copy()
            nf = b["nfile"]
            fid = ids[:nf].astype(np.int64)
            age = now - self._add_t[fid]
            m = np.flatnonzero(age < ADD_S)
            if len(m):
                s = np.clip(age[m] / ADD_S, 0.0, 1.0)
                radius[m] *= s
                color[m, 3] = 3.0 - 2.0 * s
            age = now - self._ren_t[fid]
            m = np.flatnonzero((age < REN_S) & (self._ren_to[fid] == b["places"]))
            if len(m):
                t = ease_in_out_cubic(np.clip(age[m] / REN_S, 0.0, 1.0))[:, None]
                f = fid[m]
                pos[m] = ((1 - t) * self._place_pos[self._ren_from[f]]
                          + t * self._place_pos[self._ren_to[f]])
            d = self._dying
            if len(d):
                age = now - self._del_t[d]
                self._dying = d = d[age < DEL_S]
                age = now - self._del_t[d]
                keep = ~b["alive"][d] & ~self._hidden(self.plan.member_to_group[d])
                d, age = d[keep], age[keep]
                if len(d):
                    s = (1.0 - np.clip(age / DEL_S, 0.0, 1.0))
                    pos = np.concatenate([pos, self._place_pos[self._del_place[d]]])
                    radius = np.concatenate([radius, self._del_r[d] * s])
                    color = np.concatenate([color, np.hstack([self.file_rgb[d] * s[:, None],
                                                              np.ones((len(d), 1))])])
                    ids = np.concatenate([ids, d.astype(np.int32)])
        return PlanetBatch(pos.astype(_F), radius.astype(_F), color.astype(_F),
                           ids.astype(np.int32))

    def stars(self, playhead: int) -> StarBatch:
        b = self._ensure(playhead)
        return StarBatch(b["star_pos"], b["star_radius"], b["star_color"])

    def pulses(self, now: float) -> PulseBatch:
        _prune(self._pulses, PULSE_LIFE, now)
        p = self._pulses
        return PulseBatch(p["start"].astype(_F), p["end"].astype(_F), p["color"].astype(_F),
                          p["born"].astype(_F), np.full(len(p["born"]), PULSE_LIFE, dtype=_F))

    def comets(self, now: float) -> CometBatch:
        _prune(self._comets, COMET_LIFE, now)
        c = self._comets
        return CometBatch(c["pos"].astype(_F), c["color"].astype(_F), c["color2"].astype(_F),
                          c["commit"].astype(np.int32), c["born"].astype(_F),
                          np.full(len(c["born"]), COMET_LIFE, dtype=_F))
