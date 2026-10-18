import time

import numpy as np
import pytest

from constellation.picking import pick

O = np.zeros(3)
D = np.array([0.0, 0.0, -1.0])


def _spheres(zs, r=1.0):
    c = np.array([[0, 0, z] for z in zs], float)
    return c, np.full(len(zs), r), np.ones(len(zs), bool)


def test_nearer_wins():
    c, r, m = _spheres([-20, -10])
    assert pick(O, D, c, r, m) == 1


def test_masked_out_ignored():
    c, r, m = _spheres([-20, -10])
    m[1] = False
    assert pick(O, D, c, r, m) == 0


def test_behind_ignored():
    c, r, m = _spheres([10, -20])
    assert pick(O, D, c, r, m) == 1


def test_miss():
    c, r, m = _spheres([-10])
    c[0, 0] = 5
    assert pick(O, D, c, r, m) is None
    assert pick(O, D, c[:0], r[:0], m[:0]) is None


@pytest.mark.slow
def test_perf_100k():
    rng = np.random.default_rng(1)
    c = rng.uniform(-500, 500, (100_000, 3))
    r = rng.uniform(0.5, 2, 100_000)
    m = rng.random(100_000) > 0.1
    d = np.array([0.3, 0.2, -1.0])
    d /= np.linalg.norm(d)
    pick(O, d, c, r, m)
    best = min(
        (lambda t0: (pick(O, d, c, r, m), time.perf_counter() - t0)[1])(time.perf_counter())
        for _ in range(10)
    )
    assert best < 0.005
