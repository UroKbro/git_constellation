"""Vectorised ray-sphere picking. Pure numpy."""
from __future__ import annotations

import numpy as np


def pick(origin, direction, centers: np.ndarray, radii: np.ndarray,
         mask: np.ndarray) -> int | None:
    """Index of the nearest masked sphere hit by the ray in front of origin, or None."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    oc = centers[idx] - origin
    tca = oc @ direction
    r = radii[idx]
    d2 = np.einsum("ij,ij->i", oc, oc) - tca * tca
    h2 = r * r - d2
    ok = (h2 >= 0) & (tca + r > 0)  # cheap bound: far hit is at most tca + r
    if not ok.any():
        return None
    thc = np.sqrt(np.where(ok, h2, 0.0))
    t0, t1 = tca - thc, tca + thc
    t = np.where(t0 > 0, t0, t1)
    t = np.where(ok & (t > 0), t, np.inf)
    i = int(np.argmin(t))
    return int(idx[i]) if np.isfinite(t[i]) else None
