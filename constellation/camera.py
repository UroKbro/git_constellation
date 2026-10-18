"""Orbit camera with fly-to animation. Pure numpy.

Matrices are standard row-major math (``clip = proj @ view @ [x, y, z, 1]``),
right-handed, camera looks down -Z, NDC z in [-1, 1]. The renderer uploads
``view().T.copy()`` / ``proj(aspect).T.copy()`` as column-major uniforms.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_FAR, DEFAULT_MAX_DISTANCE = 20000.0, 5000.0
FLY_SECONDS = 0.8


def _ease(t: float) -> float:
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


class OrbitCamera:
    def __init__(self, target=(0.0, 0.0, 0.0), distance: float = 60.0,
                 yaw: float = 0.6, pitch: float = 0.35, fov_deg: float = 50.0):
        self.target = np.array(target, dtype=np.float64)
        self.distance = float(distance)
        self.yaw = yaw
        self.pitch = pitch
        self.fov_deg = fov_deg
        self.far = DEFAULT_FAR
        self.max_distance = DEFAULT_MAX_DISTANCE
        self._fly: tuple | None = None

    @property
    def flying(self) -> bool:
        return self._fly is not None

    def orbit(self, dx_px: float, dy_px: float) -> None:
        self.yaw -= dx_px * 0.005
        self.pitch = float(np.clip(self.pitch + dy_px * 0.005, -1.5, 1.5))

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, forward) unit vectors in world space."""
        fwd = -self._offset() / self.distance
        right = np.cross(fwd, (0.0, 1.0, 0.0))
        right /= np.linalg.norm(right)
        return right, np.cross(right, fwd), fwd

    def _offset(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return self.distance * np.array(
            [cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])

    def pan(self, dx_px: float, dy_px: float, viewport_h: float) -> None:
        """Drag the scene with the cursor: move target in the view plane."""
        scale = 2 * self.distance * math.tan(math.radians(self.fov_deg) / 2) / viewport_h
        right, up, _ = self._basis()
        self.target = self.target - right * dx_px * scale + up * dy_px * scale

    def zoom(self, wheel_steps: float) -> None:
        self.distance = float(np.clip(self.distance * 0.9**wheel_steps, 2.0, self.max_distance))

    def frame(self, center, radius: float, aspect: float = 1.6) -> None:
        """Snap (no animation) so the sphere fits both axes (tangent-cone), 10% margin.

        Also scales the far plane and zoom limit to the framed radius.
        """
        self._fly = None
        radius = float(radius) if radius > 0 else 1.0
        half_v = math.radians(self.fov_deg) / 2
        theta = min(half_v, math.atan(math.tan(half_v) * aspect))
        self.target = np.array(center, dtype=np.float64)
        self.distance = radius / math.sin(theta) * 1.1
        self.far = max(DEFAULT_FAR, 4 * radius)
        self.max_distance = max(DEFAULT_MAX_DISTANCE, 3 * radius)

    def fly_to(self, target, distance: float, now: float) -> None:
        self._fly = (now, self.target.copy(), self.distance,
                     np.array(target, dtype=np.float64), float(distance))

    def update(self, now: float) -> None:
        if self._fly is None:
            return
        t0, tgt0, d0, tgt1, d1 = self._fly
        t = (now - t0) / FLY_SECONDS
        if t >= 1.0 - 1e-9:
            self.target, self.distance, self._fly = tgt1.copy(), d1, None
            return
        k = _ease(max(t, 0.0))
        self.target = tgt0 + (tgt1 - tgt0) * k
        self.distance = d0 + (d1 - d0) * k

    def eye(self) -> np.ndarray:
        return self.target + self._offset()

    def view(self) -> np.ndarray:
        """4x4 float32 view matrix (row-major math; upload ``.T.copy()``)."""
        right, up, fwd = self._basis()
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3] = right, up, -fwd
        m[:3, 3] = -m[:3, :3] @ self.eye()
        return m.astype(np.float32)

    def proj(self, aspect: float) -> np.ndarray:
        """4x4 float32 perspective matrix (row-major math; upload ``.T.copy()``)."""
        near, far = max(0.01, self.distance / 1000), self.far
        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2)
        m = np.zeros((4, 4))
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = 2 * far * near / (near - far)
        m[3, 2] = -1.0
        return m.astype(np.float32)

    def ray(self, px: float, py: float, viewport_w: float, viewport_h: float):
        """World-space ray (origin, unit direction) through pixel (origin top-left)."""
        right, up, fwd = self._basis()
        th = math.tan(math.radians(self.fov_deg) / 2)
        nx = 2 * px / viewport_w - 1
        ny = 1 - 2 * py / viewport_h
        d = fwd + right * nx * th * viewport_w / viewport_h + up * ny * th
        return self.eye(), d / np.linalg.norm(d)
