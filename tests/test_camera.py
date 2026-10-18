import numpy as np
import pytest

from constellation.camera import OrbitCamera


def _ndc(cam, p, aspect):
    c = cam.proj(aspect).astype(np.float64) @ cam.view().astype(np.float64) @ np.append(p, 1.0)
    return c[:3] / c[3]


@pytest.mark.parametrize("aspect", [1.0, 1.6])
def test_frame_contains_sphere(aspect):
    cam = OrbitCamera()
    c, r = np.array([5.0, -3.0, 8.0]), 30.0
    cam.frame(c, r)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    pts = c + r * d / np.linalg.norm(d, axis=1)[:, None]
    for p in pts:
        n = _ndc(cam, p, aspect)
        assert np.all(np.abs(n[:2]) <= 1.0)


def test_centre_ray_is_forward():
    cam = OrbitCamera(yaw=0.9, pitch=-0.4)
    o, d = cam.ray(400, 300, 800, 600)
    fwd = np.asarray(cam.target) - cam.eye()
    fwd /= np.linalg.norm(fwd)
    assert np.allclose(d, fwd, atol=1e-6)
    assert np.allclose(o, cam.eye(), atol=1e-4)
    assert abs(np.linalg.norm(d) - 1) < 1e-6


def test_ray_round_trip():
    cam = OrbitCamera(yaw=0.3, pitch=0.5)
    w, h = 800, 500
    p = np.array([3.0, 4.0, -5.0])
    n = _ndc(cam, p, w / h)
    px, py = (n[0] + 1) / 2 * w, (1 - n[1]) / 2 * h
    o, d = cam.ray(px, py, w, h)
    t = (p - o) @ d
    assert np.allclose(o + t * d, p, atol=1e-4)


def test_zoom_round_trip_and_clamp():
    cam = OrbitCamera(distance=60.0)
    cam.zoom(10)
    cam.zoom(-10)
    assert cam.distance == pytest.approx(60.0)
    cam.zoom(1000)
    assert cam.distance == 2
    cam.zoom(-1000)
    assert cam.distance == 5000


def test_pitch_clamp():
    cam = OrbitCamera()
    cam.orbit(0, 100000)
    assert cam.pitch == 1.5
    cam.orbit(0, -1000000)
    assert cam.pitch == -1.5


def test_pan_moves_in_view_plane():
    cam = OrbitCamera()
    t0 = np.array(cam.target, float)
    fwd = t0 - cam.eye()
    cam.pan(50, 20, 600)
    delta = np.asarray(cam.target) - t0
    assert abs(delta @ fwd) < 1e-6
    assert np.linalg.norm(delta) > 0


def test_fly_to():
    cam = OrbitCamera(target=(0, 0, 0), distance=50.0)
    cam.fly_to((10.0, 20.0, 30.0), 10.0, now=100.0)
    assert cam.flying
    cam.update(100.4)
    assert np.allclose(cam.target, (5, 10, 15), atol=1e-6)
    assert cam.distance == pytest.approx(30.0, abs=1e-6)
    assert cam.flying
    cam.update(100.8)
    assert np.array_equal(np.asarray(cam.target), [10.0, 20.0, 30.0])
    assert cam.distance == 10.0
    assert not cam.flying


def test_matrices_dtype_shape():
    cam = OrbitCamera()
    assert cam.view().shape == (4, 4) and cam.view().dtype == np.float32
    assert cam.proj(1.5).dtype == np.float32
