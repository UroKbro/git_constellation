import numpy as np

from constellation.camera import OrbitCamera
from constellation.render.renderer import Renderer


def _target(ctx, size):
    return ctx.framebuffer([ctx.renderbuffer(size, 4)])


def test_point_at_origin_lights_centre_not_corner(gl_ctx):
    size = (64, 64)
    fbo = _target(gl_ctx, size)
    r = Renderer(gl_ctx, size, target=fbo)
    r.set_points(np.zeros((1, 3), "f4"), np.ones((1, 4), "f4"), np.full(1, 8.0, "f4"))
    cam = OrbitCamera()
    cam.frame((0.0, 0.0, 0.0), 5.0)
    r.render(cam.view(), cam.proj(1.0))
    px = r.read_pixels()
    assert px.shape == (64, 64, 4) and px.dtype == np.uint8
    assert px[32, 32, :3].max() > 200
    assert px[0, 0, :3].max() < 20  # background is near-black navy


def test_no_points_renders_background(gl_ctx):
    size = (16, 16)
    r = Renderer(gl_ctx, size, target=_target(gl_ctx, size))
    r.set_points(np.zeros((0, 3), "f4"), np.zeros((0, 4), "f4"), np.zeros(0, "f4"))
    cam = OrbitCamera()
    r.render(cam.view(), cam.proj(1.0))
    assert r.read_pixels()[..., :3].max() < 40


def test_resize_changes_read_size(gl_ctx):
    r = Renderer(gl_ctx, (16, 16), target=_target(gl_ctx, (32, 24)))
    r.resize((32, 24))
    cam = OrbitCamera()
    r.render(cam.view(), cam.proj(32 / 24))
    assert r.read_pixels().shape == (24, 32, 4)
