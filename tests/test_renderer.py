import numpy as np

from constellation.camera import OrbitCamera
from constellation.render.renderer import Renderer
from constellation.scene import CometBatch, PlanetBatch, PulseBatch, StarBatch

SIZE = (128, 128)


def _target(ctx, size=SIZE):
    return ctx.framebuffer([ctx.renderbuffer(size, 4)])


def _cam():
    cam = OrbitCamera()
    cam.frame((0.0, 0.0, 0.0), 5.0, aspect=1.0)
    return cam


def _planets(pos=((0, 0, 0),), radius=1.0, rgb=(1.0, 1.0, 1.0), emissive=1.0):
    pos = np.asarray(pos, "f4").reshape(-1, 3)
    n = len(pos)
    color = np.tile(np.array([*rgb, emissive], "f4"), (n, 1))
    return PlanetBatch(pos, np.full(n, radius, "f4"), color, np.arange(n, dtype=np.int32))


def _stars(pos=(), radius=1.0):
    pos = np.asarray(pos, "f4").reshape(-1, 3)
    n = len(pos)
    return StarBatch(pos, np.full(n, radius, "f4"), np.tile(np.array([1, .92, .78], "f4"), (n, 1)))


def _pulses(start=(), end=(), born=0.0, life=1.0):
    s = np.asarray(start, "f4").reshape(-1, 3)
    n = len(s)
    return PulseBatch(s, np.asarray(end, "f4").reshape(-1, 3),
                      np.tile(np.array([0.4, 0.8, 1.0, 1.0], "f4"), (n, 1)),
                      np.full(n, born, "f4"), np.full(n, life, "f4"))


def _comets(pos=(), born=0.0, life=1.5):
    p = np.asarray(pos, "f4").reshape(-1, 3)
    n = len(p)
    one = np.ones((n, 4), "f4")
    return CometBatch(p, one * [1, .5, .2, 1], one * [.2, .5, 1, 1],
                      np.arange(n, dtype=np.int32), np.full(n, born, "f4"), np.full(n, life, "f4"))


def _draw(r, cam, planets=None, stars=None, pulses=None, comets=None, now=0.0, **kw):
    w, h = r.size
    r.draw(cam.view(), cam.proj(w / h), planets or _planets(()), stars or _stars(),
           pulses or _pulses(), comets or _comets(), now, None, **kw)
    return r.read_pixels()


def _lum(px, y, x):
    return int(px[y, x, :3].astype(int).sum())


def test_planet_lights_centre_not_far_pixels_or_corners(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    px = _draw(r, _cam(), _planets())
    assert px.shape == (128, 128, 4) and px.dtype == np.uint8
    assert px[64, 64, :3].max() > 150
    assert _lum(px, 64, 64) > _lum(px, 64, 104)
    for y, x in ((0, 0), (0, 127), (127, 0), (127, 127)):
        assert px[y, x, :3].max() < 20


def test_bright_planet_blooms_but_emissive_zero_does_not(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    bg = _lum(_draw(r, cam), 64, 64 + 14)
    # disc radius ~10.5 px at this framing; 14 px out is outside the disc
    bright = _draw(r, cam, _planets(rgb=(1.0, 0.6, 0.3), emissive=6.0))
    dull = _draw(r, cam, _planets(rgb=(1.0, 0.6, 0.3), emissive=0.0))
    assert _lum(bright, 64, 64 + 14) > bg + 10
    assert _lum(dull, 64, 64 + 14) <= bg + 2
    assert _lum(dull, 64, 64) > 100  # the dull planet itself is still visible


def test_resize_changes_read_size(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    r.resize((64, 48))
    assert _draw(r, _cam()).shape == (48, 64, 4)


def test_bloom_scale_rebuilds_chain(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    assert r.bloom_scale == 0.5 and r.bloom_size == (64, 64)
    r.set_bloom_scale(0.25)
    assert r.bloom_scale == 0.25 and r.bloom_size == (32, 32)
    _draw(r, _cam(), _planets())


def test_empty_batches_render_background(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    px = _draw(r, _cam())
    assert np.median(px[..., :3]) < 20


def test_star_is_brighter_than_planet(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    star = _draw(r, cam, stars=_stars([(0, 0, 0)], radius=0.6))
    planet = _draw(r, cam, _planets(radius=0.6, rgb=(0.9, 0.9, 0.9)))
    # halo around the star lights a ring the planet leaves dark
    assert _lum(star, 64, 64 + 14) > _lum(planet, 64, 64 + 14) + 20
    assert star[64, 64, :3].min() > 230


def test_far_planet_gets_tiny_size_floor(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    cam.distance = 5000.0  # projected radius << 1 px
    px = _draw(r, cam, _planets(radius=0.25, emissive=0.0))
    assert px[62:67, 62:67, :3].max() > 60


def test_fog_dims_far_side(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    eye = cam.eye()
    clear = _draw(r, cam, _planets(emissive=0.0))
    # bounds put the planet at the far edge of the galaxy -> full 35% dimming
    foggy = _draw(r, cam, _planets(emissive=0.0), eye=eye, bounds=(eye * 0.5, 0.3 * np.linalg.norm(eye)))
    a, b = _lum(clear, 64, 64), _lum(foggy, 64, 64)
    assert b < 0.8 * a and b > 0.4 * a


def test_pulse_midflight_and_comet_render(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    px = _draw(r, cam, pulses=_pulses([(-2, 0, 0)], [(2, 0, 0)], born=0.0, life=1.0), now=0.5)
    assert px[60:69, 60:69, :3].max() > 100
    empty = _draw(r, cam)
    dead = _draw(r, cam, pulses=_pulses([(-2, 0, 0)], [(2, 0, 0)], born=0.0, life=1.0), now=2.0)
    assert np.array_equal(dead, empty)  # starfield only
    px = _draw(r, cam, comets=_comets([(0, 0, 0)]), now=0.1)
    assert px[58:71, 58:71, :3].max() > 150


def test_instance_buffers_reused_for_same_batch(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    p, s, pu, c = _planets(), _stars(), _pulses(), _comets()
    _draw(r, cam, p, s, pu, c)
    n = r.uploads
    _draw(r, cam, p, s, pu, c)
    assert r.uploads == n
    _draw(r, cam, _planets(), s, pu, c)
    assert r.uploads == n + 1


def test_overlay_is_composited(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    rgba = np.zeros((128, 128, 4), np.uint8)
    rgba[:, :64] = (255, 0, 0, 255)  # left half opaque red
    r.draw(cam.view(), cam.proj(1.0), _planets(()), _stars(), _pulses(), _comets(), 0.0,
           (rgba.tobytes(), (128, 128)))
    px = r.read_pixels()
    assert tuple(px[10, 10, :3]) == (255, 0, 0)
    assert px[10, 120, 0] < 20
