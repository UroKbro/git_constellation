import numpy as np

from constellation.camera import OrbitCamera
from constellation.render.renderer import (Renderer, nearest_star, star_falloff,
                                          star_intensity)
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


def _draw(r, cam, planets=None, stars=None, pulses=None, comets=None, now=0.0, overlay=None,
          **kw):
    w, h = r.size
    r.draw(cam.view(), cam.proj(w / h), planets or _planets(()), stars or _stars(),
           pulses or _pulses(), comets or _comets(), now, overlay, **kw)
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
    bright = _draw(r, cam, _planets(rgb=(1.0, 0.6, 0.3), emissive=3.0))  # add-flash max
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


def _radius(alive):
    return 0.6 + 0.12 * np.sqrt(np.asarray(alive, "f8"))


def test_star_intensity_monotonic_capped_and_floor_above_median():
    alive = np.array([0, 1, 3, 10, 100, 5000])
    inten, floor = star_intensity(_radius(alive))
    assert np.all(np.diff(inten) > 0)
    assert inten.max() == np.float32(4.0) and inten.min() >= 0
    assert inten[1] < 0.5  # a single-file leaf dir is faint
    assert list(floor > 0) == list(alive > np.median(alive))
    assert star_intensity(np.zeros(0))[0].shape == (0,)


def test_star_falloff_smooth_and_zero_at_edge():
    d = np.linspace(0, 1, 201)
    f = star_falloff(d)
    assert f[0] == 1.0 and abs(f[-1]) < 1e-12
    assert np.all(np.diff(f) < 0)
    assert np.abs(np.diff(f)).max() < 0.05  # no hard rings
    assert star_falloff(1.5) == 0.0


def test_star_billboard_edge_reaches_background(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    empty = _draw(r, cam)
    px = _draw(r, cam, stars=_stars([(0, 0, 0)], radius=1.0))  # half-size ~31.6 px
    assert px[64, 64, :3].min() > 230
    for x in (64 + 33, 64 - 33):
        assert abs(_lum(px, 64, x) - _lum(empty, 64, x)) <= 6


def test_planet_lit_from_behind_keeps_ambient(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    base = np.array([0x35, 0x72, 0xA5]) / 255  # Python blue
    behind = -cam.eye()[None, :]
    px = _draw(r, cam, _planets(rgb=tuple(base), emissive=0.0), planet_light=behind)
    want = 255 * (1 - np.exp(-0.45 * base))
    assert np.all(px[64, 64, :3] >= 0.95 * want - 1)
    assert px[64, 64, 2] > px[64, 64, 0] + 20  # clearly blue


def test_planet_hdr_capped(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    px = _draw(r, _cam(), _planets(emissive=50.0))
    assert px[64, 64, :3].max() < 245


def test_nearest_star_and_light_cache():
    stars = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], "f4")
    pos = np.array([[1, 1, 0], [9, 0, 1], [0, 8, 0], [6, 0, 0]], "f4")
    assert list(nearest_star(pos, stars)) == [0, 1, 2, 1]
    assert list(nearest_star(pos, stars[:0])) == [-1] * 4


def test_planet_light_param_and_nearest_fallback_upload_once(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    p, s, pu, c = _planets(), _stars([(30, 0, 0)]), _pulses(), _comets()
    _draw(r, cam, p, s, pu, c)
    n = r.uploads
    _draw(r, cam, p, s, pu, c)
    assert r.uploads == n
    light = np.array([[0, 0, 30]], "f4")
    _draw(r, cam, p, s, pu, c, planet_light=light)
    assert r.uploads == n + 1
    _draw(r, cam, p, s, pu, c, planet_light=light)
    assert r.uploads == n + 1


def test_overlay_upload_split_from_draw(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    rgba = np.zeros((128, 128, 4), np.uint8)
    rgba[:, :64] = (0, 255, 0, 255)
    data = rgba.tobytes()
    r.set_overlay(data, (128, 128))
    assert r.overlay_uploads == 1
    px = _draw(r, cam)  # overlay=None draws the last upload
    assert tuple(px[5, 5, :3]) == (0, 255, 0)
    _draw(r, cam, overlay=(data, (128, 128)))  # unchanged overlay: no re-upload
    assert r.overlay_uploads == 1
    r.set_overlay(None)
    assert _draw(r, cam)[5, 5, 1] < 40


def test_set_target_redirects_output(gl_ctx):
    a, b = _target(gl_ctx), _target(gl_ctx)
    r = Renderer(gl_ctx, SIZE, target=a)
    r.set_target(b)
    _draw(r, _cam(), _planets())
    assert r.read_pixels()[64, 64, :3].max() > 150
    a.use()
    raw = np.frombuffer(a.read(components=4), np.uint8)
    assert raw.max() == 0  # never drawn into


def test_resize_rereads_screen_when_no_target(gl_ctx):
    r = Renderer(gl_ctx, SIZE)
    sentinel = _target(gl_ctx)
    type(gl_ctx)  # ctx.screen is read-only; check the re-read via set_target(None)
    r.set_target(sentinel)
    r.set_target(None)
    assert r.target is gl_ctx.screen
    r.resize((32, 32))
    assert r.target is gl_ctx.screen


def _to_px(cam, p, size=SIZE):
    clip = cam.proj(size[0] / size[1]) @ cam.view() @ np.array([*p, 1.0])
    ndc = clip[:3] / clip[3]
    return int(round((1 - ndc[1]) / 2 * size[1])), int(round((ndc[0] + 1) / 2 * size[0]))


def test_comet_tail_points_away_from_bounds_centre_and_braids(gl_ctx):
    r = Renderer(gl_ctx, SIZE, target=_target(gl_ctx))
    cam = _cam()
    centre = np.array([-5.0, 0.0, 0.0])
    one = np.ones((1, 4), "f4")
    comet = CometBatch(np.zeros((1, 3), "f4"), one * [1, 0, 0, 1], one * [0, 0, 1, 1],
                       np.zeros(1, np.int32), np.zeros(1, "f4"), np.full(1, 1.5, "f4"))
    px = _draw(r, cam, comets=comet, now=0.05, bounds=(centre, 10.0)).astype(int)
    empty = _draw(r, cam).astype(int)
    diff = px[..., :3] - empty[..., :3]
    ty, tx = _to_px(cam, (2.0, 0.0, 0.0))     # along the tail (away from centre)
    my, mx = _to_px(cam, (-2.0, 0.0, 0.0))    # mirrored, towards the centre
    near = lambda y, x: diff[y - 3:y + 4, x - 3:x + 4].sum()
    assert near(ty, tx) > 50 * max(near(my, mx), 1)  # only bloom leaks behind the head
    (hy, hx), (ey, ex) = _to_px(cam, (0, 0, 0)), _to_px(cam, (3.0, 0, 0))
    ys, xs = sorted((hy, ey)), sorted((hx, ex))
    tail = diff[ys[0] - 6:ys[1] + 7, xs[0] - 6:xs[1] + 7]
    assert ((tail[..., 0] > tail[..., 2] + 20)).any()  # red strand
    assert ((tail[..., 2] > tail[..., 0] + 20)).any()  # blue strand
