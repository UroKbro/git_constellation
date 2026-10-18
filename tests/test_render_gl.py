import numpy as np
import pygame

from constellation.render.gl import configure_gl_attributes


def test_standalone_context_is_33_core(gl_ctx):
    assert gl_ctx.version_code >= 330


def test_render_to_f16_fbo(gl_ctx):
    tex = gl_ctx.texture((8, 8), 4, dtype="f2")
    fbo = gl_ctx.framebuffer([tex])
    fbo.use()
    fbo.clear(1.0, 0.5, 0.25, 1.0)
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
