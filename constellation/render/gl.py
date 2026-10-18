"""OpenGL context helpers: GL 3.3 core window and headless contexts."""
from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import moderngl  # noqa: E402
import pygame  # noqa: E402


def configure_gl_attributes() -> None:
    """Request a GL 3.3 core forward-compatible context (call before set_mode)."""
    set_attr = pygame.display.gl_set_attribute
    set_attr(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    set_attr(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    set_attr(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    set_attr(pygame.GL_CONTEXT_FLAGS, pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)


def open_window(size: tuple[int, int], title: str) -> tuple[pygame.Surface, moderngl.Context]:
    """Open a resizable OpenGL window and return it with a blending-enabled context."""
    configure_gl_attributes()
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    surface = pygame.display.set_mode(size, flags)
    pygame.display.set_caption(title)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    return surface, ctx


def standalone_context() -> moderngl.Context | None:
    """Headless GL 3.3 context, or None when unavailable."""
    try:
        return moderngl.create_standalone_context(require=330)
    except Exception:
        return None


def drawable_scale(ctx: moderngl.Context, window_size: tuple[int, int]) -> float:
    """Drawable pixels per window point (2.0 on Retina)."""
    if window_size[0] == 0:
        return 1.0
    return ctx.screen.size[0] / window_size[0]
