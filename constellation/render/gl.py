"""OpenGL context helpers: GL 3.3 core window and headless contexts."""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
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


def open_window(size: tuple[int, int], title: str, *, hidden: bool = False,
                display: int = 0, vsync: bool | None = None
                ) -> tuple[pygame.Surface, moderngl.Context]:
    """Open a resizable OpenGL window and return it with a blending-enabled context.

    `vsync=False` requests swap interval 0 (macOS may still force vsync)."""
    configure_gl_attributes()
    if vsync is not None:
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, int(vsync))
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    if hidden:
        flags |= pygame.HIDDEN
    surface = pygame.display.set_mode(size, flags, display=display)
    pygame.display.set_caption(title)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    return surface, ctx


_sdl_lib: ctypes.CDLL | None | bool = False


def _sdl() -> ctypes.CDLL | None:
    """The SDL2 library pygame is running on (bundled with the wheel), or None."""
    global _sdl_lib
    if _sdl_lib is False:
        base = os.path.dirname(pygame.__file__)
        found = (glob.glob(os.path.join(base, ".dylibs", "libSDL2*.dylib"))
                 + glob.glob(os.path.join(base, "..", "pygame_ce.libs", "libSDL2*.so*"))
                 + glob.glob(os.path.join(base, "SDL2.dll")))
        name = found[0] if found else ctypes.util.find_library("SDL2")
        try:
            lib = ctypes.CDLL(name) if name else None
            if lib is not None:
                lib.SDL_GL_GetCurrentWindow.restype = ctypes.c_void_p
                lib.SDL_GL_GetDrawableSize.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        except (OSError, AttributeError):
            lib = None
        _sdl_lib = lib
    return _sdl_lib


def drawable_size() -> tuple[int, int] | None:
    """Current GL drawable size in pixels (SDL_GL_GetDrawableSize), or None if unknown.

    moderngl's ctx.screen / detect_framebuffer() keep the size seen at creation
    (they read GL_VIEWPORT), so a resize or display change must ask SDL."""
    lib = _sdl()
    if lib is None:
        return None
    win = lib.SDL_GL_GetCurrentWindow()
    if not win:
        return None
    w, h = ctypes.c_int(), ctypes.c_int()
    lib.SDL_GL_GetDrawableSize(win, ctypes.byref(w), ctypes.byref(h))
    return (w.value, h.value) if w.value > 0 and h.value > 0 else None


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
