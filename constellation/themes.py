"""Galaxy themes: scene colours, backdrop nebula, UI colours and the soundscape's key.

Every theme gives each language family (palette.FAMILY_NAMES) a distinct colour so the
legend stays readable; lineage colours tint pulses, comets and the timeline."""
from __future__ import annotations

from dataclasses import dataclass

from constellation.palette import FAMILY_COLORS, FAMILY_NAMES, LINEAGE_COLORS, _rgb


def _rgbs(s: str) -> tuple[tuple[float, float, float], ...]:
    return tuple(_rgb(c) for c in s.split())


def _u8(rgb, a: int = 255) -> tuple[int, int, int, int]:
    return (*(round(c * 255) for c in rgb), a)


@dataclass(frozen=True)
class Theme:
    name: str
    background: tuple[float, float, float]
    starfield: tuple[float, float, float]       # tint of the distant point stars
    star: tuple[float, float, float]            # directory star glow
    nebula: tuple[tuple[float, float, float], tuple[float, float, float]]
    nebula_amount: float
    bloom: float
    languages: tuple[tuple[float, float, float], ...]  # one per FAMILY_NAMES entry
    lineage: tuple[tuple[float, float, float], ...]    # mainline first
    accent: tuple[float, float, float]          # UI highlight (playhead, buttons, tags)
    panel: tuple[float, float, float]           # UI panel / bar background
    text: tuple[float, float, float]
    dim: tuple[float, float, float]
    root_hz: float                              # soundscape key
    scale: tuple[int, ...]                      # semitone offsets of the chime scale

    # UI colours as pygame RGBA tuples
    def ui(self, name: str, alpha: int = 255) -> tuple[int, int, int, int]:
        return _u8(getattr(self, name), alpha)


DEEP_SPACE = Theme(
    name="Deep Space",
    background=(0.02, 0.02, 0.05),
    starfield=(0.82, 0.88, 1.0),
    star=(1.0, 0.92, 0.78),
    nebula=(_rgb("#1B2C5A"), _rgb("#3A1F5C")),
    nebula_amount=0.55,
    bloom=1.0,
    languages=tuple(FAMILY_COLORS),
    lineage=tuple(LINEAGE_COLORS),
    accent=_rgb("#FFD666"),
    panel=_rgb("#0B0E18"),
    text=_rgb("#E6E8F0"),
    dim=_rgb("#8C93A8"),
    root_hz=220.0,                  # A minor pentatonic
    scale=(0, 3, 5, 7, 10),
)

CYBERPUNK = Theme(
    name="Cyberpunk Nebula",
    background=(0.03, 0.008, 0.06),
    starfield=(0.95, 0.65, 1.0),
    star=(1.0, 0.5, 0.92),
    nebula=(_rgb("#5A0F6E"), _rgb("#0B4F6C")),
    nebula_amount=0.9,
    bloom=1.35,
    languages=_rgbs("#00E5FF #FFE600 #FF2E88 #7A5CFF #00FFC6 #FF8A3D #FF5C5C #FF3366 "
                    "#B6FF3B #F3E8FF #9B8CC4 #6B4E9B #A88BD6"),
    lineage=_rgbs("#00F0FF #FF2BD6 #B6FF3B #FFD319 #8A5CFF #FF6B35 #00FFA3 #FF3864 #5CE1E6"),
    accent=_rgb("#FF2EC8"),
    panel=_rgb("#12061F"),
    text=_rgb("#F4E9FF"),
    dim=_rgb("#A58BC2"),
    root_hz=146.83,                 # D dorian
    scale=(0, 2, 3, 7, 9),
)

SOLAR_FLARE = Theme(
    name="Solar Flare",
    background=(0.03, 0.012, 0.004),
    starfield=(1.0, 0.82, 0.58),
    star=(1.0, 0.72, 0.38),
    nebula=(_rgb("#6B1D05"), _rgb("#5C3A06")),
    nebula_amount=0.7,
    bloom=1.25,
    languages=_rgbs("#FFB000 #FFE08A #FF5A1F #E8833A #FFD166 #C2410C #F59E0B #DC2626 "
                    "#FACC15 #FFF1D6 #A8876A #7C4A2D #C9A27E"),
    lineage=_rgbs("#FFD27F #FF6B35 #FFF3B0 #E63946 #F4A261 #FFB4A2 #E9C46A #D62828 #F77F00"),
    accent=_rgb("#FF8C28"),
    panel=_rgb("#160A04"),
    text=_rgb("#FFF1E0"),
    dim=_rgb("#C29A78"),
    root_hz=196.0,                  # G major pentatonic
    scale=(0, 2, 4, 7, 9),
)

THEMES = (DEEP_SPACE, CYBERPUNK, SOLAR_FLARE)
assert all(len(t.languages) == len(FAMILY_NAMES) for t in THEMES)


def theme_named(name: str | None) -> Theme:
    return next((t for t in THEMES if t.name == name), DEEP_SPACE)


def next_theme(t: Theme) -> Theme:
    return THEMES[(THEMES.index(t) + 1) % len(THEMES)]
