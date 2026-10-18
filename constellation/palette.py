"""Colours: language by file extension, lineage for branches.

Files are grouped into language families; a theme (themes.py) supplies one colour per
family. The functions here return the default (Deep Space) colours."""


def _rgb(s: str) -> tuple[float, float, float]:
    return tuple(int(s[i:i + 2], 16) / 255 for i in (1, 3, 5))


_FAMILIES = [
    ("Python", "#3572A5", "py pyi pyx pyw ipynb"),
    ("JS / TS", "#F1E05A", "js jsx mjs cjs ts tsx mts cts"),
    ("Web", "#E34C26", "html htm xhtml css scss sass less vue svelte"),
    ("C / C++", "#6E7FD2", "c h cc cpp cxx c++ hpp hh hxx h++ m mm"),
    ("Go", "#00ADD8", "go"),
    ("Rust", "#DEA584", "rs"),
    ("JVM", "#B07219", "java kt kts scala sc"),
    ("Ruby", "#CC342D", "rb erb rake gemspec"),
    ("Shell", "#89E051", "sh bash zsh fish ksh"),
    ("Docs", "#C9D1D9", "md markdown rst txt adoc"),
    ("Config", "#8B949E", "json jsonc yaml yml toml ini cfg conf xml csv tsv lock"),
    ("Binary", "#6E5494", "png jpg jpeg gif bmp ico webp tif tiff psd svg pdf zip gz tgz bz2 xz "
                          "7z tar jar war so dylib dll exe o a bin dat woff woff2 ttf otf eot "
                          "mp3 mp4 mov wav ogg flac avi webm class pyc"),
]
FAMILY_NAMES = [name for name, _, _ in _FAMILIES] + ["Other"]
OTHER_FAMILY = len(_FAMILIES)
FAMILY_COLORS = [_rgb(col) for _, col, _ in _FAMILIES] + [_rgb("#9DA5B4")]
_FAMILY_BY_EXT = {ext: i for i, (_, _, exts) in enumerate(_FAMILIES) for ext in exts.split()}
OTHER = FAMILY_COLORS[OTHER_FAMILY]

LINEAGE_COLORS = [_rgb(s) for s in
                  "#7FD4FF #FF9E64 #C3E88D #F07178 #C792EA #FFCB6B #89DDFF #F78C6C #82AAFF".split()]


def extension(path: str) -> str:
    """Lower-case extension of the basename ('' for none or a bare dotfile)."""
    stem, dot, ext = path.rpartition("/")[2].lstrip(".").rpartition(".")
    return ext.lower() if dot and stem else ""


def language_family(path: str) -> int:
    """Index into FAMILY_NAMES (OTHER_FAMILY when the extension is unknown)."""
    return _FAMILY_BY_EXT.get(extension(path), OTHER_FAMILY)


def language_color(path: str) -> tuple[float, float, float]:
    return FAMILY_COLORS[language_family(path)]


def lineage_from(table, k: int) -> tuple[float, float, float]:
    """table[0] is the mainline colour; k >= 1 cycles through the rest."""
    return table[0] if k <= 0 else table[1 + (k - 1) % (len(table) - 1)]


def lineage_color(k: int) -> tuple[float, float, float]:
    return lineage_from(LINEAGE_COLORS, k)
