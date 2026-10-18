"""Colours: language by file extension, lineage for branches."""


def _rgb(s: str) -> tuple[float, float, float]:
    return tuple(int(s[i:i + 2], 16) / 255 for i in (1, 3, 5))


_FAMILIES = [
    ("#3572A5", "py pyi pyx pyw ipynb"),
    ("#F1E05A", "js jsx mjs cjs ts tsx mts cts"),
    ("#E34C26", "html htm xhtml css scss sass less vue svelte"),
    ("#6E7FD2", "c h cc cpp cxx c++ hpp hh hxx h++ m mm"),
    ("#00ADD8", "go"),
    ("#DEA584", "rs"),
    ("#B07219", "java kt kts scala sc"),
    ("#CC342D", "rb erb rake gemspec"),
    ("#89E051", "sh bash zsh fish ksh"),
    ("#C9D1D9", "md markdown rst txt adoc"),
    ("#8B949E", "json jsonc yaml yml toml ini cfg conf xml csv tsv lock"),
    ("#6E5494", "png jpg jpeg gif bmp ico webp tif tiff psd svg pdf zip gz tgz bz2 xz "
                "7z tar jar war so dylib dll exe o a bin dat woff woff2 ttf otf eot "
                "mp3 mp4 mov wav ogg flac avi webm class pyc"),
]
_BY_EXT = {ext: _rgb(col) for col, exts in _FAMILIES for ext in exts.split()}
OTHER = _rgb("#9DA5B4")

_LINEAGE0 = _rgb("#7FD4FF")
_LINEAGE = [_rgb(s) for s in
            "#FF9E64 #C3E88D #F07178 #C792EA #FFCB6B #89DDFF #F78C6C #82AAFF".split()]


def extension(path: str) -> str:
    """Lower-case extension of the basename ('' for none or a bare dotfile)."""
    stem, dot, ext = path.rpartition("/")[2].lstrip(".").rpartition(".")
    return ext.lower() if dot and stem else ""


def language_color(path: str) -> tuple[float, float, float]:
    return _BY_EXT.get(extension(path), OTHER)


def lineage_color(k: int) -> tuple[float, float, float]:
    """0 is the mainline colour; k >= 1 cycles through 8 hues."""
    return _LINEAGE0 if k <= 0 else _LINEAGE[(k - 1) % len(_LINEAGE)]
