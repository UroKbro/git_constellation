from constellation.palette import language_color, lineage_color


def _hex(s: str) -> tuple[float, float, float]:
    return tuple(int(s[i:i + 2], 16) / 255 for i in (1, 3, 5))


def test_language_colors():
    assert language_color("src/a.py") == _hex("#3572A5")
    assert language_color("ui/b.tsx") == _hex("#F1E05A")
    assert language_color("README.md") == _hex("#C9D1D9")
    assert language_color("Makefile") == _hex("#9DA5B4")
    assert language_color("x/STYLE.SCSS") == _hex("#E34C26")
    assert language_color("logo.png") == _hex("#6E5494")
    assert language_color(".gitignore") == _hex("#9DA5B4")


def test_lineage_colors():
    assert lineage_color(0) == _hex("#7FD4FF")
    assert lineage_color(1) == _hex("#FF9E64")
    assert lineage_color(8) == _hex("#82AAFF")
    assert lineage_color(9) == lineage_color(1)
