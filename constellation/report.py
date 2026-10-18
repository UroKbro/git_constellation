"""Self-contained HTML report."""
from html import escape

from constellation.insights import Finding
from constellation.model import RepoHistory, Window

SECTIONS = (("summary", "Summary"), ("hotspot", "Hotspots"),
            ("coupling", "Hidden dependencies"),
            ("bus_factor", "Bus-factor risks"), ("fossil", "Fossils"))

_CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1c1e21;--muted:#5f6670;--card:#f4f5f7;--accent:#3b5bdb}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--muted:#9aa3ad;--card:#1e2127;--accent:#8fa5ff}}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;max-width:48rem;margin:0 auto;padding:1.5rem 1rem}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;color:var(--accent)}
ul{list-style:none;padding:0}li{background:var(--card);border-radius:8px;padding:.6rem .9rem;margin:.5rem 0}
.title{font-weight:600;overflow-wrap:anywhere}.detail{color:var(--muted)}
"""


def render_report(repo_name: str, h: RepoHistory, w: Window, items: list[Finding]) -> str:
    name = escape(repo_name)
    parts = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f"<title>{name} · Git-Constellation</title>", f"<style>{_CSS}</style>",
             f"</head><body><h1>{name}</h1>"]
    for kind, label in SECTIONS:
        rows = [f for f in items if f.kind == kind]
        if not rows:
            continue
        parts.append(f"<h2>{label}</h2><ul>")
        for f in rows:
            parts.append(f'<li><div class="title">{escape(f.title)}</div>'
                         f'<div class="detail">{escape(f.detail)}</div></li>')
        parts.append("</ul>")
    parts.append("</body></html>")
    return "\n".join(parts)
