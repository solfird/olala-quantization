"""Check that every local link and image of a Markdown file exists.

    python tools/check_links.py README.md

Looks at ``[text](target)``, ``![alt](target)`` and the ``src=`` and ``srcset=`` attributes of inline HTML. Web links (http, https), mail links and
pure anchors are counted but not fetched: the check needs no network. Exit status 1 when a local target is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

MD = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML = re.compile(r"""(?:src|srcset|href)\s*=\s*["']([^"']+)["']""")


def targets(text: str) -> list[str]:
    """Every link target of a Markdown text, outside fenced code blocks."""
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        out += MD.findall(line) + HTML.findall(line)
    return out


def main(argv: list[str]) -> int:
    """Check every Markdown file named in ``argv``."""
    bad = 0
    for name in argv or ["README.md"]:
        path = Path(name).resolve()
        local = web = 0
        for t in targets(path.read_text()):
            u = urlparse(t)
            if u.scheme in ("http", "https", "mailto"):
                web += 1
                continue
            if not u.path:  # a pure anchor
                continue
            local += 1
            target = (path.parent / unquote(u.path)).resolve()
            if not target.exists():
                print(f"MISSING  {name}: {t}")
                bad += 1
        print(f"{name}: {local} local links and images checked, {web} web links not fetched, {bad} missing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
