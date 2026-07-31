"""runtime_tools.archival_translation.sources — source-format adapters.

An adapter turns one saved-page format into the flat block list the rest of
the pipeline works on. A block is a dict::

    {"tag": "h3" | "h5" | "p" | "blockquote", "lines": ["…", …]}

``lines`` keeps the internal line structure of a block (a blockquote's
signature lines, an address block) so the translation can be checked
line-for-line and reassembled without guessing.

Only ``militera`` exists today. It is deliberately the sole adapter rather
than a "general HTML" one: each archive marks up documents differently, and a
parser that guesses across formats silently mis-slices instead of failing.
Add the next adapter when there is a second real source to design against.
"""

from __future__ import annotations

import html as htmllib
import re
from typing import Callable

BLOCK_RE = re.compile(r"(?is)<(h3|h5|p|blockquote)\b[^>]*>(.*?)</\1>")
INNER_P_RE = re.compile(r"(?is)<p\b[^>]*>(.*?)</p>")


def _text(fragment: str) -> str:
    s = re.sub(r"(?is)<br\s*/?>", "\n", fragment)
    s = re.sub(r"(?is)<[^>]+>", "", s)
    s = htmllib.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return "\n".join(line.strip() for line in s.split("\n")).strip()


def militera(raw: str) -> list[dict]:
    """Parse a saved militera.lib.ru page.

    Structure: a flat run of h3 (part), h5 (section), p and blockquote inside
    one container div. Nested <p> inside a blockquote become that block's
    lines; the blockquote match consumes them so they are not emitted twice.
    """
    blocks: list[dict] = []
    for m in BLOCK_RE.finditer(raw):
        tag, body = m.group(1).lower(), m.group(2)
        if tag == "blockquote":
            lines = [_text(x) for x in INNER_P_RE.findall(body)] or [_text(body)]
        else:
            lines = [_text(body)]
        lines = [ln for ln in lines if ln]
        if lines:
            blocks.append({"tag": tag, "lines": lines})
    return blocks


ADAPTERS: dict[str, Callable[[str], list[dict]]] = {"militera": militera}
DEFAULT_ADAPTER = "militera"


def get_adapter(name: str | None) -> Callable[[str], list[dict]]:
    key = (name or DEFAULT_ADAPTER).strip()
    if key not in ADAPTERS:
        raise KeyError(f"unknown source format {key!r} (have: {', '.join(sorted(ADAPTERS))})")
    return ADAPTERS[key]
