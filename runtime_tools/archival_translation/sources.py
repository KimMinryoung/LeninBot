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

CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁіІїЇєЄ]")
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


WIKISOURCE_BODY_RE = re.compile(
    r'<div class="mw-content-ltr mw-parser-output"[^>]*>(.*?)'
    r'(?=<!--\s*NewPP|<div class="printfooter")', re.S)
WIKISOURCE_BLOCK_RE = re.compile(
    r"(?is)<(center|h2|h3|h4|p|dd|li|table)\b[^>]*>(.*?)</\1>")
TABLE_ROW_RE = re.compile(r"(?is)<tr\b[^>]*>(.*?)</tr>")
TABLE_CELL_RE = re.compile(r"(?is)<t[hd]\b[^>]*>(.*?)</t[hd]>")


def wikisource(raw: str) -> list[dict]:
    """Parse a saved ru.wikisource.org document page.

    Wikisource wraps the transcription in mw-parser-output together with its
    own chrome — a header box, a source note, a "Наверх" link, the licence
    notice. None of it is stripped here: the spec's block range and its
    startsWith/endsWith guards decide what belongs to the document, the same
    way they do for militera. A parser that guessed at the boundary would
    quietly shift when the wiki page is re-saved.

    Tables come back as ``{"tag": "table", "rows": [[cell, …], …]}``. Order
    00447 carries its regional quotas in one, and a table flattened into prose
    is a table whose numbers can go missing without anything noticing.
    """
    body = WIKISOURCE_BODY_RE.search(raw)
    if not body:
        return []
    blocks: list[dict] = []
    for m in WIKISOURCE_BLOCK_RE.finditer(body.group(1)):
        tag, inner = m.group(1).lower(), m.group(2)
        if tag == "table":
            rows = [
                [_text(c) for c in TABLE_CELL_RE.findall(row)]
                for row in TABLE_ROW_RE.findall(inner)
            ]
            rows = [r for r in rows if any(c for c in r)]
            if rows:
                # Only the cells carrying words go to the model, deduplicated
                # and in order of first appearance; the assembler puts the
                # translations back into the grid. A quota table sent through
                # as 133 lines of prose can come back a row short or a digit
                # different, and nothing downstream would notice — here the
                # numbers never leave the code.
                vocab, seen = [], set()
                for row in rows:
                    for cell in row:
                        if cell and CYRILLIC_RE.search(cell) and cell not in seen:
                            seen.add(cell)
                            vocab.append(cell)
                blocks.append({"tag": "table", "rows": rows, "lines": vocab})
            continue
        lines = [ln for ln in _text(inner).split("\n") if ln]
        if lines:
            blocks.append({"tag": tag, "lines": lines})
    return blocks


ADAPTERS: dict[str, Callable[[str], list[dict]]] = {
    "militera": militera,
    "wikisource": wikisource,
}
DEFAULT_ADAPTER = "militera"


def get_adapter(name: str | None) -> Callable[[str], list[dict]]:
    key = (name or DEFAULT_ADAPTER).strip()
    if key not in ADAPTERS:
        raise KeyError(f"unknown source format {key!r} (have: {', '.join(sorted(ADAPTERS))})")
    return ADAPTERS[key]
