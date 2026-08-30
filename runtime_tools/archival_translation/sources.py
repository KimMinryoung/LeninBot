"""runtime_tools.archival_translation.sources — source-format adapters.

An adapter turns one saved-page format into the flat block list the rest of
the pipeline works on. A block is a dict::

    {"tag": "h3" | "h5" | "p" | "blockquote", "lines": ["…", …]}

``lines`` keeps the internal line structure of a block (a blockquote's
signature lines, an address block) so the translation can be checked
line-for-line and reassembled without guessing.

There is one adapter per archive (``ADAPTERS``), deliberately not a
"general HTML" one: each archive marks up documents differently, and a parser
that guesses across formats silently mis-slices instead of failing. Add the
next adapter when there is a real source to design against.
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
            # 칸 하나가 곧 번역 한 줄이다. 서명란처럼 <br>로 여러 줄을 담은 칸이
            # 줄바꿈을 그대로 지니면 모델이 그것을 각각 다른 줄로 보고, 조립기가
            # 칸과 번역을 짝지을 때 한 칸씩 밀린다 — 독소조약 서명란에서 독일
            # 측 서명이 통째로 사라졌다. 칸 안의 줄바꿈은 공백으로 접는다.
            rows = [
                [" ".join(_text(c).split("\n")) for c in TABLE_CELL_RE.findall(row)]
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


STALINISM_BODY_RE = re.compile(
    r'<div class="com-content-article__body"[^>]*>(.*?)(?=<div class="[^"]*(?:pagenavigation|'
    r'com-content-article__footer)|</main|<footer)', re.S)
SCRIPT_LEAK_RE = re.compile(r"(?is)(?:var\s|function\s*\(|document\.|\{|\}|;\s*$)")


def stalinism(raw: str) -> list[dict]:
    """Parse a saved stalinism.ru document page.

    The document sits in ``com-content-article__body`` surrounded by the
    site's own furniture; a stray inline script also lands inside a <p>, so
    paragraphs that read as code are dropped. What is *not* dropped here is
    the modern compiler's apparatus — the publication note, the archival
    reference — because deciding that is the spec's job, not the parser's.
    """
    body = STALINISM_BODY_RE.search(raw)
    scope = body.group(1) if body else raw
    blocks: list[dict] = []
    for m in re.finditer(r"(?is)<p\b[^>]*>(.*?)</p>", scope):
        text = _text(m.group(1))
        if not text or not CYRILLIC_RE.search(text):
            continue
        if SCRIPT_LEAK_RE.search(text) and not text.endswith((".", "!", "?", "»")):
            continue
        blocks.append({"tag": "p", "lines": [ln for ln in text.split("\n") if ln]})
    return blocks


LIBRU_BODY_RE = re.compile(r"(?is)<pre>(.*?)</pre>")


def libru(raw: str) -> list[dict]:
    """Parse a saved lib.ru text page.

    lib.ru serves a plain-text file inside one <pre>, hard-wrapped at about
    seventy columns, with paragraphs separated by blank lines. The wrapping is
    an artefact of the format, not of the document, so the lines of a paragraph
    are joined back into one; a blank line is the only paragraph boundary there
    is. The site's own furniture (the download menu, the OCR credit, the rule
    of equals signs) is left in place — the spec's block range decides what
    belongs to the document, as with every other adapter.

    Save the page as UTF-8: lib.ru serves CP1251 and load_blocks reads UTF-8.
    """
    body = LIBRU_BODY_RE.search(raw)
    scope = body.group(1) if body else raw
    blocks: list[dict] = []

    def add_paragraphs(text: str) -> None:
        # 70칸 줄바꿈은 포맷의 산물이라 이어 붙이지만, <br>은 문서 자신의
        # 줄바꿈이라 살려야 한다. 둘을 섞어 지우면 편지 수신인란처럼 짧은 행이
        # 잇달아 오는 곳에서 낱말이 서로 붙는다("스탈린 동지에게.사본:").
        text = re.sub(r"(?is)<br\s*/?>", "\x00", text)
        text = re.sub(r"(?is)<[^>]+>", "", text)
        for chunk in re.split(r"\n[ \t]*\n", htmllib.unescape(text)):
            joined = " ".join(part.strip() for part in chunk.split("\n"))
            lines = [re.sub(r"[ \t]+", " ", part).strip()
                     for part in joined.split("\x00")]
            lines = [line for line in lines if line]
            if lines:
                blocks.append({"tag": "p", "lines": lines})

    # 절 제목은 <pre> 안에서도 <h2>로 표시된다. 태그를 먼저 전부 지우면 제목이
    # 앞 문단 끝에 붙어 버려, 범위를 자를 자리도 사라지고 마지막 문단에 엉뚱한
    # 낱말이 남는다("… 모두 일어섬.) 주석").
    pos = 0
    for m in re.finditer(r"(?is)<h([1-6])>(.*?)</h\1>", scope):
        add_paragraphs(scope[pos:m.start()])
        title = re.sub(r"[ \t]+", " ", re.sub(r"(?is)<[^>]+>", "", m.group(2))).strip()
        if title:
            blocks.append({"tag": "h" + m.group(1), "lines": [title]})
        pos = m.end()
    add_paragraphs(scope[pos:])
    return blocks


LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
MARXISTS_HEAD_RE = re.compile(r"(?is)<h([1-6])\b[^>]*>(.*?)</h\1>")
# The page is HTML 3.2: <P> opens a paragraph and nothing closes it, so a
# paragraph runs to the next block-level tag. Splitting on the openers is the
# only way to find its end.
MARXISTS_SPLIT_RE = re.compile(r"(?is)<(?:p|h[1-6]|hr|table|blockquote)\b[^>]*>")
MARXISTS_BOLD_ONLY_RE = re.compile(r"(?is)\A\s*<b>(.*?)</b>\s*\Z")


def marxists(raw: str) -> list[dict]:
    """Parse a saved marxists.org page (Russian section).

    The archive keeps these files as they were typed in the 1990s: HTML 3.2
    with unclosed ``<P>``, one ``<H3>`` carrying author and title, and section
    headings written as a centred paragraph whose whole content is bold. So
    paragraphs are cut at the next block-level opener rather than at a closing
    tag, and a paragraph that is nothing but ``<b>…</b>`` is emitted as ``h4``
    — the spec's tagMap decides what that becomes in the fragment.

    The site's own furniture (the "Оглавление" links top and bottom, the
    provenance line, the issue of Бюллетень оппозиции at the end) is left in
    place: as with every other adapter, the block range is the spec's call.

    Save the page as UTF-8. marxists.org serves the Russian section in
    CP1251 and load_blocks reads UTF-8.
    """
    body = re.search(r"(?is)<body\b[^>]*>(.*?)(?:</body>|\Z)", raw)
    scope = body.group(1) if body else raw

    blocks: list[dict] = []

    def emit(tag: str, fragment: str) -> None:
        # The file is hard-wrapped at about seventy columns, so a raw newline
        # is an artefact of how it was typed, not a line of the document —
        # left in, _text turns every wrapped line into its own line and the
        # reader gets one word per row. Only <br> is the document's own break.
        marked = re.sub(r"(?is)<br\s*/?>", "\x00", fragment).replace("\n", " ")
        lines = [ln for ln in (_text(part) for part in marked.split("\x00")) if ln]
        # Not a Cyrillic test: the same archive holds the Spanish section, and
        # José Díaz's speeches would be dropped whole by one. A block only has
        # to carry a letter — the spacer paragraphs this markup is full of
        # (&nbsp; alone) are what needs dropping.
        if not lines or not LETTER_RE.search(" ".join(lines)):
            return
        blocks.append({"tag": tag, "lines": lines})

    # Headings first: they are properly closed, so their spans are known, and
    # cutting them out keeps the paragraph splitter from swallowing them.
    pos = 0
    for m in MARXISTS_HEAD_RE.finditer(scope):
        _split_paragraphs(scope[pos:m.start()], emit)
        emit("h" + m.group(1), m.group(2))
        pos = m.end()
    _split_paragraphs(scope[pos:], emit)
    return blocks


def _split_paragraphs(scope: str, emit: Callable[[str, str], None]) -> None:
    starts = [m.end() for m in MARXISTS_SPLIT_RE.finditer(scope)]
    bounds = [m.start() for m in MARXISTS_SPLIT_RE.finditer(scope)] + [len(scope)]
    for i, start in enumerate(starts):
        fragment = scope[start:bounds[i + 1]]
        bold = MARXISTS_BOLD_ONLY_RE.match(fragment.strip())
        emit("h4" if bold else "p", bold.group(1) if bold else fragment)


ADAPTERS: dict[str, Callable[[str], list[dict]]] = {
    "militera": militera,
    "wikisource": wikisource,
    "stalinism": stalinism,
    "libru": libru,
    "marxists": marxists,
}
DEFAULT_ADAPTER = "militera"


def get_adapter(name: str | None) -> Callable[[str], list[dict]]:
    key = (name or DEFAULT_ADAPTER).strip()
    if key not in ADAPTERS:
        raise KeyError(f"unknown source format {key!r} (have: {', '.join(sorted(ADAPTERS))})")
    return ADAPTERS[key]
