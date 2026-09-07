"""Structural guards and protected spans shared by site translation entrypoints."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import re
from urllib.parse import unquote

from markdown_it import MarkdownIt


class HTMLSignature(HTMLParser):
    VOID = set("area base br col embed hr img input link meta param source track wbr".split())
    TRANSLATABLE = {"alt", "title", "aria-label", "placeholder"}
    PROTECTED_TEXT = {"code", "pre", "script", "style"}

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.events = []
        self.visible = []
        self.stack = []
        self.invalid = False
        self.feed(text)
        self.close()
        self.invalid |= bool(self.stack)

    def handle_starttag(self, tag, attrs):
        self.events.append(("open", tag, tuple(sorted(
            (k, v) for k, v in attrs if k not in self.TRANSLATABLE))))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        self.events.append(("close", tag))
        if not self.stack or self.stack[-1] != tag:
            self.invalid = True
        else:
            self.stack.pop()

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if any(tag in self.PROTECTED_TEXT for tag in self.stack):
            self.events.append(("protected_text", data))
        else:
            self.visible.append(data)


def html_problems(source: str, target: str) -> list[str]:
    a, b = HTMLSignature(source), HTMLSignature(target)
    problems = []
    if a.events != b.events:
        problems.append("HTML structure or protected attributes differ from source")
    if not a.invalid and b.invalid:
        problems.append("translated HTML is not balanced")
    return problems


def markdown_parser():
    return MarkdownIt("commonmark", {"html": True}).enable("table")


def markdown_signature(text: str):
    env = {}
    tokens = markdown_parser().parse(text, env)
    structure, links, code, html = [], [], [], []
    for token in tokens:
        if token.type != "inline":
            structure.append((token.type, token.tag, token.nesting))
        if token.type in ("fence", "code_block"):
            code.append((token.type, token.info, token.content))
        if token.type == "html_block":
            html.append(token.content)
        for child in token.children or []:
            if child.type == "link_open":
                links.append(("link", child.attrGet("href"), child.attrGet("title")))
            elif child.type == "image":
                links.append(("image", child.attrGet("src"), child.attrGet("title")))
            elif child.type == "code_inline":
                code.append(("inline", "", child.content))
            elif child.type == "html_inline":
                html.append(child.content)
    # Unused reference definitions and footnote IDs must survive as well.
    refs = Counter((k, v["href"], v.get("title", ""))
                   for k, v in env.get("references", {}).items())
    footnotes = Counter(re.findall(r"\[\^([^\]\n]+)\]", text))
    return structure, Counter(links), code, "".join(html), refs, footnotes


def markdown_problems(source: str, target: str) -> list[str]:
    a, b = markdown_signature(source), markdown_signature(target)
    problems = []
    for index, label in ((0, "block structure"), (1, "link destinations"),
                         (2, "code"), (4, "reference definitions"), (5, "footnotes")):
        if a[index] != b[index]:
            problems.append(f"Markdown {label} differs from source")
    problems.extend(html_problems(a[3], b[3]))
    return problems


def protect_markdown(text: str) -> tuple[str, dict[str, str]]:
    """Mask code and destinations before generation. Placeholders are deterministic."""
    protected = {}
    prefix = "TRKEEP" + sha256(text.encode()).hexdigest()[:12].upper()

    def keep(value):
        marker = f"{prefix}X{len(protected)}Z"
        protected[marker] = value
        return marker

    # Block token maps give exact source spans, including fenced code contents.
    lines = text.splitlines(keepends=True)
    spans = []
    for token in markdown_parser().parse(text):
        if token.type in ("fence", "code_block") and token.map:
            spans.append(token.map)
    for start, end in reversed(spans):
        # The span already ends with its own newline; adding another one
        # left a blank line after every fence in the restored document.
        lines[start:end] = [keep("".join(lines[start:end]))]
    masked = "".join(lines)
    masked = re.sub(r"<(code|pre|script|style)\b[^>]*>.*?</\1\s*>",
                    lambda m: keep(m[0]), masked, flags=re.DOTALL | re.IGNORECASE)
    masked = re.sub(r"(`+)(.+?)\1", lambda m: keep(m[0]), masked, flags=re.DOTALL)
    # Protect raw HTML tags/attributes, but leave their visible prose translatable.
    masked = re.sub(r"<!--.*?-->|</?[A-Za-z][^>]*>", lambda m: keep(m[0]), masked,
                    flags=re.DOTALL)
    signature = markdown_signature(text)
    destinations = {x[1] for x in signature[1]} | {x[1] for x in signature[4]}
    destinations |= {unquote(d) for d in destinations}
    for destination in sorted(destinations, key=len, reverse=True):
        if destination and destination in masked:
            masked = masked.replace(destination, keep(destination))
    masked = re.sub(r"\[\^[^\]\n]+\]", lambda m: keep(m[0]), masked)
    masked = re.sub(r"(?<=\])\[[^\]\n]*\]|(?m:^[ ]{0,3}\[[^\]\n]+\]:)",
                    lambda m: keep(m[0]), masked)
    return masked, protected


def restore_markdown(text: str, protected: dict[str, str]) -> str:
    for marker, original in protected.items():
        # A destination can occur several times; use a unique marker per value.
        if marker not in text:
            raise ValueError("translation omitted protected code or link placeholder")
        text = text.replace(marker, original)
    return text


def markdown_chunks(text: str, max_chars: int = 8000) -> list[str]:
    """Split at top-level block boundaries; lists/tables/fences stay intact."""
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines(keepends=True)
    boundaries = {0, len(lines)}
    depth = 0
    for token in markdown_parser().parse(text):
        if depth == 0 and token.map:
            boundaries.add(token.map[0])
        depth += token.nesting
    points = sorted(boundaries)
    chunks, current = [], ""
    for start, end in zip(points, points[1:]):
        block = "".join(lines[start:end])
        if current and len(current) + len(block) > max_chars:
            chunks.append(current)
            current = ""
        current += block
    if current:
        chunks.append(current)
    return chunks


def semantic_review(source: str, target: str) -> list[dict]:
    """Review hints, never automatic corrections or claims of semantic equivalence."""
    numbers = lambda s: Counter(re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|퍼센트)?", s))
    issues = []
    if numbers(source) != numbers(target):
        issues.append({"kind": "numbers_dates_amounts", "source": dict(numbers(source)),
                       "target": dict(numbers(target))})
    if re.search(r"않|없|아니|금지|경우|한하여|조건|not\b|unless\b|если|не\s|不得|如果", source):
        issues.append({"kind": "negation_conditions", "note": "원문 대조 검토 필요"})
    return issues


def split_prose(text: str, limit: int) -> list[str]:
    """Prefer sentence/word boundaries; hard cap exceptionally unbroken OCR text."""
    parts = []
    while len(text) > limit:
        window = text[:limit]
        ends = [m.end() for m in re.finditer(r"[.!?。！？]\s+|\n", window)]
        cut = ends[-1] if ends else window.rfind(" ") + 1
        if cut < limit // 3:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:]
    if text.strip():
        parts.append(text.strip())
    return parts
