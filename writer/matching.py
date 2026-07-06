"""Whitespace/quote-tolerant passage matching.

Models quote manuscript text from memory with drifted whitespace and
straight-vs-curly quotes; these helpers locate the real saved span anyway.
"""

from __future__ import annotations

import re

# Straight and curly single/double quotes (plus the corner brackets used in
# Korean/Japanese typography) drift between what the model remembers and what
# the manuscript stores; treat them as interchangeable when locating a passage.
_QUOTE_CHARS = "\"'‘’“”「」『』"
_QUOTE_CLASS = f"[{_QUOTE_CHARS}]"


def normalized_pattern(find: str) -> re.Pattern | None:
    """Whitespace- and quote-insensitive pattern for locating a passage the
    model quoted from memory. Returns None for degenerate input."""
    tokens = find.split()
    if not tokens or len(tokens) > 500:
        return None
    parts = []
    for token in tokens:
        escaped = re.sub(_QUOTE_CLASS, _QUOTE_CLASS, re.escape(token))
        parts.append(escaped)
    try:
        return re.compile(r"\s+".join(parts))
    except re.error:
        return None


def find_normalized_matches(body: str, find: str, max_matches: int = 3) -> list[tuple[int, int]]:
    pattern = normalized_pattern(find)
    if pattern is None:
        return []
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(body):
        spans.append((match.start(), match.end()))
        if len(spans) >= max_matches:
            break
    return spans
