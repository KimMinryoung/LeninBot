"""Sanitize persisted chat text before display or reuse as model context."""

from __future__ import annotations

import re


CHAT_HISTORY_ARTIFACTS = (
    "[...older response truncated for context hygiene...]",
    "...older response truncated for context hygiene...",
    "[...too long, omitted...]",
    "...too long, omitted...",
    "[...result too long, omitted...]",
    "...result too long, omitted...",
    "[...result too long, omitted]",
    "...result too long, omitted",
    "[... truncated]",
    "... [truncated]",
    "... (context truncated)",
    "\u2026(response truncated at 20000 chars)",
)

_BRACKETED_OMISSION_RE = re.compile(
    r"\[\s*(?:\.\.\.|\u2026)?\s*"
    r"(?:result\s+)?too\s+long,\s+omitted"
    r"\s*(?:\.\.\.|\u2026)?\s*\]",
    re.IGNORECASE,
)


def clean_chat_history_text(text: str) -> str:
    """Remove internal truncation artifacts from persisted chat text."""
    cleaned = str(text or "")
    for marker in CHAT_HISTORY_ARTIFACTS:
        cleaned = cleaned.replace(marker, "")
    cleaned = _BRACKETED_OMISSION_RE.sub("", cleaned)
    return cleaned
