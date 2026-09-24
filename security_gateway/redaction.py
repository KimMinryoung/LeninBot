"""Redact structured tool data before it enters durable operational logs."""

from __future__ import annotations

import json
import re
from typing import Any

SECRET_KEY = re.compile(
    r"(token|api[_-]?key|access[_-]?key|secret|password|passwd|private|credential|bearer|cookie|authorization)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:token|api[_-]?key|access[_-]?key|secret|password|passwd|private[_-]?key|credential|bearer|cookie|authorization)\b\s*[:=]\s*)([^\s,;}]+)"
)
_AUTHORIZATION = re.compile(r"(?i)(\bauthorization\s*[:=]\s*)[^\n,;}]+")
_BEARER = re.compile(r"(?i)(\bbearer\s+)[^\s,;)}]+")
MASK = "«redacted»"


def redact_value(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded, JSON-safe copy with secret-labelled fields removed."""
    if depth >= 12:
        return "…"
    if isinstance(value, dict):
        return {
            str(key): MASK if SECRET_KEY.search(str(key)) else redact_value(item, depth=depth + 1)
            for key, item in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item, depth=depth + 1) for item in value[:200]]
    if isinstance(value, str):
        safe = _AUTHORIZATION.sub(lambda match: match.group(1) + MASK, value[:10000])
        safe = _BEARER.sub(lambda match: match.group(1) + MASK, safe)
        return _ASSIGNMENT.sub(lambda match: match.group(1) + MASK, safe)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:300]


def redact_log_text(value: Any) -> str:
    """Mask labelled secrets in a tool result or freeform progress excerpt."""
    if not isinstance(value, str):
        return json.dumps(redact_value(value), ensure_ascii=False, default=str)
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return redact_value(value)
    return json.dumps(redact_value(decoded), ensure_ascii=False, default=str)


def tool_input_summary(args: Any) -> str:
    return json.dumps(redact_value(args), ensure_ascii=False, default=str)
