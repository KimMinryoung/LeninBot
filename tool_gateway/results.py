"""Handler-reported tool failures.

Most tool handlers catch their own exceptions and return an explanatory string
so the model gets a useful message instead of a generic stack trace. That made
the failure invisible to the audit log: the dispatcher only saw a handler that
returned normally, so `tool_audit_log.result_status` said `ok` for calls that
did nothing — eight Tavily quota rejections on 2026-07-27 were all logged as
successes, and nothing in the audit trail distinguished them from real hits.

`ToolFailure` keeps the useful message while making the failure legible:

    return ToolFailure(f"Vector search failed: {exc}")

It is a `str` subclass, so every existing consumer (truncation, provenance,
serialization into a tool_result block) treats it exactly like the string it
replaces. Only the dispatcher looks at the type, and it audits the call as
`error` and flags the tool_result as an error for the model.

Use it for a call that could not do its job — an external system failed, a
dependency was unavailable, an exception was swallowed. Do NOT use it for a
well-formed call with a negative answer ("no documents found", "person not in
the registry"): those are successful lookups, and marking them errors would
bury the real failures.
"""

from __future__ import annotations


class ToolFailure(str):
    """A tool result string that the gateway must audit as a failure."""

    __slots__ = ()


def is_failure(result: object) -> bool:
    """True when a handler flagged its own result as a failure."""
    return isinstance(result, ToolFailure)
