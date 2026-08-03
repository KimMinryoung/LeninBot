"""Handler-reported tool outcomes the dispatcher cannot infer on its own.

Two verdicts live here, and they are opposites. `ToolFailure` promotes a
swallowed failure out of `ok`; `ToolRejection` rescues a designed refusal out
of `error`. Both exist because the dispatcher can only see "the handler
returned" or "the handler raised", and neither says whether the tool worked.


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


class ToolRejection(ValueError):
    """A designed refusal: the tool worked and is turning the caller away.

    The mirror image of `ToolFailure`. Handlers that enforce their own domain
    rules — a duplicate candidate, a bio over the length limit, a survey budget
    spent — raise to unwind, and the dispatcher's generic `except` recorded all
    of it as `error`. 4,025 CommuLingo curator rows in 30 days were that: the
    tool doing exactly its job, filed next to genuine breakage.

    Raising this audits the call as `rejected` and hands the model the message
    verbatim, without the "external outcome may be unknown" prefix that a real
    exception earns. The tool_result is still flagged as an error so the model
    changes course rather than repeating the call.

    Raise it only when the refusal happened BEFORE any side effect — that is
    what makes it safe to say nothing about the external world is in doubt. A
    handler that already wrote something and then failed owes the caller a
    plain exception (or `ToolFailure`), not a rejection.

    It subclasses `ValueError` because every site being converted already
    raised one, and callers in between catch it by that name; inheriting keeps
    them working instead of trading an audit fix for a swallowed exception.
    """
