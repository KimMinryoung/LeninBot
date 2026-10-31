"""Cross-turn tool-result evidence for the Telegram orchestrator.

The provider loops rebuild native tool protocol only inside one call, and the
persisted chat history is text-only (see ``llm.claude_loop._normalize_initial_messages``).
``tool_audit_log`` keeps argument excerpts but no results, so on the next turn
the orchestrator knew *that* it had searched or read something but not *what*
it found, apart from whatever its own prose reply repeated.

This module closes that gap without replaying native tool calls:

* ``capture_turn`` installs the ``tool_gateway.observations`` result observer
  for one orchestrator call and collects (tool, args, result) per call.
* ``persist_turn`` stores redacted excerpts in ``telegram_tool_evidence``.
* ``recent_tool_evidence`` returns them as a ``prior_tool_results`` context
  record for the turns inside the retained raw-history window.

Records are reference data with provenance (``llm.execution_context``), not
receipts of new actions. Writes and reads degrade to a logged warning, so a
missing table (migration not yet applied) never breaks a chat turn.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

RESULT_EXCERPT_CHARS = 1200
ARGS_EXCERPT_CHARS = 300
MAX_CALLS_PER_TURN = 30
LOAD_LIMIT = 12
LOAD_TOTAL_CHARS = 12000
RETENTION_DAYS = 14

# Calls whose result says nothing about the world (bookkeeping/meta tools).
_SKIP_TOOLS = frozenset({"list_agent_tools"})

DDL = """
CREATE TABLE IF NOT EXISTS telegram_tool_evidence (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    scope_type TEXT,
    scope_id TEXT,
    tool_name TEXT NOT NULL,
    args_excerpt TEXT NOT NULL DEFAULT '',
    result_excerpt TEXT NOT NULL DEFAULT '',
    result_chars INTEGER NOT NULL DEFAULT 0,
    is_error BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS telegram_tool_evidence_user_ts
    ON telegram_tool_evidence (user_id, created_at DESC);
"""


def ensure_table() -> None:
    from db import execute
    execute(DDL)


def owner_user_id(session_id: str | None) -> int | None:
    """Telegram orchestrator sessions are ``telegram:<chat id>``."""
    if not session_id or not str(session_id).startswith("telegram:"):
        return None
    try:
        return int(str(session_id).split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def _excerpt(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f" … [excerpt: {limit} of {len(text)} chars]"


@contextmanager
def capture_turn(enabled: bool):
    """Collect tool results for one call; disable observation when not enabled.

    Setting ``None`` for non-orchestrator calls keeps a nested delegated agent
    run (``run_agent`` inside an orchestrator turn) out of the orchestrator's
    evidence.
    """
    from tool_gateway.observations import tool_result_observer

    entries: list[dict] = []

    def observe(name, args, result, is_error):
        if name in _SKIP_TOOLS or len(entries) >= MAX_CALLS_PER_TURN:
            return
        entries.append({"name": name, "args": args, "result": result, "is_error": is_error})

    token = tool_result_observer.set(observe if enabled else None)
    try:
        yield entries
    finally:
        tool_result_observer.reset(token)


def persist_turn(user_id: int, entries: list[dict], *, scope_type=None, scope_id=None) -> None:
    if not entries:
        return
    from db import get_conn
    from security_gateway.redaction import redact_log_text, tool_input_summary

    rows = []
    for entry in entries:
        result = str(entry.get("result") or "")
        rows.append((
            user_id, scope_type, str(scope_id) if scope_id is not None else None,
            entry["name"],
            _excerpt(tool_input_summary(entry.get("args") or {}), ARGS_EXCERPT_CHARS),
            _excerpt(redact_log_text(result), RESULT_EXCERPT_CHARS),
            len(result), bool(entry.get("is_error")),
        ))
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO telegram_tool_evidence (user_id, scope_type, scope_id, "
                    "tool_name, args_excerpt, result_excerpt, result_chars, is_error) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    rows,
                )
                cur.execute(
                    "DELETE FROM telegram_tool_evidence WHERE user_id = %s "
                    f"AND created_at < NOW() - INTERVAL '{RETENTION_DAYS} days'",
                    (user_id,),
                )
    except Exception as exc:
        logger.warning("telegram tool evidence not persisted: %s", exc)


def recent_tool_evidence(user_id: int, since) -> list[dict]:
    """Context records for tool results recorded since ``since``."""
    from db import query
    from llm.execution_context import context_record

    try:
        rows = query(
            "SELECT id, created_at, scope_type, tool_name, args_excerpt, result_excerpt, "
            "result_chars, is_error FROM telegram_tool_evidence "
            "WHERE user_id = %s AND created_at >= %s ORDER BY id DESC LIMIT %s",
            (user_id, since, LOAD_LIMIT),
        )
    except Exception as exc:
        logger.warning("telegram tool evidence unavailable: %s", exc)
        return [context_record(
            "prior_tool_results", "telegram_tool_evidence", [],
            scope=f"telegram:{user_id}", coverage="unavailable",
        )]
    if not rows:
        return []
    kept, total = [], 0
    for row in rows:  # newest first; keep the newest within the char budget
        size = len(row.get("result_excerpt") or "") + len(row.get("args_excerpt") or "")
        if kept and total + size > LOAD_TOTAL_CHARS:
            break
        kept.append(row)
        total += size
    kept.reverse()
    return [context_record(
        "prior_tool_results", "telegram_tool_evidence",
        [{
            "id": r["id"], "at": str(r["created_at"]), "tool": r["tool_name"],
            "args": r["args_excerpt"], "status": "error" if r["is_error"] else "ok",
            "result_chars": r["result_chars"], "result_excerpt": r["result_excerpt"],
        } for r in kept],
        scope=f"telegram:{user_id}", temporal_scope="earlier turns",
        coverage=(
            f"redacted result excerpts (≤{RESULT_EXCERPT_CHARS} chars) of the newest "
            f"{len(kept)} orchestrator tool calls in the retained history window; "
            "repeat a read-only call for its full or current output"
        ),
        reference="historical results; not new actions and not proof of current state",
    )]
