"""Bounded, owner-scoped execution evidence for Telegram chat context."""

from db import query


def recent_execution_events(user_id: int, since) -> list[dict]:
    """Read receipts, never infer executions from saved assistant answers."""
    rows = query(
        "SELECT id, ts, scope_type, scope_id, tool_name, decision, result_status, "
        "left(args_summary, 350) AS args_excerpt, "
        "left(error_excerpt, 200) AS error_excerpt "
        "FROM tool_audit_log WHERE interface = 'telegram' "
        "AND agent_name IS NULL AND user_id = %s AND session_id = %s "
        "AND ts >= %s ORDER BY ts DESC, id DESC LIMIT 24",
        (str(user_id), f"telegram:{user_id}", since),
    )
    return [{
        "source": "tool_audit_log",
        "coverage": "latest 24 recorded calls since retained chat history; excerpts only; audit is best-effort",
        "events": list(reversed(rows)),
    }]
