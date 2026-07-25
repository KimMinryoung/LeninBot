"""Postgres-backed idempotency records for scoped side-effect tool calls."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from psycopg2.extras import RealDictCursor

_RESULT_CAP = 50_000

_DDL = """
CREATE TABLE IF NOT EXISTS tool_idempotency (
    key_hash           TEXT PRIMARY KEY,
    scope              TEXT NOT NULL,
    tool_name          TEXT NOT NULL,
    args_hash          TEXT NOT NULL,
    tool_call_id       TEXT,
    reservation_token TEXT NOT NULL,
    status             TEXT NOT NULL
                       CHECK (status IN ('running', 'succeeded', 'outcome_unknown')),
    result_text        TEXT,
    is_error           BOOLEAN NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS tool_idempotency_scope_created_idx "
    "ON tool_idempotency (scope, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS tool_idempotency_status_updated_idx "
    "ON tool_idempotency (status, updated_at DESC)",
]


@dataclass(frozen=True)
class IdempotencyRecord:
    key_hash: str
    scope: str
    tool_name: str
    status: str
    reservation_token: str
    result_text: str | None = None
    is_error: bool = False
    tool_call_id: str | None = None
    acquired: bool = False


def ensure_tool_idempotency_table() -> None:
    """Create the durable idempotency table via the explicit migration runner."""
    from db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            for stmt in _INDEXES:
                cur.execute(stmt)
        conn.commit()


def scope_for_context(ctx: Any) -> str | None:
    """Return a stable retry scope, avoiding unsafe identity-wide deduplication."""
    interface = str(getattr(ctx, "interface", "unknown") or "unknown")
    task_id = getattr(ctx, "task_id", None)
    if task_id is not None and str(task_id).strip():
        return f"{interface}:task:{str(task_id).strip()}"
    session_id = getattr(ctx, "session_id", None)
    if session_id is not None and str(session_id).strip():
        return f"{interface}:session:{str(session_id).strip()}"
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def build_key(scope: str, tool_name: str, args: dict) -> tuple[str, str]:
    args_blob = _canonical_json(args)
    args_hash = hashlib.sha256(args_blob.encode("utf-8")).hexdigest()
    payload = _canonical_json([scope, tool_name, args_hash])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), args_hash


def _row_to_record(row: dict, *, acquired: bool = False) -> IdempotencyRecord:
    return IdempotencyRecord(
        key_hash=str(row["key_hash"]),
        scope=str(row["scope"]),
        tool_name=str(row["tool_name"]),
        status=str(row["status"]),
        reservation_token=str(row["reservation_token"]),
        result_text=row.get("result_text"),
        is_error=bool(row.get("is_error")),
        tool_call_id=row.get("tool_call_id"),
        acquired=acquired,
    )


def lookup(scope: str, tool_name: str, args: dict) -> IdempotencyRecord | None:
    from db import get_conn

    key_hash, _args_hash = build_key(scope, tool_name, args)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT key_hash, scope, tool_name, status, reservation_token, "
                "result_text, is_error, tool_call_id "
                "FROM tool_idempotency WHERE key_hash = %s",
                (key_hash,),
            )
            row = cur.fetchone()
    return _row_to_record(dict(row)) if row else None


def reserve(
    scope: str,
    tool_name: str,
    args: dict,
    *,
    tool_call_id: str | None,
) -> IdempotencyRecord:
    """Atomically reserve an action or return the existing record."""
    from db import get_conn

    key_hash, args_hash = build_key(scope, tool_name, args)
    token = uuid.uuid4().hex
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO tool_idempotency
                    (key_hash, scope, tool_name, args_hash, tool_call_id,
                     reservation_token, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'running')
                ON CONFLICT (key_hash) DO NOTHING
                RETURNING key_hash, scope, tool_name, status, reservation_token,
                          result_text, is_error, tool_call_id
                """,
                (key_hash, scope, tool_name, args_hash, tool_call_id, token),
            )
            row = cur.fetchone()
            inserted = row is not None
            if row is None:
                cur.execute(
                    "SELECT key_hash, scope, tool_name, status, reservation_token, "
                    "result_text, is_error, tool_call_id "
                    "FROM tool_idempotency WHERE key_hash = %s",
                    (key_hash,),
                )
                row = cur.fetchone()
        conn.commit()
    if row is None:
        raise RuntimeError("idempotency reservation disappeared")
    return _row_to_record(dict(row), acquired=inserted)


def mark_succeeded(record: IdempotencyRecord, result: str) -> None:
    from db import execute

    execute(
        """
        UPDATE tool_idempotency
        SET status = 'succeeded', result_text = %s, is_error = FALSE, updated_at = now()
        WHERE key_hash = %s AND reservation_token = %s AND status = 'running'
        """,
        (result[:_RESULT_CAP], record.key_hash, record.reservation_token),
    )


def mark_outcome_unknown(record: IdempotencyRecord, error: str) -> None:
    from db import execute

    execute(
        """
        UPDATE tool_idempotency
        SET status = 'outcome_unknown', result_text = %s, is_error = TRUE, updated_at = now()
        WHERE key_hash = %s AND reservation_token = %s AND status = 'running'
        """,
        (error[:_RESULT_CAP], record.key_hash, record.reservation_token),
    )


__all__ = [
    "IdempotencyRecord",
    "build_key",
    "ensure_tool_idempotency_table",
    "lookup",
    "mark_outcome_unknown",
    "mark_succeeded",
    "reserve",
    "scope_for_context",
]
