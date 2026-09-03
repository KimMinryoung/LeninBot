"""audit.py — append-only security audit log for tool calls.

Dual sink, both non-fatal:
  1. A structured JSON line on the ``security_gateway.audit`` logger (lands in
     journald; queryable via ``ops/logs.py``). Always emitted, synchronously.
  2. A best-effort row in the ``tool_audit_log`` Postgres table, written from a
     single background worker thread so it never blocks the async tool loop and
     never raises into tool execution. Since 2026-09-04 the row is POSTed to
     the LLM proxy's audit sink (audit_sink.py) — the proxy is the only
     process that inserts, so callers need no DB password.

Tool arguments are redacted (secret-looking keys masked) and truncated before
they are recorded.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading

logger = logging.getLogger("security_gateway.audit")

_ARGS_SUMMARY_CAP = 2000
_ERROR_EXCERPT_CAP = 1000
_SECRET_KEY_RE = re.compile(
    r"(token|api[_-]?key|secret|password|passwd|private|credential|bearer|cookie)",
    re.IGNORECASE,
)
_MASK = "«redacted»"

# ── DDL ───────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS tool_audit_log (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    interface     TEXT,
    agent_name    TEXT,
    user_id       TEXT,
    is_owner      BOOLEAN,
    task_id       TEXT,
    session_id    TEXT,
    request_id    TEXT,
    parent_request_id TEXT,
    scope_type    TEXT,
    scope_id      TEXT,
    chat_log_id   BIGINT,
    tool_name     TEXT NOT NULL,
    risk_class    TEXT,
    decision      TEXT NOT NULL,
    enforced      BOOLEAN,
    deny_reason   TEXT,
    args_summary  TEXT,
    result_status TEXT,
    latency_ms    INTEGER,
    error_excerpt TEXT
);
"""
_ALTERS = [
    "ALTER TABLE tool_audit_log ADD COLUMN IF NOT EXISTS session_id TEXT",
    "ALTER TABLE tool_audit_log ADD COLUMN IF NOT EXISTS request_id TEXT",
    "ALTER TABLE tool_audit_log ADD COLUMN IF NOT EXISTS parent_request_id TEXT",
    "ALTER TABLE tool_audit_log ADD COLUMN IF NOT EXISTS scope_type TEXT",
    "ALTER TABLE tool_audit_log ADD COLUMN IF NOT EXISTS scope_id TEXT",
    "ALTER TABLE tool_audit_log ADD COLUMN IF NOT EXISTS chat_log_id BIGINT",
]
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS tool_audit_log_ts_idx ON tool_audit_log (ts DESC)",
    "CREATE INDEX IF NOT EXISTS tool_audit_log_tool_ts_idx ON tool_audit_log (tool_name, ts DESC)",
    "CREATE INDEX IF NOT EXISTS tool_audit_log_decision_ts_idx ON tool_audit_log (decision, ts DESC)",
    "CREATE INDEX IF NOT EXISTS tool_audit_log_interface_ts_idx ON tool_audit_log (interface, ts DESC)",
    "CREATE INDEX IF NOT EXISTS tool_audit_log_request_ts_idx ON tool_audit_log (request_id, ts DESC) WHERE request_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS tool_audit_log_parent_request_ts_idx ON tool_audit_log (parent_request_id, ts DESC) WHERE parent_request_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS tool_audit_log_scope_ts_idx ON tool_audit_log (scope_type, scope_id, ts DESC) WHERE scope_type IS NOT NULL AND scope_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS tool_audit_log_chat_ts_idx ON tool_audit_log (chat_log_id, ts DESC) WHERE chat_log_id IS NOT NULL",
]

_IMMUTABILITY_DDL = """
CREATE OR REPLACE FUNCTION prevent_tool_audit_log_mutation()
RETURNS trigger AS $$
BEGIN
    IF current_setting($setting$leninbot.audit_log_mutation_approved$setting$, true) = $setting$on$setting$ THEN
        IF TG_OP = $setting$DELETE$setting$ THEN
            RETURN OLD;
        ELSIF TG_OP = $setting$TRUNCATE$setting$ THEN
            RETURN NULL;
        END IF;
        RETURN NEW;
    END IF;

    RAISE EXCEPTION $message$tool_audit_log is append-only; set leninbot.audit_log_mutation_approved=on in an explicit admin maintenance transaction to modify it$message$;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tool_audit_log_no_update_delete ON tool_audit_log;
CREATE TRIGGER tool_audit_log_no_update_delete
BEFORE UPDATE OR DELETE ON tool_audit_log
FOR EACH ROW EXECUTE FUNCTION prevent_tool_audit_log_mutation();

DROP TRIGGER IF EXISTS tool_audit_log_no_truncate ON tool_audit_log;
CREATE TRIGGER tool_audit_log_no_truncate
BEFORE TRUNCATE ON tool_audit_log
FOR EACH STATEMENT EXECUTE FUNCTION prevent_tool_audit_log_mutation();
"""


def ensure_tool_audit_log_table() -> None:
    """Create the tool_audit_log table and indexes. Applied via schema_migrations."""
    from db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            for stmt in _ALTERS:
                cur.execute(stmt)
            for stmt in _INDEXES:
                cur.execute(stmt)
            cur.execute(_IMMUTABILITY_DDL)
        conn.commit()


# ── Redaction ─────────────────────────────────────────────────────────
def redact_args(args: dict | None) -> str:
    """Return a redacted, truncated JSON summary of tool arguments."""
    if not args:
        return "{}"
    safe: dict = {}
    for key, value in args.items():
        if _SECRET_KEY_RE.search(str(key)):
            safe[key] = _MASK
        elif isinstance(value, str) and len(value) > 300:
            safe[key] = value[:300] + "…"
        elif isinstance(value, (dict, list)):
            try:
                blob = json.dumps(value, ensure_ascii=False)
            except Exception:
                blob = str(value)
            safe[key] = blob[:300] + ("…" if len(blob) > 300 else "")
        else:
            safe[key] = value
    try:
        out = json.dumps(safe, ensure_ascii=False, default=str)
    except Exception:
        out = str(safe)
    if len(out) > _ARGS_SUMMARY_CAP:
        out = out[:_ARGS_SUMMARY_CAP] + "…"
    return out


# ── Background DB writer ──────────────────────────────────────────────
_DB_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
_worker_started = False
_worker_lock = threading.Lock()


_DRAIN_BATCH = 50
_insert_failed_before = False


def _drain_batch(rows: list[dict]) -> None:
    """Persist a batch: POST to the proxy sink (audit_sink.mode() == "proxy"),
    else insert directly (the proxy itself, or no proxy configured)."""
    global _insert_failed_before
    import audit_sink

    if audit_sink.mode() == "proxy":
        audit_sink.post_rows("tool", rows)
        return
    try:
        audit_sink.insert_rows("tool", rows)
        _insert_failed_before = False
    except Exception as e:
        log = logger.debug if _insert_failed_before else logger.warning
        log("audit DB insert failed (%d row(s) dropped): %s", len(rows), e)
        _insert_failed_before = True


def _worker_loop() -> None:
    while True:
        rows = [_DB_QUEUE.get()]
        while len(rows) < _DRAIN_BATCH:
            try:
                rows.append(_DB_QUEUE.get_nowait())
            except queue.Empty:
                break
        try:
            _drain_batch(rows)
        finally:
            for _ in rows:
                _DB_QUEUE.task_done()


def _ensure_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_loop, name="tool-audit-writer", daemon=True)
        t.start()
        _worker_started = True


# ── Public entry point ────────────────────────────────────────────────
def audit(
    ctx,
    tool_name: str,
    args: dict | None,
    decision,
    *,
    result_status: str,
    latency_ms: int | None = None,
    error_excerpt: str | None = None,
) -> None:
    """Record one tool-call audit event. Never raises into the caller."""
    try:
        from security_gateway.gateway import ALLOW

        enforced = decision.label == "deny"
        deny_reason = None if decision.label == ALLOW else decision.reason
        if error_excerpt and len(error_excerpt) > _ERROR_EXCERPT_CAP:
            error_excerpt = error_excerpt[:_ERROR_EXCERPT_CAP] + "…"

        row = {
            "interface": ctx.interface,
            "agent_name": ctx.agent_name,
            "user_id": str(ctx.user_id) if ctx.user_id is not None else None,
            "is_owner": bool(ctx.is_owner),
            "task_id": str(ctx.task_id) if ctx.task_id is not None else None,
            "session_id": (
                str(getattr(ctx, "session_id"))
                if getattr(ctx, "session_id", None) is not None else None
            ),
            "request_id": (
                str(getattr(ctx, "request_id"))
                if getattr(ctx, "request_id", None) is not None else None
            ),
            "parent_request_id": (
                str(getattr(ctx, "parent_request_id"))
                if getattr(ctx, "parent_request_id", None) is not None else None
            ),
            "scope_type": (
                str(getattr(ctx, "scope_type"))
                if getattr(ctx, "scope_type", None) is not None else None
            ),
            "scope_id": (
                str(getattr(ctx, "scope_id"))
                if getattr(ctx, "scope_id", None) is not None else None
            ),
            "chat_log_id": (
                int(getattr(ctx, "chat_log_id"))
                if getattr(ctx, "chat_log_id", None) is not None else None
            ),
            "tool_name": tool_name,
            "risk_class": decision.risk_class,
            "decision": decision.label,
            "enforced": enforced,
            "deny_reason": deny_reason,
            "args_summary": redact_args(args),
            "result_status": result_status,
            "latency_ms": latency_ms,
            "error_excerpt": error_excerpt,
        }

        # Sink 1: structured log line (always, synchronous, cheap).
        log_fn = logger.warning if decision.label != ALLOW else logger.info
        log_fn(
            "tool_audit %s",
            json.dumps(
                {k: v for k, v in row.items() if k != "args_summary"},
                ensure_ascii=False, default=str,
            ),
        )

        # Sink 2: Postgres, via the background worker (fire-and-forget).
        _ensure_worker()
        try:
            _DB_QUEUE.put_nowait(row)
        except queue.Full:
            logger.warning("audit queue full; dropped DB row for %s", tool_name)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("audit() failed (ignored) for %s: %s", tool_name, e)
