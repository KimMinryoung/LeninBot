"""agent_worker_tasks persistence (worker/schema.sql)."""
from __future__ import annotations

import hashlib
import json

from psycopg2.extras import RealDictCursor

from db import get_conn

LEASE = "interval '15 minutes'"
MAX_ATTEMPTS = 2


class IdempotencyConflict(ValueError):
    pass


def _hash(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _public(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "taskId": str(row["id"]), "status": row["status"], "attempts": row["attempts"],
        "result": row["result"], "sources": row["sources"], "usage": row["usage"],
        "rejections": row["rejections"], "error": row["error"],
        "createdAt": row["created_at"].isoformat(),
        "startedAt": row["started_at"].isoformat() if row["started_at"] else None,
        "finishedAt": row["finished_at"].isoformat() if row["finished_at"] else None,
    }


def submit(client: str, idempotency_key: str, request: dict) -> dict:
    """Queue a task; the same key with the same request returns the existing task."""
    digest = _hash(request)
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """INSERT INTO agent_worker_tasks (client, idempotency_key, request, request_hash)
               VALUES (%s, %s, %s::jsonb, %s)
               ON CONFLICT (client, idempotency_key) DO NOTHING RETURNING *""",
            (client, idempotency_key, json.dumps(request, ensure_ascii=False), digest))
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT * FROM agent_worker_tasks WHERE client = %s AND idempotency_key = %s",
                        (client, idempotency_key))
            row = cur.fetchone()
            if row["request_hash"] != digest:
                raise IdempotencyConflict("idempotencyKey reused with a different request")
    return _public(row)


def get(client: str, task_id: str) -> dict | None:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM agent_worker_tasks WHERE id = %s AND client = %s", (int(task_id), client))
        return _public(cur.fetchone())


def cancel(client: str, task_id: str) -> dict | None:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """UPDATE agent_worker_tasks SET status = 'cancelled', finished_at = now(), updated_at = now(),
                      lease_until = NULL
                WHERE id = %s AND client = %s AND status IN ('queued', 'running') RETURNING *""",
            (int(task_id), client))
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT * FROM agent_worker_tasks WHERE id = %s AND client = %s", (int(task_id), client))
            row = cur.fetchone()
    return _public(row)


def claim() -> dict | None:
    """Lease the oldest runnable task: queued, or running with an expired lease
    (the worker restarted mid-task) and attempts left."""
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""UPDATE agent_worker_tasks SET status = 'failed', error = 'worker stopped during every attempt',
                       finished_at = now(), updated_at = now(), lease_until = NULL
                 WHERE status = 'running' AND lease_until < now() AND attempts >= {MAX_ATTEMPTS}""")
        cur.execute(
            f"""UPDATE agent_worker_tasks t SET status = 'running', attempts = attempts + 1,
                       lease_until = now() + {LEASE}, started_at = COALESCE(started_at, now()), updated_at = now()
                 WHERE id = (SELECT id FROM agent_worker_tasks
                              WHERE status = 'queued' OR (status = 'running' AND lease_until < now())
                              ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED)
                RETURNING *""")
        return cur.fetchone()


def finish(task_id: int, *, status: str, result=None, sources=(), usage=None, rejections=(), error=None) -> bool:
    """Record the outcome unless the task was cancelled meanwhile."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE agent_worker_tasks SET status = %s, result = %s::jsonb, sources = %s::jsonb,
                      usage = %s::jsonb, rejections = %s::jsonb, error = %s, lease_until = NULL,
                      finished_at = now(), updated_at = now()
                WHERE id = %s AND status = 'running'""",
            (status, json.dumps(result, ensure_ascii=False) if result is not None else None,
             json.dumps(list(sources), ensure_ascii=False), json.dumps(usage or {}, ensure_ascii=False),
             json.dumps(list(rejections), ensure_ascii=False), error, task_id))
        return cur.rowcount == 1


def is_cancelled(task_id: int) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT status FROM agent_worker_tasks WHERE id = %s", (task_id,))
        row = cur.fetchone()
        return not row or row[0] == "cancelled"
