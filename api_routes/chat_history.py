"""Chat log and history API routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query, Request

from api_security import require_admin, trusted_proxy_request
from chat_history_sanitize import clean_chat_history_text
from db import query as db_query

router = APIRouter()


def _clean_chat_history_rows(rows: list[dict]) -> list[dict]:
    cleaned = []
    for row in rows:
        item = dict(row)
        item["user_query"] = clean_chat_history_text(item.get("user_query", ""))
        item["bot_answer"] = clean_chat_history_text(item.get("bot_answer", ""))
        cleaned.append(item)
    return cleaned


def _parse_user_fingerprints(http_req: Request) -> list[str]:
    if not trusted_proxy_request(http_req):
        return []
    raw = http_req.headers.get("x-user-fingerprints", "")
    return [f.strip()[:256] for f in raw.split(",") if f.strip()][:20]


@router.get("/logs", dependencies=[Depends(require_admin)])
async def get_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Fetch chat logs (admin view, ordered by most recent first)."""
    rows = db_query(
        "SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limit, offset),
    )
    return {"logs": rows, "count": len(rows)}


@router.get("/history")
async def get_history(
    http_req: Request,
    fingerprint: str | None = Query(default=None, description="Browser fingerprint (anonymous visitors)"),
    session_id: str | None = Query(default=None, description="Restrict to a single conversation session"),
    persona: str | None = Query(default=None, description="Restrict to a single chat persona"),
    limit: int = Query(default=50, ge=1, le=200),
):
    fps = list({f for f in (_parse_user_fingerprints(http_req) + [fingerprint or ""]) if f})
    if not fps:
        return {"history": []}

    persona_clause = " AND persona = %s" if persona else ""
    if session_id:
        params = (session_id, fps) + ((persona,) if persona else ()) + (limit,)
        rows = db_query(
            f"""SELECT user_query, bot_answer, created_at
               FROM chat_logs
               WHERE session_id = %s AND fingerprint = ANY(%s){persona_clause}
               ORDER BY created_at ASC
               LIMIT %s""",
            params,
        )
    else:
        params = (fps,) + ((persona,) if persona else ()) + (limit,)
        rows = db_query(
            f"""SELECT user_query, bot_answer, created_at
               FROM chat_logs
               WHERE fingerprint = ANY(%s){persona_clause}
               ORDER BY created_at ASC
               LIMIT %s""",
            params,
        )
    return {"history": _clean_chat_history_rows(rows)}


@router.get("/sessions")
async def list_sessions(
    http_req: Request,
    fingerprint: str | None = Query(default=None, description="Anonymous browser fingerprint"),
    persona: str | None = Query(default=None, description="Restrict to a single chat persona"),
    limit: int = Query(default=50, ge=1, le=200),
):
    fps = list({f for f in (_parse_user_fingerprints(http_req) + [fingerprint or ""]) if f})
    if not fps:
        return {"sessions": []}

    persona_clause = " AND persona = %s" if persona else ""
    params = (fps,) + ((persona,) if persona else ()) + (limit,)
    rows = db_query(
        f"""WITH scoped AS (
              SELECT id, session_id, fingerprint, user_query, created_at
                FROM chat_logs
               WHERE fingerprint = ANY(%s){persona_clause}
            ),
            agg AS (
              SELECT session_id,
                     MIN(created_at) AS first_at,
                     MAX(created_at) AS last_at,
                     COUNT(*)::int   AS message_count
                FROM scoped
               GROUP BY session_id
            ),
            first_msg AS (
              SELECT DISTINCT ON (session_id)
                     session_id, user_query AS first_query
                FROM scoped
               ORDER BY session_id, created_at ASC
            )
            SELECT agg.session_id, agg.first_at, agg.last_at, agg.message_count,
                   first_msg.first_query
              FROM agg
              JOIN first_msg USING (session_id)
             ORDER BY agg.last_at DESC
             LIMIT %s""",
        params,
    )
    for row in rows:
        query = row.get("first_query") or ""
        row["first_query"] = query[:120]
    return {"sessions": rows}


@router.delete("/session/{session_id}", dependencies=[Depends(require_admin)])
async def clear_session(session_id: str):
    """Clear chat history for a session."""
    result = await asyncio.to_thread(
        db_query,
        "DELETE FROM chat_logs WHERE session_id = %s RETURNING id",
        (session_id,),
    )
    return {"session_id": session_id, "cleared": len(result) if result else 0}
