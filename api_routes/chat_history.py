"""Chat log and history API routes."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api_security import require_admin, trusted_proxy_request
from chat_history_sanitize import clean_chat_history_text
from db import execute_returning_rowcount as db_execute_rowcount, query as db_query

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


def _parse_authenticated_user_id(http_req: Request) -> int | None:
    if not trusted_proxy_request(http_req):
        return None
    try:
        value = int((http_req.headers.get("x-authenticated-user-id") or "").strip())
    except ValueError:
        return None
    return value if value > 0 else None


class DeactivateChatMessageRequest(BaseModel):
    part: Literal["user", "assistant"]
    fingerprint: str = Field(default="", max_length=256)
    session_id: str = Field(min_length=1, max_length=128)
    persona: str = Field(min_length=1, max_length=64)


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
            f"""SELECT id AS message_id,
                      CASE WHEN user_query_active THEN user_query ELSE NULL END AS user_query,
                      CASE WHEN bot_answer_active THEN bot_answer ELSE NULL END AS bot_answer,
                      user_query_active, bot_answer_active, created_at
               FROM chat_logs
               WHERE session_id = %s AND fingerprint = ANY(%s){persona_clause}
               ORDER BY created_at ASC
               LIMIT %s""",
            params,
        )
    else:
        params = (fps,) + ((persona,) if persona else ()) + (limit,)
        rows = db_query(
            f"""SELECT id AS message_id,
                      CASE WHEN user_query_active THEN user_query ELSE NULL END AS user_query,
                      CASE WHEN bot_answer_active THEN bot_answer ELSE NULL END AS bot_answer,
                      user_query_active, bot_answer_active, created_at
               FROM chat_logs
               WHERE fingerprint = ANY(%s){persona_clause}
               ORDER BY created_at ASC
               LIMIT %s""",
            params,
        )
    return {"history": _clean_chat_history_rows(rows)}


@router.post("/chat/messages/{message_id}/deactivate")
async def deactivate_chat_message(
    message_id: int,
    payload: DeactivateChatMessageRequest,
    http_req: Request,
):
    """Soft-delete one side of a stored web-chat exchange."""
    if message_id < 1:
        raise HTTPException(status_code=422, detail="Invalid message id")
    if not trusted_proxy_request(http_req):
        raise HTTPException(status_code=403, detail="Trusted frontend proxy required")

    account_user_id = _parse_authenticated_user_id(http_req)
    fingerprints = list({f for f in (_parse_user_fingerprints(http_req) + [payload.fingerprint]) if f})
    if not account_user_id and not fingerprints:
        raise HTTPException(status_code=403, detail="Chat identity required")

    column = "user_query_active" if payload.part == "user" else "bot_answer_active"
    identity_clause = "user_id = %s" if account_user_id else "fingerprint = ANY(%s)"
    identity_value = account_user_id or fingerprints
    changed = db_execute_rowcount(
        f"""UPDATE chat_logs
               SET {column} = false
             WHERE id = %s
               AND session_id = %s
               AND persona = %s
               AND {identity_clause}""",
        (message_id, payload.session_id, payload.persona, identity_value),
    )
    if changed < 1:
        raise HTTPException(status_code=404, detail="Chat message not found")
    return {"deactivated": True, "message_id": message_id, "part": payload.part}


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
              SELECT id, session_id, fingerprint, user_query,
                     user_query_active, bot_answer_active, created_at
                FROM chat_logs
               WHERE fingerprint = ANY(%s){persona_clause}
                 AND (user_query_active OR bot_answer_active)
            ),
            agg AS (
              SELECT session_id,
                     MIN(created_at) AS first_at,
                     MAX(created_at) AS last_at,
                     SUM(user_query_active::int + bot_answer_active::int)::int AS message_count
                FROM scoped
               GROUP BY session_id
            ),
            first_msg AS (
              SELECT DISTINCT ON (session_id)
                     session_id, user_query AS first_query
                FROM scoped
               WHERE user_query_active
               ORDER BY session_id, created_at ASC
            )
            SELECT agg.session_id, agg.first_at, agg.last_at, agg.message_count,
                   first_msg.first_query
              FROM agg
              LEFT JOIN first_msg USING (session_id)
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
