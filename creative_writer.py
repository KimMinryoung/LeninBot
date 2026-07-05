"""Personal fiction-writing workspace backed by Claude Fable 5.

This module is intentionally separate from the Cyber-Lenin web chat/provider
configuration. It reuses the shared Anthropic Messages loop for API calls while
storing its own project sessions in writer-specific tables.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import anthropic

from db import execute as db_execute
from db import query as db_query
from db import query_one as db_query_one
from claude_loop import chat_with_tools
from secrets_loader import get_secret

logger = logging.getLogger(__name__)

WRITER_MODEL = "claude-fable-5"
WRITER_MODEL_DISPLAY = "Claude Fable 5"
WRITER_INPUT_PRICE_PER_MTOK = 10.0
WRITER_OUTPUT_PRICE_PER_MTOK = 50.0
_MAX_CONTEXT_MESSAGES = 80

_writer_client: anthropic.AsyncAnthropic | None = None


def _client() -> anthropic.AsyncAnthropic:
    global _writer_client
    if _writer_client is None:
        api_key = get_secret("WRITER_ANTHROPIC_API_KEY", "") or get_secret("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("WRITER_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY is required")
        _writer_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _writer_client


def ensure_writer_tables() -> None:
    """Create the writer tables. Called only by explicit schema migrations."""
    db_execute(
        """CREATE TABLE IF NOT EXISTS writer_projects (
               id BIGSERIAL PRIMARY KEY,
               title TEXT NOT NULL,
               premise TEXT NOT NULL DEFAULT '',
               style_notes TEXT NOT NULL DEFAULT '',
               status TEXT NOT NULL DEFAULT 'active',
               created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
           )"""
    )
    db_execute(
        """CREATE TABLE IF NOT EXISTS writer_messages (
               id BIGSERIAL PRIMARY KEY,
               project_id BIGINT NOT NULL REFERENCES writer_projects(id) ON DELETE CASCADE,
               role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
               content TEXT NOT NULL,
               request_kind TEXT NOT NULL DEFAULT '',
               model TEXT NOT NULL DEFAULT '',
               stop_reason TEXT NOT NULL DEFAULT '',
               usage JSONB NOT NULL DEFAULT '{}'::jsonb,
               cost_usd NUMERIC(12, 6),
               created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
           )"""
    )
    db_execute(
        "CREATE INDEX IF NOT EXISTS writer_messages_project_created_idx "
        "ON writer_messages(project_id, created_at ASC, id ASC)"
    )
    db_execute(
        "CREATE INDEX IF NOT EXISTS writer_projects_updated_idx "
        "ON writer_projects(updated_at DESC, id DESC)"
    )


def list_projects(limit: int = 100) -> list[dict]:
    return db_query(
        """SELECT p.id, p.title, p.premise, p.style_notes, p.status,
                  p.created_at, p.updated_at,
                  COALESCE(m.message_count, 0)::int AS message_count
             FROM writer_projects p
        LEFT JOIN (
                  SELECT project_id, COUNT(*) AS message_count
                    FROM writer_messages
                   GROUP BY project_id
             ) m ON m.project_id = p.id
            ORDER BY p.updated_at DESC, p.id DESC
            LIMIT %s""",
        (limit,),
    )


def create_project(title: str, premise: str = "", style_notes: str = "") -> dict:
    return db_query_one(
        """INSERT INTO writer_projects (title, premise, style_notes)
             VALUES (%s, %s, %s)
          RETURNING id, title, premise, style_notes, status, created_at, updated_at""",
        (title.strip(), premise.strip(), style_notes.strip()),
    ) or {}


def update_project(project_id: int, title: str, premise: str, style_notes: str) -> dict | None:
    return db_query_one(
        """UPDATE writer_projects
              SET title = %s,
                  premise = %s,
                  style_notes = %s,
                  updated_at = NOW()
            WHERE id = %s
        RETURNING id, title, premise, style_notes, status, created_at, updated_at""",
        (title.strip(), premise.strip(), style_notes.strip(), project_id),
    )


def get_project(project_id: int) -> dict | None:
    return db_query_one(
        """SELECT id, title, premise, style_notes, status, created_at, updated_at
             FROM writer_projects
            WHERE id = %s""",
        (project_id,),
    )


def get_project_with_messages(project_id: int) -> dict | None:
    project = get_project(project_id)
    if not project:
        return None
    messages = db_query(
        """SELECT id, role, content, request_kind, model, stop_reason, usage,
                  cost_usd, created_at
             FROM writer_messages
            WHERE project_id = %s
            ORDER BY created_at ASC, id ASC""",
        (project_id,),
    )
    project["messages"] = messages
    return project


def delete_project(project_id: int) -> bool:
    row = db_query_one(
        "DELETE FROM writer_projects WHERE id = %s RETURNING id",
        (project_id,),
    )
    return bool(row)


def _insert_message(
    *,
    project_id: int,
    role: str,
    content: str,
    request_kind: str = "",
    model: str = "",
    stop_reason: str = "",
    usage: dict[str, Any] | None = None,
    cost_usd: float | None = None,
) -> dict:
    return db_query_one(
        """INSERT INTO writer_messages
                  (project_id, role, content, request_kind, model, stop_reason, usage, cost_usd)
             VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
          RETURNING id, role, content, request_kind, model, stop_reason, usage,
                    cost_usd, created_at""",
        (
            project_id,
            role,
            content,
            request_kind,
            model,
            stop_reason,
            json.dumps(usage or {}, ensure_ascii=False),
            cost_usd,
        ),
    ) or {}


def _touch_project(project_id: int) -> None:
    db_execute("UPDATE writer_projects SET updated_at = NOW() WHERE id = %s", (project_id,))


def _build_system_prompt(project: dict, request_kind: str) -> str:
    kind_guidance = {
        "draft": "Write polished prose for the requested scene or chapter.",
        "continue": "Continue from the latest established scene without summarizing unless asked.",
        "revise": "Revise the supplied passage while preserving the user's intent and continuity.",
        "plan": "Produce structural planning, scene beats, or continuity notes.",
        "critique": "Give precise editorial critique and actionable revision notes.",
    }.get(request_kind, "Help with the fiction project according to the user's request.")

    premise = str(project.get("premise") or "").strip() or "(No premise recorded yet.)"
    style_notes = str(project.get("style_notes") or "").strip() or "(No style notes recorded yet.)"
    title = str(project.get("title") or "Untitled").strip()
    return (
        "You are a private fiction-writing collaborator for one user's personal novel project.\n"
        "This workspace is separate from Cyber-Lenin. Do not adopt Cyber-Lenin's identity, "
        "political persona, tools, memory, or public-chat behavior.\n\n"
        f"Project title: {title}\n"
        f"Premise:\n{premise}\n\n"
        f"Style and continuity notes:\n{style_notes}\n\n"
        f"Current task mode: {request_kind or 'draft'}\n"
        f"Task guidance: {kind_guidance}\n\n"
        "Default to substantial, publication-minded prose when drafting. Preserve continuity, "
        "avoid generic disclaimers, and ask at most one concise clarification only when the "
        "request is impossible to fulfill responsibly without it."
    )


def _messages_for_model(project_id: int, user_prompt: str, request_kind: str) -> list[dict]:
    rows = db_query(
        """SELECT role, content
             FROM writer_messages
            WHERE project_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s""",
        (project_id, _MAX_CONTEXT_MESSAGES),
    )
    ordered = list(reversed(rows))
    messages = [
        {"role": row["role"], "content": str(row["content"])}
        for row in ordered
        if row.get("role") in {"user", "assistant"} and str(row.get("content") or "").strip()
    ]
    prefix = f"[mode: {request_kind}]\n" if request_kind else ""
    messages.append({"role": "user", "content": prefix + user_prompt.strip()})
    return messages


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_writer_reply(
    *,
    project_id: int,
    prompt: str,
    request_kind: str,
    max_tokens: int,
) -> AsyncIterator[str]:
    project = get_project(project_id)
    if not project:
        yield _sse({"type": "error", "content": "Project not found."})
        return

    request_kind = request_kind if request_kind in {"draft", "continue", "revise", "plan", "critique"} else "draft"
    model_messages = _messages_for_model(project_id, prompt, request_kind)
    user_row = _insert_message(
        project_id=project_id,
        role="user",
        content=prompt.strip(),
        request_kind=request_kind,
    )
    yield _sse({"type": "user_saved", "message_id": user_row.get("id")})

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    answer_holder: list[str] = []
    error_holder: list[str] = []
    budget_tracker: dict = {}

    async def on_progress(event: str, detail: str):
        if event == "text_delta" and detail:
            await queue.put(_sse({"type": "text_delta", "content": detail}))
        elif event == "budget" and detail:
            await queue.put(_sse({"type": "budget", "content": detail}))

    async def run_llm() -> None:
        try:
            result = await chat_with_tools(
                model_messages,
                client=_client(),
                model=WRITER_MODEL,
                tools=[],
                tool_handlers={},
                system_prompt=_build_system_prompt(project, request_kind),
                max_rounds=1,
                max_tokens=max(256, min(int(max_tokens or 4096), 16000)),
                budget_usd=100.0,
                budget_tracker=budget_tracker,
                on_progress=on_progress,
                agent_name="writer",
            )
            answer_holder.append(result)
        except Exception as exc:
            logger.exception("writer Fable request failed project_id=%s", project_id)
            error_holder.append(str(exc))
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_llm())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await task
    finally:
        if not task.done():
            task.cancel()

    if error_holder:
        yield _sse({"type": "error", "content": f"Claude Fable 5 request failed: {error_holder[0]}"})
        return

    final_text = (answer_holder[0] if answer_holder else "").strip()
    usage = budget_tracker.get("usage") or {}
    cost = budget_tracker.get("total_cost")
    stop_reason = str(budget_tracker.get("stop_reason") or "")
    assistant_row = _insert_message(
        project_id=project_id,
        role="assistant",
        content=final_text,
        request_kind=request_kind,
        model=WRITER_MODEL,
        stop_reason=stop_reason,
        usage=usage,
        cost_usd=cost,
    )
    _touch_project(project_id)
    yield _sse(
        {
            "type": "done",
            "message_id": assistant_row.get("id"),
            "model": WRITER_MODEL,
            "model_display": WRITER_MODEL_DISPLAY,
            "stop_reason": stop_reason,
            "usage": usage,
            "cost_usd": cost,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    )
