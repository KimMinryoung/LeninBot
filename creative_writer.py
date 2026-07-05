"""Personal fiction-writing workspace backed by Claude Fable 5.

This module keeps personal fiction projects separate from the public web chat.
It reuses the shared Anthropic Messages loop for API calls while storing its own
project sessions, canonical manuscript text, searchable chunks, and revisions in
writer-specific tables.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import anthropic
from psycopg2.extras import RealDictCursor, execute_values

from db import get_conn
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

# Selectable models. Fable is the default; DeepSeek options route through the
# same chat_with_tools loop via bot_config's Anthropic-compatible DeepSeek
# client. Prices are display hints only — authoritative cost comes from
# claude_loop.PRICING_TABLE (keyed by model id) at runtime.
WRITER_MODEL_CHOICES: dict[str, dict] = {
    "fable": {
        "provider": "anthropic",
        "model": "claude-fable-5",
        "display": "Claude Fable 5",
        "input_price_per_mtok": 10.0,
        "output_price_per_mtok": 50.0,
    },
    "deepseek_pro": {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "display": "DeepSeek V4 Pro",
        "input_price_per_mtok": 0.435,
        "output_price_per_mtok": 0.87,
    },
    "deepseek_flash": {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "display": "DeepSeek V4 Flash",
        "input_price_per_mtok": 0.14,
        "output_price_per_mtok": 0.28,
    },
}
WRITER_DEFAULT_CHOICE = "fable"

WRITER_DEFAULT_MAX_TOKENS = 12000
# Tool-use rounds: allow a few research/continuity searches before the model
# writes. 1 = no tools (legacy). Each round is a Fable-priced model call.
_WRITER_MAX_ROUNDS = 6
# Web search is disabled by design: Fable 5's internal knowledge is strong and
# the writer prefers reference material supplied directly by the user. Flip to
# True to re-enable the Tavily-backed web_search tool (and its prompt guidance).
_WRITER_WEB_SEARCH_ENABLED = False
_WRITER_CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}
_MAX_CONTEXT_MESSAGES = 80
_MANUSCRIPT_CHUNK_SIZE = 3500
_MANUSCRIPT_CHUNK_OVERLAP = 300
_MANUSCRIPT_TAIL_CHARS = 7000
_MANUSCRIPT_SELECTION_LIMIT = 20000

_writer_client: anthropic.AsyncAnthropic | None = None


def _client() -> anthropic.AsyncAnthropic:
    global _writer_client
    if _writer_client is None:
        api_key = get_secret("WRITER_ANTHROPIC_API_KEY", "") or get_secret("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("WRITER_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY is required")
        _writer_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _writer_client


def _deepseek_available() -> bool:
    try:
        import bot_config
        return bot_config._deepseek_anthropic_client is not None
    except Exception:
        return False


def list_writer_models() -> list[dict]:
    """Public metadata for the model picker: keys, display names, prices, availability."""
    out: list[dict] = []
    for key, spec in WRITER_MODEL_CHOICES.items():
        available = True if spec["provider"] == "anthropic" else _deepseek_available()
        out.append({
            "key": key,
            "id": spec["model"],
            "display_name": spec["display"],
            "provider": spec["provider"],
            "input_price_per_mtok": spec["input_price_per_mtok"],
            "output_price_per_mtok": spec["output_price_per_mtok"],
            "available": available,
            "default": key == WRITER_DEFAULT_CHOICE,
        })
    return out


def _resolve_writer_model(choice: str | None) -> tuple[Any, str, str, dict]:
    """Resolve a model choice key to (client, model_id, display_name, extra_kwargs).

    extra_kwargs carries provider-specific chat_with_tools kwargs (e.g. DeepSeek
    thinking/output_config). Raises ValueError for an unknown key and
    RuntimeError if the chosen provider is not configured.
    """
    key = (choice or "").strip() or WRITER_DEFAULT_CHOICE
    spec = WRITER_MODEL_CHOICES.get(key)
    if spec is None:
        raise ValueError(f"Unknown writer model choice: {choice!r}")
    if spec["provider"] == "deepseek":
        import bot_config
        client = bot_config._deepseek_anthropic_client
        if client is None:
            raise RuntimeError("DeepSeek is not configured (DEEPSEEK_API_KEY missing).")
        # Thinking-on, effort-high by default (same as other non-web DeepSeek
        # agents). The writer never forces tool_choice, so thinking is stable.
        extra = bot_config._get_deepseek_thinking_params()
        return client, spec["model"], spec["display"], extra
    return _client(), spec["model"], spec["display"], {}


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
        """CREATE TABLE IF NOT EXISTS writer_manuscripts (
               project_id BIGINT PRIMARY KEY REFERENCES writer_projects(id) ON DELETE CASCADE,
               body TEXT NOT NULL DEFAULT '',
               created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
           )"""
    )
    db_execute(
        """CREATE TABLE IF NOT EXISTS writer_manuscript_chunks (
               id BIGSERIAL PRIMARY KEY,
               project_id BIGINT NOT NULL REFERENCES writer_projects(id) ON DELETE CASCADE,
               chunk_index INTEGER NOT NULL,
               start_offset INTEGER NOT NULL,
               end_offset INTEGER NOT NULL,
               heading TEXT NOT NULL DEFAULT '',
               excerpt TEXT NOT NULL,
               created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               UNIQUE(project_id, chunk_index)
           )"""
    )
    db_execute(
        """CREATE TABLE IF NOT EXISTS writer_manuscript_revisions (
               id BIGSERIAL PRIMARY KEY,
               project_id BIGINT NOT NULL REFERENCES writer_projects(id) ON DELETE CASCADE,
               action TEXT NOT NULL,
               before_body TEXT NOT NULL DEFAULT '',
               after_body TEXT NOT NULL DEFAULT '',
               note TEXT NOT NULL DEFAULT '',
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
    db_execute(
        "CREATE INDEX IF NOT EXISTS writer_manuscript_chunks_project_idx "
        "ON writer_manuscript_chunks(project_id, chunk_index ASC)"
    )
    db_execute(
        "CREATE INDEX IF NOT EXISTS writer_manuscript_revisions_project_idx "
        "ON writer_manuscript_revisions(project_id, created_at DESC, id DESC)"
    )


def list_projects(limit: int = 100) -> list[dict]:
    return db_query(
        """SELECT p.id, p.title, p.premise, p.style_notes, p.status,
                  p.created_at, p.updated_at,
                  COALESCE(m.message_count, 0)::int AS message_count,
                  COALESCE(length(w.body), 0)::int AS manuscript_char_count,
                  w.updated_at AS manuscript_updated_at
             FROM writer_projects p
        LEFT JOIN (
                  SELECT project_id, COUNT(*) AS message_count
                    FROM writer_messages
                   GROUP BY project_id
             ) m ON m.project_id = p.id
        LEFT JOIN writer_manuscripts w ON w.project_id = p.id
            ORDER BY p.updated_at DESC, p.id DESC
            LIMIT %s""",
        (limit,),
    )


def create_project(title: str, premise: str = "", style_notes: str = "") -> dict:
    project = db_query_one(
        """INSERT INTO writer_projects (title, premise, style_notes)
             VALUES (%s, %s, %s)
          RETURNING id, title, premise, style_notes, status, created_at, updated_at""",
        (title.strip(), premise.strip(), style_notes.strip()),
    ) or {}
    if project.get("id"):
        _ensure_manuscript_row(int(project["id"]))
    return project


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
    project = db_query_one(
        """SELECT p.id, p.title, p.premise, p.style_notes, p.status,
                  p.created_at, p.updated_at,
                  COALESCE(length(w.body), 0)::int AS manuscript_char_count,
                  w.updated_at AS manuscript_updated_at
             FROM writer_projects p
        LEFT JOIN writer_manuscripts w ON w.project_id = p.id
            WHERE p.id = %s""",
        (project_id,),
    )
    if project:
        _ensure_manuscript_row(project_id)
    return project


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


def _ensure_manuscript_row(project_id: int) -> None:
    db_execute(
        """INSERT INTO writer_manuscripts (project_id, body)
             SELECT %s, ''
              WHERE EXISTS (SELECT 1 FROM writer_projects WHERE id = %s)
         ON CONFLICT (project_id) DO NOTHING""",
        (project_id, project_id),
    )


def _chunk_manuscript(body: str) -> list[dict[str, Any]]:
    if not body:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    index = 0
    length = len(body)
    while start < length:
        hard_end = min(length, start + _MANUSCRIPT_CHUNK_SIZE)
        end = hard_end
        if hard_end < length:
            window = body[start:hard_end]
            split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if split_at >= int(_MANUSCRIPT_CHUNK_SIZE * 0.55):
                end = start + split_at + (2 if window[split_at:split_at + 2] == "\n\n" else 1)
        excerpt = body[start:end]
        chunks.append(
            {
                "chunk_index": index,
                "start_offset": start,
                "end_offset": end,
                "heading": _heading_for_offset(body, start),
                "excerpt": excerpt,
            }
        )
        if end >= length:
            break
        start = max(end - _MANUSCRIPT_CHUNK_OVERLAP, start + 1)
        index += 1
    return chunks


def _heading_for_offset(body: str, offset: int) -> str:
    prefix = body[max(0, offset - 1400):offset]
    for line in reversed(prefix.splitlines()):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped[:160]
    for line in body[offset:offset + 500].splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:160]
    return ""


def _reindex_manuscript(project_id: int, body: str) -> None:
    db_execute("DELETE FROM writer_manuscript_chunks WHERE project_id = %s", (project_id,))
    for chunk in _chunk_manuscript(body):
        db_execute(
            """INSERT INTO writer_manuscript_chunks
                      (project_id, chunk_index, start_offset, end_offset, heading, excerpt)
                 VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                project_id,
                chunk["chunk_index"],
                chunk["start_offset"],
                chunk["end_offset"],
                chunk["heading"],
                chunk["excerpt"],
            ),
        )


def get_manuscript(project_id: int) -> dict | None:
    if not get_project(project_id):
        return None
    _ensure_manuscript_row(project_id)
    row = db_query_one(
        """SELECT project_id, body, length(body)::int AS char_count,
                  created_at, updated_at
             FROM writer_manuscripts
            WHERE project_id = %s""",
        (project_id,),
    )
    return row


def _write_manuscript(project_id: int, body: str, *, action: str, note: str = "") -> dict | None:
    chunks = _chunk_manuscript(body)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT p.id, w.body
                     FROM writer_projects p
                LEFT JOIN writer_manuscripts w ON w.project_id = p.id
                    WHERE p.id = %s""",
                (project_id,),
            )
            previous = cur.fetchone()
            if not previous:
                return None
            before_body = str(previous.get("body") or "")
            cur.execute(
                """INSERT INTO writer_manuscripts (project_id, body)
                     VALUES (%s, '')
                ON CONFLICT (project_id) DO NOTHING""",
                (project_id,),
            )
            cur.execute(
                """INSERT INTO writer_manuscript_revisions
                          (project_id, action, before_body, after_body, note)
                     VALUES (%s, %s, %s, %s, %s)""",
                (project_id, action, before_body, body, note.strip()),
            )
            cur.execute(
                """UPDATE writer_manuscripts
                      SET body = %s,
                          updated_at = NOW()
                    WHERE project_id = %s""",
                (body, project_id),
            )
            cur.execute("DELETE FROM writer_manuscript_chunks WHERE project_id = %s", (project_id,))
            if chunks:
                execute_values(
                    cur,
                    """INSERT INTO writer_manuscript_chunks
                              (project_id, chunk_index, start_offset, end_offset, heading, excerpt)
                         VALUES %s""",
                    [
                        (
                            project_id,
                            chunk["chunk_index"],
                            chunk["start_offset"],
                            chunk["end_offset"],
                            chunk["heading"],
                            chunk["excerpt"],
                        )
                        for chunk in chunks
                    ],
                )
            cur.execute("UPDATE writer_projects SET updated_at = NOW() WHERE id = %s", (project_id,))
            cur.execute(
                """SELECT project_id, body, length(body)::int AS char_count,
                          created_at, updated_at
                     FROM writer_manuscripts
                    WHERE project_id = %s""",
                (project_id,),
            )
            return dict(cur.fetchone())


def save_manuscript(project_id: int, body: str, note: str = "") -> dict | None:
    return _write_manuscript(project_id, body, action="save", note=note)


def append_manuscript(project_id: int, text: str, note: str = "") -> dict | None:
    manuscript = get_manuscript(project_id)
    if not manuscript:
        return None
    current = str(manuscript.get("body") or "")
    addition = text.strip("\n")
    if not addition:
        return manuscript
    separator = "\n\n" if current and not current.endswith("\n\n") else ""
    return _write_manuscript(project_id, current + separator + addition, action="append", note=note)


def replace_manuscript_range(
    project_id: int,
    start: int,
    end: int,
    replacement: str,
    note: str = "",
) -> dict | None:
    manuscript = get_manuscript(project_id)
    if not manuscript:
        return None
    current = str(manuscript.get("body") or "")
    if start < 0 or end < start or end > len(current):
        raise ValueError("invalid manuscript range")
    updated = current[:start] + replacement + current[end:]
    return _write_manuscript(project_id, updated, action="replace", note=note)


def search_manuscript(project_id: int, query: str, limit: int = 20) -> list[dict]:
    needle = query.strip()
    if not needle:
        return []
    _ensure_manuscript_row(project_id)
    rows = db_query(
        """SELECT id, chunk_index, start_offset, end_offset, heading, excerpt
             FROM writer_manuscript_chunks
            WHERE project_id = %s
              AND excerpt ILIKE %s
            ORDER BY chunk_index ASC
            LIMIT %s""",
        (project_id, f"%{needle}%", limit),
    )
    results: list[dict] = []
    lowered = needle.lower()
    for row in rows:
        excerpt = str(row.get("excerpt") or "")
        found = excerpt.lower().find(lowered)
        if found < 0:
            found = 0
        snippet_start = max(0, found - 180)
        snippet_end = min(len(excerpt), found + len(needle) + 220)
        results.append(
            {
                "id": row.get("id"),
                "chunk_index": row.get("chunk_index"),
                "start_offset": row.get("start_offset"),
                "end_offset": row.get("end_offset"),
                "match_start": int(row.get("start_offset") or 0) + found,
                "match_end": int(row.get("start_offset") or 0) + found + len(needle),
                "heading": row.get("heading") or "",
                "snippet": excerpt[snippet_start:snippet_end],
            }
        )
    return results


def list_manuscript_revisions(project_id: int, limit: int = 30) -> list[dict]:
    return db_query(
        """SELECT id, project_id, action, note,
                  length(before_body)::int AS before_char_count,
                  length(after_body)::int AS after_char_count,
                  created_at
             FROM writer_manuscript_revisions
            WHERE project_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s""",
        (project_id, limit),
    )


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


def _research_tools_section() -> str:
    """The '# Research & continuity tools' prompt block, gated to the tools that
    are actually wired (see _build_writer_tools). Advertising a tool the model
    can't call would just make it hallucinate calls."""
    lines = [
        "# Research & continuity tools\n",
        "You have tools; use them silently and on your own initiative before writing, then produce the response below.\n",
        "- search_manuscript: search the FULL manuscript (names, facts, earlier scenes) beyond the recent tail you were given. "
        "Use it to verify continuity before introducing or contradicting an established detail.\n",
    ]
    if _WRITER_WEB_SEARCH_ENABLED:
        lines.append(
            "- web_search: look up real-world facts (history, geography, technical or domain detail) when accuracy would ground the fiction. "
            "Do not over-research or let it flatten the prose into an encyclopedia; use it only to get concrete details right.\n"
        )
    lines.append(
        "Tool calls are invisible to the reader — never narrate that you searched. "
        "Fold what you learn directly into the prose or commentary.\n\n"
    )
    return "".join(lines)


def _build_base_system_prompt(project: dict) -> str:
    premise = str(project.get("premise") or "").strip() or "(No premise recorded yet.)"
    style_notes = str(project.get("style_notes") or "").strip() or "(No style notes recorded yet.)"
    title = str(project.get("title") or "Untitled").strip()
    return (
        "You are an elite fiction-writing collaborator on one writer's personal novel. "
        "You are not a chatbot or an assistant persona — you are a craftsperson serving this manuscript, "
        "and prose quality is the only thing that matters.\n\n"
        f"Project title: {title}\n"
        f"Premise:\n{premise}\n\n"
        f"Style and continuity notes:\n{style_notes}\n\n"
        "# Craft standards\n"
        "- Write in the language, voice, point of view, and tense already established in the manuscript context. "
        "Match its prose rhythm and diction; never reset to a generic default voice.\n"
        "- Dramatize through concrete action, sensory detail, and subtext. Do not summarize emotion, "
        "explain the subtext, or state the theme outright — trust the reader.\n"
        "- Vary sentence length and structure. Favor strong verbs and specific nouns over adverbs and abstraction.\n"
        "- Write dialogue that carries character, tension, and information indirectly; avoid on-the-nose exposition.\n"
        "- Cut clichés, filler, and AI tells (e.g. 'little did they know', 'a testament to', "
        "reflexive over-explaining, neatly moralized endings, purple padding).\n"
        "- Honor continuity absolutely: names, timeline, established facts, and what each character can plausibly know. "
        "If the request conflicts with the established draft, follow continuity and flag the conflict in commentary.\n\n"
        + _research_tools_section()
        + "# Manuscript delta discipline\n"
        "- The manuscript context is the authoritative draft.\n"
        "- To continue the story, write prose that flows seamlessly from the manuscript tail — never repeat existing text.\n"
        "- When the user has selected a range to revise, return ONLY the replacement text for that range.\n"
        "- Put nothing but publishable manuscript prose inside <manuscript_delta>: no headings, labels, or meta-notes.\n\n"
        "# Response format — use exactly these tags:\n"
        "<manuscript_delta>\n"
        "Manuscript-ready prose to append, or the replacement for the selected range. "
        "Leave empty if this turn needs no manuscript text.\n"
        "</manuscript_delta>\n"
        "<commentary>\n"
        "Concise working notes: choices made, continuity assumptions, risks, and any genuine question for the writer. "
        "No praise, no filler, no boilerplate caveats.\n"
        "</commentary>"
    )

def _manuscript_context(project_id: int, selection_start: int | None, selection_end: int | None) -> str:
    manuscript = get_manuscript(project_id) or {}
    body = str(manuscript.get("body") or "")
    parts = [f"Manuscript character count: {len(body)}"]
    if body:
        tail = body[-_MANUSCRIPT_TAIL_CHARS:]
        parts.append("Recent manuscript tail:\n" + tail)
    if selection_start is not None and selection_end is not None and body:
        start = max(0, min(selection_start, len(body)))
        end = max(start, min(selection_end, len(body)))
        selected = body[start:end]
        if selected:
            if len(selected) > _MANUSCRIPT_SELECTION_LIMIT:
                selected = selected[:_MANUSCRIPT_SELECTION_LIMIT] + "\n[selection truncated]"
            parts.append(f"Selected manuscript range {start}:{end}:\n{selected}")
    return "\n\n".join(parts)




def _build_system_blocks(
    project: dict,
    project_id: int,
    selection_start: int | None,
    selection_end: int | None,
) -> list[dict]:
    return [
        {
            "type": "text",
            "text": _build_base_system_prompt(project),
            "cache_control": _WRITER_CACHE_CONTROL_1H,
        },
        {
            "type": "text",
            "text": "<manuscript_context>\n"
            + _manuscript_context(project_id, selection_start, selection_end)
            + "\n</manuscript_context>",
            "cache_control": _WRITER_CACHE_CONTROL_1H,
        },
    ]

def _messages_for_model(
    project_id: int,
    user_prompt: str,
    selection_start: int | None = None,
    selection_end: int | None = None,
) -> list[dict]:
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
    current_turn = "<user_request>\n" + user_prompt.strip() + "\n</user_request>"
    messages.append({"role": "user", "content": current_turn})
    return messages


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


_MANUSCRIPT_DELTA_RE = re.compile(
    r"<manuscript_delta>\s*(.*?)\s*</manuscript_delta>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENTARY_RE = re.compile(
    r"<commentary>\s*(.*?)\s*</commentary>",
    re.IGNORECASE | re.DOTALL,
)


def _parse_writer_response(text: str) -> dict[str, str]:
    manuscript_match = _MANUSCRIPT_DELTA_RE.search(text)
    commentary_match = _COMMENTARY_RE.search(text)
    manuscript_text = manuscript_match.group(1).strip() if manuscript_match else ""
    commentary_text = commentary_match.group(1).strip() if commentary_match else ""
    if manuscript_match or commentary_match:
        remaining = _MANUSCRIPT_DELTA_RE.sub("", text)
        remaining = _COMMENTARY_RE.sub("", remaining).strip()
        if remaining and not commentary_text:
            commentary_text = remaining
    else:
        manuscript_text = text.strip()
    display_parts = []
    if manuscript_text:
        display_parts.append("Manuscript\n" + manuscript_text)
    if commentary_text:
        display_parts.append("Notes\n" + commentary_text)
    return {
        "manuscript_text": manuscript_text,
        "commentary_text": commentary_text,
        "display_text": "\n\n".join(display_parts) or text.strip(),
    }


_SEARCH_MANUSCRIPT_TOOL = {
    "name": "search_manuscript",
    "description": (
        "Search the full manuscript for a phrase (a character name, an established fact, an earlier scene) "
        "beyond the recent tail you were given in context. Returns matching passages with their character "
        "offsets. Use before introducing or contradicting an established detail to keep continuity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Substring or phrase to find in the manuscript."},
            "limit": {"type": "integer", "description": "Max passages to return (1-20).", "default": 8},
        },
        "required": ["query"],
    },
}


def _build_writer_tools(project_id: int) -> tuple[list[dict], dict]:
    """Build the (tools, handlers) pair for a writer turn, bound to one project.

    - search_manuscript: closure over project_id so the model never supplies it.
    - web_search: reuse the main runtime's Tavily-backed tool as-is.
    Both are read-only and pass the security gateway (uncategorized / fetch).
    """

    async def _handle_search_manuscript(query: str, limit: int = 8) -> str:
        try:
            n = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            n = 8
        rows = await asyncio.to_thread(search_manuscript, project_id, query, n)
        if not rows:
            return f"No manuscript matches for: {query}"
        blocks = []
        for r in rows:
            heading = str(r.get("heading") or "").strip()
            label = f"[{heading}] " if heading else ""
            snippet = str(r.get("snippet") or "").strip()
            blocks.append(
                f"{label}chars {r.get('start_offset')}–{r.get('end_offset')} "
                f"(match {r.get('match_start')}:{r.get('match_end')}):\n…{snippet}…"
            )
        return "\n\n".join(blocks)

    tools: list[dict] = [_SEARCH_MANUSCRIPT_TOOL]
    handlers: dict = {"search_manuscript": _handle_search_manuscript}

    # Web search is disabled by default (see _WRITER_WEB_SEARCH_ENABLED). When on,
    # reuse the main runtime's web_search (schema + Tavily handler).
    if _WRITER_WEB_SEARCH_ENABLED:
        try:
            from runtime_tools.registry import TOOLS as _RT_TOOLS, TOOL_HANDLERS as _RT_HANDLERS

            web_spec = next((t for t in _RT_TOOLS if t.get("name") == "web_search"), None)
            web_handler = _RT_HANDLERS.get("web_search")
            if web_spec and web_handler:
                tools.append(web_spec)
                handlers["web_search"] = web_handler
        except Exception:
            logger.exception("writer: web_search tool unavailable; continuing with manuscript search only")

    return tools, handlers


async def stream_writer_reply(
    *,
    project_id: int,
    prompt: str,
    selection_start: int | None = None,
    selection_end: int | None = None,
    model_choice: str | None = None,
) -> AsyncIterator[str]:
    project = get_project(project_id)
    if not project:
        yield _sse({"type": "error", "content": "Project not found."})
        return

    try:
        writer_client, writer_model, writer_display, model_extra = _resolve_writer_model(model_choice)
    except (ValueError, RuntimeError) as exc:
        yield _sse({"type": "error", "content": str(exc)})
        return

    request_kind = ""
    model_messages = _messages_for_model(project_id, prompt, selection_start, selection_end)
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

    writer_tools, writer_handlers = _build_writer_tools(project_id)

    async def on_progress(event: str, detail: str):
        if event == "text_delta" and detail:
            await queue.put(_sse({"type": "text_delta", "content": detail}))
        elif event == "budget" and detail:
            await queue.put(_sse({"type": "budget", "content": detail}))
        elif event in ("tool_call", "tool_result") and detail:
            await queue.put(_sse({"type": "tool", "content": detail}))

    async def run_llm() -> None:
        try:
            result = await chat_with_tools(
                model_messages,
                client=writer_client,
                model=writer_model,
                tools=writer_tools,
                tool_handlers=writer_handlers,
                system_prompt=_build_system_blocks(project, project_id, selection_start, selection_end),
                max_rounds=_WRITER_MAX_ROUNDS,
                max_tokens=WRITER_DEFAULT_MAX_TOKENS,
                budget_usd=100.0,
                budget_tracker=budget_tracker,
                on_progress=on_progress,
                agent_name="writer",
                **model_extra,
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
        yield _sse({"type": "error", "content": f"{writer_display} request failed: {error_holder[0]}"})
        return

    final_text = (answer_holder[0] if answer_holder else "").strip()
    parsed_response = _parse_writer_response(final_text)
    usage = budget_tracker.get("usage") or {}
    cost = budget_tracker.get("total_cost")
    stop_reason = str(budget_tracker.get("stop_reason") or "")
    assistant_row = _insert_message(
        project_id=project_id,
        role="assistant",
        content=final_text,
        request_kind=request_kind,
        model=writer_model,
        stop_reason=stop_reason,
        usage=usage,
        cost_usd=cost,
    )
    _touch_project(project_id)
    yield _sse(
        {
            "type": "done",
            "message_id": assistant_row.get("id"),
            "model": writer_model,
            "model_display": writer_display,
            "stop_reason": stop_reason,
            "usage": usage,
            "cost_usd": cost,
            "manuscript_text": parsed_response["manuscript_text"],
            "commentary_text": parsed_response["commentary_text"],
            "display_text": parsed_response["display_text"],
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    )
