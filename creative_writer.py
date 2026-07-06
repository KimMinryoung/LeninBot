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
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Awaitable, Callable

import anthropic
from psycopg2.extras import RealDictCursor, execute_values

from db import get_conn
from db import execute as db_execute
from db import query as db_query
from db import query_one as db_query_one
from claude_loop import chat_with_tools
from secrets_loader import get_secret
from tool_gateway.security import CallerContext, caller_scope

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
# Tool-use rounds: allow extended continuity searches and edit retries before
# the model writes. 1 = no tools (legacy). Each round is a Fable-priced model call.
_WRITER_MAX_ROUNDS = 16
_WRITER_IDLE_TIMEOUT_SEC = 240
_WRITER_PROVIDER_IDLE_TIMEOUT_SEC = 70
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
_writer_background_tasks: set[asyncio.Task] = set()


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
        """CREATE TABLE IF NOT EXISTS writer_documents (
               id BIGSERIAL PRIMARY KEY,
               project_id BIGINT NOT NULL REFERENCES writer_projects(id) ON DELETE CASCADE,
               title TEXT NOT NULL,
               kind TEXT NOT NULL DEFAULT 'note',
               content TEXT NOT NULL DEFAULT '',
               created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               UNIQUE(project_id, title)
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
    db_execute(
        "CREATE INDEX IF NOT EXISTS writer_documents_project_idx "
        "ON writer_documents(project_id, updated_at DESC, id DESC)"
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
    _writer_tools_cache.pop(project_id, None)
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
    # Full strip, not just newlines: models often send a leading space or
    # trailing blank padding, which survives verbatim into the draft. The
    # paragraph break is owned by the separator below, not by the input.
    addition = text.strip()
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


# Straight and curly single/double quotes drift between what the model
# remembers and what the manuscript stores; treat them as interchangeable
# when locating a passage.
_QUOTE_CHARS = "\"'‘’“”"
_QUOTE_CLASS = f"[{_QUOTE_CHARS}]"


def _normalized_pattern(find: str) -> re.Pattern | None:
    """Whitespace- and quote-insensitive pattern for locating a passage the
    model quoted from memory. Returns None for degenerate input."""
    tokens = find.split()
    if not tokens or len(tokens) > 500:
        return None
    parts = []
    for token in tokens:
        escaped = re.sub(_QUOTE_CLASS, _QUOTE_CLASS, re.escape(token))
        parts.append(escaped)
    try:
        return re.compile(r"\s+".join(parts))
    except re.error:
        return None


def _find_normalized_matches(body: str, find: str, max_matches: int = 3) -> list[tuple[int, int]]:
    pattern = _normalized_pattern(find)
    if pattern is None:
        return []
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(body):
        spans.append((match.start(), match.end()))
        if len(spans) >= max_matches:
            break
    return spans


def replace_manuscript_text(project_id: int, find: str, replacement: str, note: str = "") -> dict:
    """Find one exact passage and replace it. Safe for LLM tool use: fails
    (ok=False) if the passage is absent or appears more than once, so the model
    can never silently edit the wrong location. Falls back to whitespace/quote-
    normalized matching when the verbatim text is not found. Revision history is
    preserved by _write_manuscript. Returns {ok, message, char_count?}."""
    if not find:
        return {"ok": False, "message": "Empty 'find' text."}
    manuscript = get_manuscript(project_id)
    if not manuscript:
        return {"ok": False, "message": "Manuscript not found."}
    current = str(manuscript.get("body") or "")
    count = current.count(find)
    if count > 1:
        return {"ok": False, "message": f"Passage appears {count} times; include more surrounding text so it is unique."}
    if count == 1:
        idx = current.find(find)
        span = (idx, idx + len(find))
        matched_how = "verbatim"
    else:
        spans = _find_normalized_matches(current, find)
        if not spans:
            return {
                "ok": False,
                "message": (
                    "Passage not found in the saved manuscript, even ignoring whitespace/quote differences. "
                    "Copy the real saved text: locate it with search_manuscript (short distinctive phrase) "
                    "or read_manuscript (character range), then retry."
                ),
            }
        if len(spans) > 1:
            return {"ok": False, "message": "Passage matches multiple places (whitespace-insensitive); include more surrounding text so it is unique."}
        span = spans[0]
        matched_how = "normalized whitespace/quotes"
    updated = current[:span[0]] + replacement + current[span[1]:]
    result = _write_manuscript(project_id, updated, action="replace", note=note or "tool replace")
    return {"ok": True, "message": f"Replaced (matched {matched_how}).", "char_count": (result or {}).get("char_count")}


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
    if not results and len(needle.split()) >= 2:
        # A multi-word query that missed is usually the model quoting from
        # memory with whitespace/quote drift; retry insensitive to both.
        manuscript = get_manuscript(project_id) or {}
        body = str(manuscript.get("body") or "")
        for start, end in _find_normalized_matches(body, needle, max_matches=limit):
            results.append(
                {
                    "id": None,
                    "chunk_index": None,
                    "start_offset": start,
                    "end_offset": end,
                    "match_start": start,
                    "match_end": end,
                    "heading": _heading_for_offset(body, start),
                    "snippet": body[max(0, start - 180):min(len(body), end + 220)],
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


_DOCUMENT_COLUMNS = "id, project_id, title, kind, length(content)::int AS char_count, created_at, updated_at"


def list_documents(project_id: int) -> list[dict]:
    return db_query(
        f"""SELECT {_DOCUMENT_COLUMNS}
             FROM writer_documents
            WHERE project_id = %s
            ORDER BY updated_at DESC, id DESC""",
        (project_id,),
    )


def get_document(project_id: int, document_id: int | None = None, title: str | None = None) -> dict | None:
    if document_id is not None:
        return db_query_one(
            f"""SELECT {_DOCUMENT_COLUMNS}, content
                 FROM writer_documents
                WHERE project_id = %s AND id = %s""",
            (project_id, document_id),
        )
    if title:
        return db_query_one(
            f"""SELECT {_DOCUMENT_COLUMNS}, content
                 FROM writer_documents
                WHERE project_id = %s AND lower(title) = lower(%s)""",
            (project_id, title.strip()),
        )
    return None


def save_document(project_id: int, title: str, content: str, kind: str = "note") -> dict | None:
    """Create or overwrite a background document, addressed by title (upsert)."""
    if not get_project(project_id):
        return None
    return db_query_one(
        f"""INSERT INTO writer_documents (project_id, title, kind, content)
             VALUES (%s, %s, %s, %s)
        ON CONFLICT (project_id, title)
          DO UPDATE SET kind = EXCLUDED.kind,
                        content = EXCLUDED.content,
                        updated_at = NOW()
          RETURNING {_DOCUMENT_COLUMNS}""",
        (project_id, title.strip(), kind.strip() or "note", content),
    )


def update_document(project_id: int, document_id: int, title: str, kind: str, content: str) -> dict | None:
    return db_query_one(
        f"""UPDATE writer_documents
               SET title = %s, kind = %s, content = %s, updated_at = NOW()
             WHERE project_id = %s AND id = %s
         RETURNING {_DOCUMENT_COLUMNS}""",
        (title.strip(), kind.strip() or "note", content, project_id, document_id),
    )


def delete_document(project_id: int, document_id: int) -> bool:
    row = db_query_one(
        "DELETE FROM writer_documents WHERE project_id = %s AND id = %s RETURNING id",
        (project_id, document_id),
    )
    return bool(row)


def search_documents(project_id: int, query: str, limit: int = 8) -> list[dict]:
    needle = query.strip()
    if not needle:
        return []
    rows = db_query(
        """SELECT id, title, kind, content
             FROM writer_documents
            WHERE project_id = %s
              AND (title ILIKE %s OR content ILIKE %s)
            ORDER BY updated_at DESC
            LIMIT %s""",
        (project_id, f"%{needle}%", f"%{needle}%", limit),
    )
    results: list[dict] = []
    lowered = needle.lower()
    for row in rows:
        content = str(row.get("content") or "")
        found = content.lower().find(lowered)
        if found < 0:
            snippet = content[:400]
        else:
            snippet = content[max(0, found - 180):found + len(needle) + 220]
        results.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "kind": row.get("kind"),
                "snippet": snippet,
            }
        )
    return results


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


def _tools_prompt_section() -> str:
    """The '# Tools' prompt block, gated to the tools that are actually wired
    (see _build_writer_tools). Advertising a tool the model can't call would
    just make it hallucinate calls."""
    lines = [
        "# Tools — you edit the manuscript yourself\n",
        "You act on the manuscript directly through tools; the writer never copies text by hand. "
        "Use tools silently and on your own initiative, then give a short commentary.\n",
        "- search_manuscript(query): exact-substring search of the FULL saved manuscript. Use SHORT distinctive "
        "queries (a name, a 2-6 word phrase) to check continuity or locate an earlier scene. Long quoted "
        "paragraphs rarely match.\n",
        "- read_manuscript(start, end): read any region of the saved manuscript by character offsets "
        "(no arguments = the last 5000 chars). Prefer this over repeated searching when you need context.\n",
        "- append_to_manuscript(text): add prose to the END of the manuscript — for continuing the story.\n",
        "- replace_in_manuscript(find, replacement): revise a specific part — 'find' must match saved text "
        "(whitespace/quote drift tolerated) and be unique. Take 'find' from read_manuscript or search_manuscript "
        "results, or from the context tail.\n",
        "- read_document(title) / search_documents(query): consult the project's background documents "
        "(character sheets, worldbuilding, outline, research). They are the authoritative reference for "
        "setting and plot facts — check them before inventing or contradicting a detail.\n",
        "- save_document(title, content, kind): create or fully overwrite a background document. Use when the "
        "writer asks you to record notes, or to keep an agreed story bible current. Never store manuscript "
        "prose in documents.\n",
    ]
    if _WRITER_WEB_SEARCH_ENABLED:
        lines.append(
            "- web_search: look up real-world facts (history, geography, technical or domain detail) when accuracy would ground the fiction. "
            "Do not over-research or let it flatten the prose into an encyclopedia; use it only to get concrete details right.\n"
        )
    lines.append(
        "Every manuscript change MUST go through append_to_manuscript or replace_in_manuscript — never paste manuscript "
        "prose into your text reply. Tool calls are invisible to the reader; never narrate that you searched or edited.\n"
        "Tool economy: the manuscript tail in your context IS the current saved draft — do not re-search or re-read text "
        "you can already see, and never search for prose you wrote earlier in this same turn (a successful tool result "
        "already confirms it was saved). One or two well-chosen lookups beat a chain of guesses.\n\n"
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
        + _tools_prompt_section()
        + "# Editing discipline\n"
        "- The saved manuscript is the authoritative draft; your tools edit it in place (every change is reversible).\n"
        "- Continue the story with append_to_manuscript; revise a specific part with replace_in_manuscript.\n"
        "- Make each edit flow seamlessly with the surrounding prose; never duplicate text that already exists.\n"
        "- If the writer selected a range to revise, replace exactly that range (use its text as 'find').\n"
        "- If the writer asks a question, asks for diagnosis, asks for options, or brainstorms without requesting a manuscript edit, "
        "make NO edit and answer directly in commentary.\n"
        "- If the request is ambiguous or would break continuity, make NO edit and ask in commentary instead.\n\n"
        "# Response format\n"
        "After applying your edits with the tools, reply with ONLY a commentary block:\n"
        "<commentary>\n"
        "A brief note: what you changed (appended vs revised, and which part), key choices, continuity assumptions, "
        "and any genuine question for the writer. Do NOT paste the manuscript prose here. "
        "No praise, no filler, no boilerplate caveats.\n"
        "</commentary>"
    )

def _manuscript_context(project_id: int, selection_start: int | None, selection_end: int | None) -> str:
    manuscript = get_manuscript(project_id) or {}
    body = str(manuscript.get("body") or "")
    parts = [f"Manuscript character count: {len(body)}"]
    if body:
        tail = body[-_MANUSCRIPT_TAIL_CHARS:]
        parts.append(
            f"Recent manuscript tail (chars {len(body) - len(tail)}–{len(body)}, already saved):\n" + tail
        )
    if selection_start is not None and selection_end is not None and body:
        start = max(0, min(selection_start, len(body)))
        end = max(start, min(selection_end, len(body)))
        selected = body[start:end]
        if selected:
            if len(selected) > _MANUSCRIPT_SELECTION_LIMIT:
                selected = selected[:_MANUSCRIPT_SELECTION_LIMIT] + "\n[selection truncated]"
            parts.append(f"Selected manuscript range {start}:{end}:\n{selected}")
    documents = list_documents(project_id)
    if documents:
        listing = "\n".join(
            f"- {str(d.get('title'))!r} (kind: {d.get('kind')}, {d.get('char_count')} chars)"
            for d in documents
        )
        parts.append("Background documents (read with read_document(title)):\n" + listing)
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


def _writer_error_message(provider_display: str, raw_error: str) -> str:
    text = str(raw_error or "").strip()
    lowered = text.lower()
    if "cancelled by server shutdown" in lowered:
        return (
            "The server restarted while this request was running, so the model run was cancelled. "
            "Manuscript edits the tools had already applied before the restart are saved; "
            "send the request again to continue from there."
        )
    if "provider stream produced no text/final event" in lowered:
        return (
            f"{provider_display} opened the request but then produced no stream data for "
            f"{_WRITER_PROVIDER_IDLE_TIMEOUT_SEC}s. The provider connection stalled before "
            "any answer or tool call completed; nothing was written to the manuscript."
        )
    if "provider stream did not finalize" in lowered:
        return (
            f"{provider_display} stream sent partial data but did not finalize cleanly. "
            "The request was stopped before a complete answer could be saved."
        )
    if any(token in lowered for token in ("timeout", "connection", "network", "remoteprotocol", "readerror", "apierror")):
        return (
            f"{provider_display} connection failed while generating. This was a provider/network "
            "stream failure, not a manuscript tool failure; no assistant result was saved."
        )
    return f"{provider_display} request failed before completion: {text}"


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
        "Exact-substring search over the SAVED manuscript (whitespace/quote-insensitive fallback for multi-word "
        "queries). Returns matching passages with character offsets. Use SHORT distinctive queries — a name, or a "
        "2-6 word phrase; long quoted paragraphs usually fail. For continuity checks on earlier scenes beyond the "
        "context tail. Never search for prose you wrote this same turn: the tail in context and tool confirmations "
        "already reflect the saved draft. To pull broader context around a hit, follow up with read_manuscript."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short substring or phrase to find (a name, 2-6 words)."},
            "limit": {"type": "integer", "description": "Max passages to return (1-20).", "default": 8},
        },
        "required": ["query"],
    },
}

_READ_MANUSCRIPT_TOOL = {
    "name": "read_manuscript",
    "description": (
        "Read an exact slice of the saved manuscript by character offsets (offsets appear in search_manuscript "
        "results and the manuscript context header). Use it to pull full surrounding context before revising a "
        "passage, or to re-read an earlier scene. With no arguments it returns the last 5000 characters. "
        "Max 20000 characters per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start": {"type": "integer", "description": "Start character offset (0-based). Omit to read the tail."},
            "end": {"type": "integer", "description": "End character offset (exclusive). Defaults to start + 5000."},
        },
    },
}

_READ_DOCUMENT_TOOL = {
    "name": "read_document",
    "description": (
        "Read one background document (worldbuilding, character sheets, outline, research notes) in full by its "
        "title. The available documents are listed in your context. Use these as the authoritative reference for "
        "setting, character, and plot facts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Exact document title (case-insensitive)."},
        },
        "required": ["title"],
    },
}

_SEARCH_DOCUMENTS_TOOL = {
    "name": "search_documents",
    "description": (
        "Substring search across all background documents of this project (titles and contents). Returns document "
        "titles with matching snippets. Use short distinctive queries; then read_document for the full text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short substring to find (a name, a term)."},
            "limit": {"type": "integer", "description": "Max documents to return (1-20).", "default": 8},
        },
        "required": ["query"],
    },
}

_SAVE_DOCUMENT_TOOL = {
    "name": "save_document",
    "description": (
        "Create or fully overwrite a background document by title (worldbuilding notes, character sheet, outline, "
        "timeline). This does NOT touch the manuscript. Use it when the writer asks you to record or update notes, "
        "or to keep an agreed story bible current after major developments. Overwrites the whole document — read it "
        "first if you are updating."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Document title. Reusing an existing title overwrites that document."},
            "content": {"type": "string", "description": "Full document content (replaces any previous content)."},
            "kind": {"type": "string", "description": "Category label, e.g. character/setting/outline/research/note.", "default": "note"},
        },
        "required": ["title", "content"],
    },
}

_APPEND_MANUSCRIPT_TOOL = {
    "name": "append_to_manuscript",
    "description": (
        "Append new prose to the END of the manuscript — use this to continue the story. "
        "The text is added after the current ending; do not repeat existing text. "
        "This edits the saved manuscript directly."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Manuscript-ready prose to append."},
        },
        "required": ["text"],
    },
}

_REPLACE_MANUSCRIPT_TOOL = {
    "name": "replace_in_manuscript",
    "description": (
        "Revise a specific part of the manuscript: find an existing passage and replace it with new prose. "
        "This edits the saved manuscript directly. Copy 'find' from the saved manuscript (whitespace and quote-style "
        "differences are tolerated, wording is not) and make it unique — if you don't have the real text at hand, "
        "read_manuscript or search_manuscript gives it to you. Fails safely if 'find' is missing or ambiguous; "
        "then include more surrounding text and retry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "find": {"type": "string", "description": "The exact existing passage to replace (verbatim, unique in the manuscript)."},
            "replacement": {"type": "string", "description": "The new prose to put in its place."},
        },
        "required": ["find", "replacement"],
    },
}


_writer_tools_cache: dict[int, tuple[list[dict], dict]] = {}


def _build_writer_tools(project_id: int) -> tuple[list[dict], dict]:
    """Build the (tools, handlers) pair for a writer turn, bound to one project.

    - Handlers are closures over project_id so the model never supplies it.
    - Memoized per project: the tool dispatcher caches handler signatures by
      object identity, and per-request closures whose ids get recycled poisoned
      that cache (observed as spurious 'unexpected keyword argument' tool
      failures). Keeping one long-lived handler set per project avoids id reuse.
    """
    cached = _writer_tools_cache.get(project_id)
    if cached is not None:
        return cached

    async def _handle_search_manuscript(query: str, limit: int = 8) -> str:
        try:
            n = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            n = 8
        rows = await asyncio.to_thread(search_manuscript, project_id, query, n)
        if not rows:
            return (
                f"No manuscript matches for: {query}\n"
                "Search is exact-substring (with a whitespace/quote-insensitive fallback). "
                "Try a SHORTER distinctive phrase (a name, 2-4 words), or read_manuscript to read a region directly. "
                "Do not search for text you wrote this turn — it is already saved as confirmed by the tool results."
            )
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

    def _read_manuscript_slice(start: int | None, end: int | None) -> str:
        row = db_query_one(
            "SELECT length(body)::int AS total FROM writer_manuscripts WHERE project_id = %s",
            (project_id,),
        )
        total = int((row or {}).get("total") or 0)
        if total == 0:
            return "The manuscript is empty."
        try:
            if start is None:
                lo = max(0, total - 5000)
            else:
                lo = max(0, min(int(start), total))
            hi = total if end is None and start is None else min(int(end) if end is not None else lo + 5000, total)
        except (TypeError, ValueError):
            return "Invalid offsets: 'start' and 'end' must be integers."
        hi = max(lo, min(hi, lo + 20000))
        # substring is 1-based; slice in SQL so the full body never leaves Postgres.
        slice_row = db_query_one(
            "SELECT substring(body FROM %s FOR %s) AS piece FROM writer_manuscripts WHERE project_id = %s",
            (lo + 1, hi - lo, project_id),
        )
        piece = str((slice_row or {}).get("piece") or "")
        return f"Manuscript chars {lo}–{hi} of {total}:\n{piece}"

    async def _handle_read_manuscript(start: int | None = None, end: int | None = None) -> str:
        return await asyncio.to_thread(_read_manuscript_slice, start, end)

    async def _handle_read_document(title: str) -> str:
        doc = await asyncio.to_thread(get_document, project_id, None, title)
        if not doc:
            titles = [str(d.get("title")) for d in await asyncio.to_thread(list_documents, project_id)]
            listing = "; ".join(titles) if titles else "(no documents exist yet)"
            return f"No document titled {title!r}. Available documents: {listing}"
        content = str(doc.get("content") or "")
        suffix = ""
        if len(content) > 30000:
            content = content[:30000]
            suffix = "\n[truncated at 30000 chars]"
        return f"Document {doc.get('title')!r} (kind: {doc.get('kind')}, {doc.get('char_count')} chars):\n{content}{suffix}"

    async def _handle_search_documents(query: str, limit: int = 8) -> str:
        try:
            n = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            n = 8
        rows = await asyncio.to_thread(search_documents, project_id, query, n)
        if not rows:
            return f"No document matches for: {query}. Use short distinctive terms, or read_document by title."
        blocks = [
            f"{r.get('title')!r} (kind: {r.get('kind')}):\n…{str(r.get('snippet') or '').strip()}…"
            for r in rows
        ]
        return "\n\n".join(blocks)

    async def _handle_save_document(title: str, content: str, kind: str = "note") -> str:
        if not title or not title.strip():
            return "No title provided; nothing saved."
        doc = await asyncio.to_thread(save_document, project_id, title, content or "", kind or "note")
        if not doc:
            return "Save failed: project not found."
        return f"Saved document {doc.get('title')!r} (kind: {doc.get('kind')}, {doc.get('char_count')} chars)."

    async def _handle_append(text: str) -> str:
        if not text or not text.strip():
            return "No text provided; nothing appended."
        result = await asyncio.to_thread(append_manuscript, project_id, text, "tool append")
        if not result:
            return "Append failed: manuscript not found."
        return f"Appended. Manuscript is now {result.get('char_count')} characters."

    async def _handle_replace(find: str, replacement: str) -> str:
        result = await asyncio.to_thread(replace_manuscript_text, project_id, find, replacement, "tool replace")
        if result.get("ok"):
            return f"Replaced. Manuscript is now {result.get('char_count')} characters."
        return "Replace failed: " + result.get("message", "unknown error")

    tools: list[dict] = [
        _SEARCH_MANUSCRIPT_TOOL,
        _READ_MANUSCRIPT_TOOL,
        _APPEND_MANUSCRIPT_TOOL,
        _REPLACE_MANUSCRIPT_TOOL,
        _READ_DOCUMENT_TOOL,
        _SEARCH_DOCUMENTS_TOOL,
        _SAVE_DOCUMENT_TOOL,
    ]
    handlers: dict = {
        "search_manuscript": _handle_search_manuscript,
        "read_manuscript": _handle_read_manuscript,
        "append_to_manuscript": _handle_append,
        "replace_in_manuscript": _handle_replace,
        "read_document": _handle_read_document,
        "search_documents": _handle_search_documents,
        "save_document": _handle_save_document,
    }

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

    _writer_tools_cache[project_id] = (tools, handlers)
    return tools, handlers


class _WriterRun:
    """Server-side state of one background writer run, kept in _active_runs so a
    browser can detach and reattach to the live stream (page reload, network
    blip) without losing the run."""

    def __init__(self, *, project_id: int, run_id: str, user_message_id: int | None,
                 model: str, model_display: str) -> None:
        self.project_id = project_id
        self.run_id = run_id
        self.user_message_id = user_message_id
        self.model = model
        self.model_display = model_display
        self.subscribers: set[asyncio.Queue[str | None]] = set()
        self._text_parts: list[str] = []
        self.final_event: dict | None = None
        self.last_progress = asyncio.get_running_loop().time()

    def append_text(self, chunk: str) -> None:
        self._text_parts.append(chunk)

    def text_snapshot(self) -> str:
        return "".join(self._text_parts)

    async def broadcast(self, item: str | None) -> None:
        stale: list[asyncio.Queue[str | None]] = []
        for queue in tuple(self.subscribers):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self.subscribers.discard(queue)


_active_runs: dict[int, _WriterRun] = {}


def get_active_run(project_id: int) -> _WriterRun | None:
    return _active_runs.get(project_id)


async def _consume_run(
    run: _WriterRun,
    queue: asyncio.Queue[str | None],
    client_disconnected: Callable[[], Awaitable[bool]] | None,
) -> AsyncIterator[str]:
    """Relay a run's broadcast queue to one SSE consumer, with keepalive pings
    and idle cutoff. The background run outlives any consumer."""
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                if client_disconnected is not None and await client_disconnected():
                    logger.warning(
                        "writer stream subscriber disconnected project_id=%s reason=request.is_disconnected",
                        run.project_id,
                    )
                    return
                idle_for = asyncio.get_running_loop().time() - run.last_progress
                if idle_for >= _WRITER_IDLE_TIMEOUT_SEC:
                    logger.warning(
                        "writer stream idle project_id=%s idle_for=%ss; background run remains active",
                        run.project_id,
                        int(idle_for),
                    )
                    yield _sse({
                        "type": "error",
                        "content": (
                            f"The browser connection has not received model progress for {int(idle_for)}s. "
                            "The server-side writer run is still active; this page will reload saved results."
                        ),
                    })
                    return
                yield _sse({"type": "ping"})
                continue
            if item is None:
                return
            yield item
    except asyncio.CancelledError:
        logger.warning(
            "writer stream subscriber disconnected project_id=%s reason=streaming_response_cancelled",
            run.project_id,
        )
        return
    finally:
        run.subscribers.discard(queue)


async def stream_active_run(
    *,
    project_id: int,
    client_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    """Reattach to a live background writer run (e.g. after a page reload or a
    dropped stream). Replays the text streamed so far, then follows live."""
    run = _active_runs.get(project_id)
    if run is None:
        yield _sse({"type": "no_active_run"})
        return
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    run.subscribers.add(queue)
    yield _sse({
        "type": "run_status",
        "run_id": run.run_id,
        "user_message_id": run.user_message_id,
        "model": run.model,
        "model_display": run.model_display,
        "text_snapshot": run.text_snapshot(),
    })
    if run.final_event is not None:
        run.subscribers.discard(queue)
        yield _sse(run.final_event)
        return
    async for item in _consume_run(run, queue, client_disconnected):
        yield item


async def stream_writer_reply(
    *,
    project_id: int,
    prompt: str,
    selection_start: int | None = None,
    selection_end: int | None = None,
    model_choice: str | None = None,
    client_disconnected: Callable[[], Awaitable[bool]] | None = None,
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

    run = _WriterRun(
        project_id=project_id,
        run_id=uuid.uuid4().hex,
        user_message_id=user_row.get("id"),
        model=writer_model,
        model_display=writer_display,
    )
    subscriber_queue: asyncio.Queue[str | None] = asyncio.Queue()
    run.subscribers.add(subscriber_queue)
    answer_holder: list[str] = []
    error_holder: list[str] = []
    budget_tracker: dict = {}
    persisted = False

    writer_tools, writer_handlers = _build_writer_tools(project_id)

    def persist_result() -> dict:
        nonlocal persisted
        if persisted:
            return {"type": "error", "content": "Writer result was already persisted."}
        persisted = True
        if error_holder:
            error_text = _writer_error_message(writer_display, error_holder[0])
            assistant_row = _insert_message(
                project_id=project_id,
                role="assistant",
                content=f"<commentary>\n{error_text}\n</commentary>",
                request_kind=request_kind,
                model=writer_model,
                stop_reason="error",
                usage=budget_tracker.get("usage") or {},
                cost_usd=budget_tracker.get("total_cost"),
            )
            _touch_project(project_id)
            return {"type": "error", "message_id": assistant_row.get("id"), "content": error_text}

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
        return {
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

    async def on_progress(event: str, detail: str):
        run.last_progress = asyncio.get_running_loop().time()
        if event == "text_delta" and detail:
            run.append_text(detail)
            await run.broadcast(_sse({"type": "text_delta", "content": detail}))
        elif event == "budget" and detail:
            await run.broadcast(_sse({"type": "budget", "content": detail}))
        elif event in ("tool_call", "tool_result") and detail:
            await run.broadcast(_sse({"type": "tool", "content": detail}))
        elif event == "provider_retry" and detail:
            await run.broadcast(_sse({"type": "provider_retry", "content": detail}))

    async def run_llm() -> None:
        try:
            system_blocks = await asyncio.to_thread(
                _build_system_blocks, project, project_id, selection_start, selection_end
            )
            with caller_scope(CallerContext(interface="system", agent_name="writer", is_owner=True)):
                result = await chat_with_tools(
                    model_messages,
                    client=writer_client,
                    model=writer_model,
                    tools=writer_tools,
                    tool_handlers=writer_handlers,
                    system_prompt=system_blocks,
                    max_rounds=_WRITER_MAX_ROUNDS,
                    max_tokens=WRITER_DEFAULT_MAX_TOKENS,
                    budget_usd=100.0,
                    budget_tracker=budget_tracker,
                    on_progress=on_progress,
                    agent_name="writer",
                    provider_idle_timeout_sec=_WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
                    **model_extra,
                )
            answer_holder.append(result)
        except asyncio.CancelledError:
            # Server shutdown/restart while the run was in flight. Persist an
            # explanatory message in the finally block, then re-raise.
            error_holder.append("writer run cancelled by server shutdown")
            raise
        except Exception as exc:
            logger.exception("writer model request failed project_id=%s", project_id)
            error_holder.append(str(exc))
        finally:
            try:
                event = persist_result()
                run.final_event = event
                await run.broadcast(_sse(event))
                logger.info(
                    "writer finalized background run project_id=%s event=%s message_id=%s connected_subscribers=%s",
                    project_id,
                    event.get("type"),
                    event.get("message_id"),
                    len(run.subscribers),
                )
            except Exception:
                logger.exception("writer failed to persist background run project_id=%s", project_id)
                await run.broadcast(_sse({
                    "type": "error",
                    "content": "Writer finished but failed to save the result. Check server logs.",
                }))
            finally:
                await run.broadcast(None)
                if _active_runs.get(project_id) is run:
                    del _active_runs[project_id]

    # Register and start the run BEFORE the first yield: a client that drops
    # right after connecting must not prevent the background run from starting.
    _active_runs[project_id] = run
    task = asyncio.create_task(run_llm())
    _writer_background_tasks.add(task)
    task.add_done_callback(_writer_background_tasks.discard)
    yield _sse({"type": "user_saved", "message_id": user_row.get("id"), "run_id": run.run_id})
    async for item in _consume_run(run, subscriber_queue, client_disconnected):
        yield item
