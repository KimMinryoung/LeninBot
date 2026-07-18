"""Writer persistence: schema, projects, manuscripts, messages, settings.

All Postgres access for the fiction workspace lives here (background documents
have their own module, writer.documents). Functions are synchronous psycopg2
calls; async callers wrap them in asyncio.to_thread.
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any

from psycopg2.extras import RealDictCursor, execute_values

# Writer tables live in the dedicated local database (db.get_writer_conn);
# the remote main DB's ~560ms RTT made every manuscript operation cost seconds.
from db import get_writer_conn as get_conn
from db import writer_execute as db_execute
from db import writer_query as db_query
from db import writer_query_one as db_query_one

from writer.matching import find_normalized_matches

logger = logging.getLogger(__name__)

_MANUSCRIPT_CHUNK_SIZE = 3500
_MANUSCRIPT_CHUNK_OVERLAP = 300


# ── Schema ───────────────────────────────────────────────────────────

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
    # Web publication flags for existing installs. The slug is minted on first
    # publish and kept on unpublish so a re-enabled project gets the same URL.
    db_execute(
        "ALTER TABLE writer_projects ADD COLUMN IF NOT EXISTS is_public BOOLEAN NOT NULL DEFAULT FALSE"
    )
    db_execute("ALTER TABLE writer_projects ADD COLUMN IF NOT EXISTS public_slug TEXT")
    db_execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS writer_projects_public_slug_key "
        "ON writer_projects(public_slug) WHERE public_slug IS NOT NULL"
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
        """CREATE TABLE IF NOT EXISTS writer_settings (
               key TEXT PRIMARY KEY,
               value JSONB NOT NULL DEFAULT '{}'::jsonb,
               updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
           )"""
    )
    db_execute(
        """CREATE TABLE IF NOT EXISTS writer_documents (
               id BIGSERIAL PRIMARY KEY,
               -- NULL project_id = shared document, visible to every project.
               project_id BIGINT REFERENCES writer_projects(id) ON DELETE CASCADE,
               title TEXT NOT NULL,
               kind TEXT NOT NULL DEFAULT 'note',
               content TEXT NOT NULL DEFAULT '',
               created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               UNIQUE(project_id, title)
           )"""
    )
    # Existing installs created project_id as NOT NULL; shared documents need
    # it nullable. DROP NOT NULL is idempotent (no-op when already nullable).
    db_execute("ALTER TABLE writer_documents ALTER COLUMN project_id DROP NOT NULL")
    # Shared documents (project_id IS NULL) need their own title uniqueness:
    # the UNIQUE(project_id, title) constraint treats NULLs as distinct. This
    # partial index is also the ON CONFLICT arbiter for the shared upsert.
    db_execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS writer_documents_shared_title_key "
        "ON writer_documents(title) WHERE project_id IS NULL"
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


# ── Settings ─────────────────────────────────────────────────────────

def get_writer_setting(key: str, default: Any = None) -> Any:
    row = db_query_one("SELECT value FROM writer_settings WHERE key = %s", (key,))
    if not row:
        return default
    return row.get("value", default)


def set_writer_setting(key: str, value: Any) -> None:
    db_execute(
        """INSERT INTO writer_settings (key, value)
             VALUES (%s, %s::jsonb)
        ON CONFLICT (key)
          DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
        (key, json.dumps(value, ensure_ascii=False)),
    )


# ── Projects ─────────────────────────────────────────────────────────

def list_projects(limit: int = 100, status: str = "active") -> list[dict]:
    return db_query(
        """SELECT p.id, p.title, p.premise, p.style_notes, p.status,
                  p.is_public, p.public_slug,
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
            WHERE p.status = %s
            ORDER BY p.updated_at DESC, p.id DESC
            LIMIT %s""",
        (status, limit),
    )


def create_project(title: str, premise: str = "", style_notes: str = "") -> dict:
    project = db_query_one(
        """INSERT INTO writer_projects (title, premise, style_notes)
             VALUES (%s, %s, %s)
          RETURNING id, title, premise, style_notes, status, is_public, public_slug,
                    created_at, updated_at""",
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
        RETURNING id, title, premise, style_notes, status, is_public, public_slug,
                  created_at, updated_at""",
        (title.strip(), premise.strip(), style_notes.strip(), project_id),
    )


def get_project(project_id: int) -> dict | None:
    project = db_query_one(
        """SELECT p.id, p.title, p.premise, p.style_notes, p.status,
                  p.is_public, p.public_slug,
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


def trash_project(project_id: int) -> bool:
    """Soft delete: the project moves to the trash (status='deleted') with all
    its data intact, and disappears from the active list. Reversible via
    restore_project; permanently removed only by delete_project."""
    row = db_query_one(
        """UPDATE writer_projects SET status = 'deleted', updated_at = NOW()
            WHERE id = %s AND status <> 'deleted' RETURNING id""",
        (project_id,),
    )
    return bool(row)


def restore_project(project_id: int) -> bool:
    row = db_query_one(
        """UPDATE writer_projects SET status = 'active', updated_at = NOW()
            WHERE id = %s AND status = 'deleted' RETURNING id""",
        (project_id,),
    )
    return bool(row)


def set_project_public(project_id: int, is_public: bool) -> dict | None:
    """Toggle web publication. The slug is minted on first publish and kept on
    unpublish, so re-enabling restores the same shared URL."""
    project = db_query_one(
        "SELECT id, public_slug FROM writer_projects WHERE id = %s", (project_id,)
    )
    if not project:
        return None
    if is_public and not project.get("public_slug"):
        for _ in range(5):  # retry on the (negligible) chance of slug collision
            try:
                db_execute(
                    "UPDATE writer_projects SET public_slug = %s WHERE id = %s AND public_slug IS NULL",
                    (secrets.token_urlsafe(6), project_id),
                )
                break
            except Exception:  # noqa: BLE001 — unique-index race, roll a new slug
                continue
    return db_query_one(
        """UPDATE writer_projects
              SET is_public = %s, updated_at = NOW()
            WHERE id = %s
        RETURNING id, title, premise, style_notes, status, is_public, public_slug,
                  created_at, updated_at""",
        (is_public, project_id),
    )


def get_public_manuscript(slug: str) -> dict | None:
    """Anonymous web read: title + manuscript for a published, active project.
    Returns None when the slug is unknown, unpublished, or trashed."""
    return db_query_one(
        """SELECT p.id, p.title, p.public_slug,
                  COALESCE(w.body, '') AS body,
                  COALESCE(length(w.body), 0)::int AS char_count,
                  p.created_at,
                  COALESCE(w.updated_at, p.updated_at) AS updated_at
             FROM writer_projects p
        LEFT JOIN writer_manuscripts w ON w.project_id = p.id
            WHERE p.public_slug = %s
              AND p.is_public
              AND p.status = 'active'""",
        (slug,),
    )


def delete_project(project_id: int) -> bool:
    """Hard delete: permanently removes the project and (via FK cascade) its
    manuscript, chunks, revisions, messages, and documents."""
    row = db_query_one(
        "DELETE FROM writer_projects WHERE id = %s RETURNING id",
        (project_id,),
    )
    from writer import tools as writer_tools  # lazy: avoids import cycle

    writer_tools.invalidate_project_tools(project_id)
    return bool(row)


# ── Manuscript internals ─────────────────────────────────────────────

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


def read_manuscript_slice(project_id: int, start: int | None, end: int | None) -> str:
    """Bounded region read used by the read_manuscript tool. Slices in SQL so
    the full body never leaves Postgres; no arguments = the last 5000 chars."""
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
    # substring is 1-based.
    slice_row = db_query_one(
        "SELECT substring(body FROM %s FOR %s) AS piece FROM writer_manuscripts WHERE project_id = %s",
        (lo + 1, hi - lo, project_id),
    )
    piece = str((slice_row or {}).get("piece") or "")
    return f"Manuscript chars {lo}–{hi} of {total}:\n{piece}"


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


# ── Manuscript operations ────────────────────────────────────────────

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
    result = _write_manuscript(project_id, current + separator + addition, action="append", note=note)
    if result:
        result = dict(result)
        result["edit_start"] = len(current) + len(separator)
        result["edit_end"] = len(current) + len(separator) + len(addition)
    return result


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
        spans = find_normalized_matches(current, find)
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
    return {
        "ok": True,
        "message": f"Replaced (matched {matched_how}).",
        "char_count": (result or {}).get("char_count"),
        "edit_start": span[0],
        "edit_end": span[0] + len(replacement),
        "delta": len(replacement) - (span[1] - span[0]),
    }


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
        for start, end in find_normalized_matches(body, needle, max_matches=limit):
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


# ── Messages ─────────────────────────────────────────────────────────

def insert_message(
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


def touch_project(project_id: int) -> None:
    db_execute("UPDATE writer_projects SET updated_at = NOW() WHERE id = %s", (project_id,))


def recent_messages(project_id: int, limit: int) -> list[dict]:
    """Newest-first (role, content) rows for prompt-context assembly."""
    return db_query(
        """SELECT role, content
             FROM writer_messages
            WHERE project_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s""",
        (project_id, limit),
    )


def total_message_chars(project_id: int) -> int:
    """Total content chars across the project's whole conversation — the
    anchor for the quantized history window (writer.prompts.messages_for_model)."""
    row = db_query_one(
        """SELECT COALESCE(SUM(CHAR_LENGTH(content)), 0) AS n
             FROM writer_messages
            WHERE project_id = %s AND role IN ('user', 'assistant')""",
        (project_id,),
    )
    return int(row["n"]) if row else 0
