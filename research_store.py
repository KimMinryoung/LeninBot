"""DB-backed storage for public research markdown documents.

The Markdown body is the source of truth.  Files under research/ and
output/research/ are legacy import/fallback sources only.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db import execute as db_execute, query as db_query, query_one as db_query_one

_ready = False


def ensure_research_table() -> None:
    global _ready
    if _ready:
        return
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS research_documents (
          id BIGSERIAL PRIMARY KEY,
          slug TEXT NOT NULL UNIQUE,
          filename TEXT NOT NULL UNIQUE,
          title TEXT NOT NULL,
          markdown TEXT NOT NULL,
          summary TEXT,
          lang TEXT NOT NULL DEFAULT 'ko',
          markdown_en TEXT,
          title_en TEXT,
          summary_en TEXT,
          status TEXT NOT NULL DEFAULT 'public',
          tags JSONB NOT NULL DEFAULT '[]'::jsonb,
          source_task_id BIGINT,
          content_sha256 TEXT NOT NULL,
          published_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    # Keep startup idempotent for older manually-created tables.
    for ddl in (
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS summary TEXT",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS lang TEXT NOT NULL DEFAULT 'ko'",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS markdown_en TEXT",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS title_en TEXT",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS summary_en TEXT",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'public'",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '[]'::jsonb",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS source_task_id BIGINT",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS content_sha256 TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "ALTER TABLE research_documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    ):
        db_execute(ddl)
    db_execute(
        """
        CREATE INDEX IF NOT EXISTS research_documents_status_updated_idx
        ON research_documents(status, updated_at DESC)
        """
    )
    _ready = True


def storage_filename(filename: str) -> str:
    return filename if filename.endswith(".md") else f"{filename}.md"


def public_slug(filename: str) -> str:
    fname = storage_filename(filename)
    return fname[:-3] if fname.endswith(".md") else fname


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_title(markdown: str, fallback: str = "Research") -> str:
    for line in markdown.splitlines()[:80]:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            if title:
                return title
    return fallback


_MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_MD_CODE_RE = re.compile(r"`([^`]+)`")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_META_LINE_RE = re.compile(r"^\s*(?:\*\*)?(?:작성자|작성일|Author|Date)(?:\*\*)?\s*:.*$", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_MD_LIST_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+")


def extract_excerpt(markdown: str, max_chars: int = 300) -> str | None:
    lines = markdown.replace("\r\n", "\n").split("\n")[:100]
    body_start = 0
    for i, raw in enumerate(lines[:15]):
        stripped = raw.strip()
        if stripped and all(c == "-" for c in stripped) and len(stripped) >= 3:
            body_start = i + 1
            break

    parts: list[str] = []
    for raw in lines[body_start:]:
        line = raw.strip()
        if not line or line.startswith("# ") or line.startswith(">"):
            continue
        if _MD_META_LINE_RE.match(line):
            continue
        if set(line) <= {"-", "="} and len(line) >= 3:
            continue
        line = _MD_HEADING_RE.sub("", line)
        line = _MD_LIST_RE.sub("", line)
        line = _MD_IMAGE_RE.sub("", line)
        line = _MD_LINK_RE.sub(r"\1", line)
        line = _MD_BOLD_RE.sub(r"\1", line)
        line = _MD_ITALIC_RE.sub(r"\1", line)
        line = _MD_CODE_RE.sub(r"\1", line).strip()
        if line:
            parts.append(line)
        if sum(len(p) for p in parts) + len(parts) - 1 >= max_chars:
            break

    text = " ".join(parts).strip()
    if not text:
        return None
    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


def _row_to_dict(row: dict | None) -> dict | None:
    return dict(row) if row else None


def get_document(filename_or_slug: str, *, include_private: bool = False) -> dict | None:
    ensure_research_table()
    fname = storage_filename(filename_or_slug)
    slug = public_slug(filename_or_slug)
    where_status = "" if include_private else "AND status = 'public'"
    return _row_to_dict(db_query_one(
        f"""
        SELECT * FROM research_documents
        WHERE (filename = %s OR slug = %s) {where_status}
        LIMIT 1
        """,
        (fname, slug),
    ))


def list_documents(*, include_private: bool = False) -> list[dict]:
    ensure_research_table()
    where_status = "" if include_private else "WHERE status = 'public'"
    return db_query(
        f"""
        SELECT * FROM research_documents
        {where_status}
        ORDER BY updated_at DESC, id DESC
        """
    )


def upsert_document(
    *,
    filename: str,
    title: str,
    markdown: str,
    summary: str | None = None,
    status: str = "public",
    source_task_id: int | None = None,
    markdown_en: str | None = None,
    title_en: str | None = None,
    summary_en: str | None = None,
    published_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> tuple[dict, bool]:
    ensure_research_table()
    fname = storage_filename(filename)
    slug = public_slug(fname)
    existing = get_document(fname, include_private=True)
    content_hash = sha256_text(markdown)
    row = db_query_one(
        """
        INSERT INTO research_documents (
          slug, filename, title, markdown, summary, status, source_task_id,
          markdown_en, title_en, summary_en, content_sha256, published_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()), COALESCE(%s, NOW()))
        ON CONFLICT (filename) DO UPDATE SET
          slug = EXCLUDED.slug,
          title = EXCLUDED.title,
          markdown = EXCLUDED.markdown,
          summary = EXCLUDED.summary,
          status = EXCLUDED.status,
          source_task_id = COALESCE(EXCLUDED.source_task_id, research_documents.source_task_id),
          markdown_en = COALESCE(EXCLUDED.markdown_en, research_documents.markdown_en),
          title_en = COALESCE(EXCLUDED.title_en, research_documents.title_en),
          summary_en = COALESCE(EXCLUDED.summary_en, research_documents.summary_en),
          content_sha256 = EXCLUDED.content_sha256,
          published_at = EXCLUDED.published_at,
          updated_at = EXCLUDED.updated_at
        RETURNING *
        """,
        (
            slug, fname, title, markdown, summary, status, source_task_id,
            markdown_en, title_en, summary_en, content_hash, published_at, updated_at,
        ),
    )
    return dict(row), existing is not None


def set_status(filename_or_slug: str, status: str) -> dict | None:
    ensure_research_table()
    fname = storage_filename(filename_or_slug)
    slug = public_slug(filename_or_slug)
    row = db_query_one(
        """
        UPDATE research_documents
        SET status = %s, updated_at = NOW()
        WHERE filename = %s OR slug = %s
        RETURNING *
        """,
        (status, fname, slug),
    )
    return _row_to_dict(row)


def timestamp_seconds(value: Any) -> float:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return 0.0


def load_markdown_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")
