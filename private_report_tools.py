"""private_report_tools.py — Admin-only private report storage.

Private reports are Markdown documents intended for Cyber-Lenin and the
Telegram/admin operator only. They are stored outside the public
research_documents/static_pages publishing tables and are not exposed through
the public web chat toolset.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from db import execute as db_execute, query as db_query, query_one as db_query_one
import research_store

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")
_ready = False


def _validate_slug(slug: str) -> str:
    value = (slug or "").strip()
    if not value:
        raise ValueError("slug is required")
    if not _SLUG_RE.fullmatch(value):
        raise ValueError("slug must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_summary(markdown: str) -> str | None:
    return research_store.extract_excerpt(markdown)


def ensure_private_reports_table() -> None:
    global _ready
    if _ready:
        return
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS private_reports (
          id BIGSERIAL PRIMARY KEY,
          slug TEXT NOT NULL UNIQUE,
          title TEXT NOT NULL,
          markdown TEXT NOT NULL,
          summary TEXT,
          content_sha256 TEXT NOT NULL,
          source_task_id BIGINT,
          published_research_id BIGINT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    for ddl in (
        "ALTER TABLE private_reports ADD COLUMN IF NOT EXISTS summary TEXT",
        "ALTER TABLE private_reports ADD COLUMN IF NOT EXISTS source_task_id BIGINT",
        "ALTER TABLE private_reports ADD COLUMN IF NOT EXISTS published_research_id BIGINT",
        "ALTER TABLE private_reports ADD COLUMN IF NOT EXISTS content_sha256 TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE private_reports ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "ALTER TABLE private_reports ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    ):
        db_execute(ddl)
    db_execute(
        """
        CREATE INDEX IF NOT EXISTS private_reports_updated_at_idx
        ON private_reports(updated_at DESC)
        """
    )
    _ready = True


def save_private_report_sync(
    *,
    title: str,
    slug: str,
    markdown_body: str,
    source_task_id: int | None = None,
) -> dict:
    ensure_private_reports_table()
    clean_title = (title or "").strip()
    clean_slug = _validate_slug(slug)
    markdown = (markdown_body or "").strip()
    if not clean_title:
        raise ValueError("title is required")
    if not markdown:
        raise ValueError("markdown_body is required")

    row = db_query_one(
        """
        INSERT INTO private_reports (
          slug, title, markdown, summary, content_sha256, source_task_id,
          created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (slug) DO UPDATE SET
          title = EXCLUDED.title,
          markdown = EXCLUDED.markdown,
          summary = EXCLUDED.summary,
          content_sha256 = EXCLUDED.content_sha256,
          source_task_id = COALESCE(EXCLUDED.source_task_id, private_reports.source_task_id),
          updated_at = NOW()
        RETURNING *
        """,
        (
            clean_slug,
            clean_title,
            markdown,
            _extract_summary(markdown),
            _sha256_text(markdown),
            source_task_id,
        ),
    )
    return dict(row)


def get_private_report_sync(report_id: int | None = None, slug: str | None = None) -> dict | None:
    ensure_private_reports_table()
    if report_id is not None:
        row = db_query_one("SELECT * FROM private_reports WHERE id = %s LIMIT 1", (int(report_id),))
        return dict(row) if row else None
    clean_slug = _validate_slug(slug or "")
    row = db_query_one("SELECT * FROM private_reports WHERE slug = %s LIMIT 1", (clean_slug,))
    return dict(row) if row else None


def list_private_reports_sync(limit: int = 20, keyword: str | None = None) -> list[dict]:
    ensure_private_reports_table()
    clauses: list[str] = []
    params: list[Any] = []
    if keyword:
        clauses.append("(title ILIKE %s OR slug ILIKE %s OR summary ILIKE %s OR markdown ILIKE %s)")
        q = f"%{keyword}%"
        params.extend([q, q, q, q])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(min(max(int(limit or 20), 1), 100))
    return db_query(
        f"""
        SELECT id, slug, title, summary, source_task_id, published_research_id,
               content_sha256, created_at, updated_at
          FROM private_reports
          {where}
         ORDER BY updated_at DESC, id DESC
         LIMIT %s
        """,
        tuple(params),
    )


def _public_url(slug: str) -> str:
    return f"https://cyber-lenin.com/reports/research/{slug}"


def _as_public_markdown(title: str, markdown: str) -> tuple[str, str]:
    body = (markdown or "").strip()
    if not body:
        raise ValueError("body is required")
    extracted_title = research_store.extract_title(body, title)
    if body.lstrip().startswith("# "):
        return extracted_title, body
    return extracted_title, f"# {extracted_title}\n\n{body}\n"


def publish_private_report_sync(
    *,
    slug: str,
    body: str | None = None,
    title: str | None = None,
) -> dict:
    ensure_private_reports_table()
    clean_slug = _validate_slug(slug)
    private = get_private_report_sync(slug=clean_slug)
    if not private:
        raise ValueError(f"no private report found for slug={clean_slug!r}")

    markdown_source = body if body is not None and body.strip() else private["markdown"]
    public_title, markdown = _as_public_markdown((title or private["title"] or "").strip(), markdown_source)
    filename = f"{clean_slug}.md"
    row, is_overwrite = research_store.upsert_document(
        filename=filename,
        title=public_title,
        markdown=markdown,
        summary=research_store.extract_excerpt(markdown),
        status="public",
        updated_at=datetime.now(timezone.utc),
    )
    db_query_one(
        """
        UPDATE private_reports
           SET published_research_id = %s, updated_at = NOW()
         WHERE slug = %s
         RETURNING id
        """,
        (row["id"], clean_slug),
    )
    return {
        "private_report": private,
        "research_document": row,
        "is_overwrite": is_overwrite,
        "public_url": _public_url(clean_slug),
        "markdown": markdown,
    }


SAVE_PRIVATE_REPORT_TOOL = {
    "name": "save_private_report",
    "description": (
        "Save or overwrite an admin-only private Markdown report. The report is "
        "not exposed on the public website. Use this for sensitive analysis meant "
        "for Cyber-Lenin and the Telegram/admin operator only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Private report title."},
            "slug": {"type": "string", "description": "Stable ASCII slug, e.g. 20260502_korea_labor_dual_structure."},
            "markdown_body": {"type": "string", "description": "Full Markdown body to store privately."},
            "source_task_id": {"type": "integer", "description": "Optional originating telegram_tasks id."},
        },
        "required": ["title", "slug", "markdown_body"],
    },
}

READ_PRIVATE_REPORT_TOOL = {
    "name": "read_private_report",
    "description": "Read one admin-only private report by slug or numeric id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Private report slug."},
            "report_id": {"type": "integer", "description": "Private report id."},
        },
        "required": [],
    },
}

LIST_PRIVATE_REPORTS_TOOL = {
    "name": "list_private_reports",
    "description": "List admin-only private reports. Returns metadata and summaries, not full bodies.",
    "input_schema": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum rows, 1-100.", "default": 20},
            "keyword": {"type": "string", "description": "Optional title/slug/body keyword filter."},
        },
        "required": [],
    },
}

PUBLISH_PRIVATE_REPORT_TOOL = {
    "name": "publish_private_report",
    "description": (
        "Publish a saved private report into the public research_documents table. "
        "Pass slug and optionally body to override the private Markdown at publish time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Private report slug; also becomes the public research slug."},
            "body": {"type": "string", "description": "Optional replacement Markdown body for publication."},
            "title": {"type": "string", "description": "Optional replacement public title."},
            "broadcast": {
                "type": "boolean",
                "description": "Whether to broadcast the newly public report to the Telegram channel. Default true.",
                "default": True,
            },
        },
        "required": ["slug"],
    },
}


def _format_ts(ts: Any) -> str:
    if hasattr(ts, "astimezone"):
        return ts.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")
    return str(ts or "?")


async def _exec_save_private_report(
    title: str,
    slug: str,
    markdown_body: str,
    source_task_id: int | None = None,
) -> str:
    try:
        row = await asyncio.to_thread(
            save_private_report_sync,
            title=title,
            slug=slug,
            markdown_body=markdown_body,
            source_task_id=source_task_id,
        )
    except Exception as e:
        logger.warning("save_private_report failed: %s", e)
        return f"Error: failed to save private report: {type(e).__name__}: {e}"
    return (
        f"Saved private report: id={row['id']} slug={row['slug']}\n"
        f"title: {row['title']}\n"
        f"sha256={row['content_sha256'][:12]} updated={_format_ts(row.get('updated_at'))}"
    )


async def _exec_read_private_report(slug: str | None = None, report_id: int | None = None) -> str:
    if report_id is None and not slug:
        return "Error: provide slug or report_id."
    try:
        row = await asyncio.to_thread(get_private_report_sync, report_id=report_id, slug=slug)
    except Exception as e:
        return f"Error: failed to read private report: {type(e).__name__}: {e}"
    if not row:
        return "No private report found."
    markdown = row.get("markdown") or ""
    return (
        f"=== PRIVATE REPORT: {row['slug']} ===\n"
        f"id={row['id']} title={row.get('title') or ''}\n"
        f"created={_format_ts(row.get('created_at'))} updated={_format_ts(row.get('updated_at'))}\n"
        f"published_research_id={row.get('published_research_id') or ''}\n"
        f"sha256={row.get('content_sha256', '')[:12]}\n\n"
        f"{markdown}"
    )


async def _exec_list_private_reports(limit: int = 20, keyword: str | None = None) -> str:
    try:
        rows = await asyncio.to_thread(list_private_reports_sync, limit=limit, keyword=keyword)
    except Exception as e:
        return f"Error: failed to list private reports: {type(e).__name__}: {e}"
    if not rows:
        return "No private reports found."
    lines = ["=== PRIVATE REPORTS ===", "Use read_private_report(slug='<slug>') for full detail."]
    for row in rows:
        summary = (row.get("summary") or "").replace("\n", " ")[:240]
        lines.append(
            f"- id={row['id']} slug={row['slug']} updated={_format_ts(row.get('updated_at'))}\n"
            f"  title: {row.get('title') or ''}\n"
            f"  summary: {summary}"
        )
    return "\n".join(lines)


async def _exec_publish_private_report(
    slug: str,
    body: str | None = None,
    title: str | None = None,
    broadcast: bool = True,
) -> str:
    try:
        result = await asyncio.to_thread(publish_private_report_sync, slug=slug, body=body, title=title)
    except Exception as e:
        return f"Error: failed to publish private report: {type(e).__name__}: {e}"

    row = result["research_document"]
    public_url = result["public_url"]
    cache_note = ""
    try:
        from research_tools import _invalidate_cache_sync

        cache = await asyncio.to_thread(_invalidate_cache_sync, row["filename"])
        cache_note = f"; cache invalidated ({cache.get('deleted', 0)} key(s))" if cache.get("ok") else f"; cache invalidation failed ({cache.get('reason')})"
    except Exception as e:
        cache_note = f"; cache invalidation failed ({e})"

    broadcast_note = ""
    if broadcast:
        try:
            from telegram.channel_broadcast import maybe_broadcast_autonomous_publication
            from publication_records import record_publication_broadcast_sync

            br = await maybe_broadcast_autonomous_publication(
                title=row["title"],
                url=public_url,
                body=result["markdown"],
                source="private report publication",
            )
            if br.ok:
                broadcast_note = f"\nTelegram channel broadcast: sent ({br.sent_count})"
                if getattr(br, "message_ids", None):
                    await asyncio.to_thread(
                        record_publication_broadcast_sync,
                        slug=row["slug"],
                        public_url=public_url,
                        channel_message_ids=br.message_ids,
                        source="publish_private_report",
                    )
            else:
                broadcast_note = f"\nTelegram channel broadcast skipped/failed: {br.message}"
        except Exception as e:
            broadcast_note = f"\nTelegram channel broadcast failed: {e}"

    status = "Overwrote public research document" if result["is_overwrite"] else "Published public research document"
    return (
        f"{status}: {row['filename']}\n"
        f"Private report slug: {slug}\n"
        f"Storage: research_documents id={row['id']} sha256={row['content_sha256'][:12]}\n"
        f"Public URL: {public_url}{cache_note}"
        f"{broadcast_note}"
    )


PRIVATE_REPORT_TOOLS = [
    SAVE_PRIVATE_REPORT_TOOL,
    READ_PRIVATE_REPORT_TOOL,
    LIST_PRIVATE_REPORTS_TOOL,
    PUBLISH_PRIVATE_REPORT_TOOL,
]

PRIVATE_REPORT_TOOL_HANDLERS = {
    "save_private_report": _exec_save_private_report,
    "read_private_report": _exec_read_private_report,
    "list_private_reports": _exec_list_private_reports,
    "publish_private_report": _exec_publish_private_report,
}
