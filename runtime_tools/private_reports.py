"""runtime_tools.private_reports — Admin-only private research document storage.

Private research documents are Markdown research documents intended for
Cyber-Lenin and the Telegram/admin operator only. They share the research
document storage model but remain private until explicitly published.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from db import query as db_query, query_one as db_query_one
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
    research_store.ensure_research_table()
    _ready = True


def _private_row_to_compat(row: dict | None) -> dict | None:
    if not row:
        return None
    data = dict(row)
    if "created_at" not in data:
        data["created_at"] = data.get("published_at")
    data["published_research_id"] = None
    return data


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

    row, _ = research_store.upsert_document(
        filename=f"{clean_slug}.md",
        title=clean_title,
        markdown=markdown,
        summary=_extract_summary(markdown),
        status="private",
        source_task_id=source_task_id,
        updated_at=datetime.now(timezone.utc),
    )
    return _private_row_to_compat(row)


def get_private_report_sync(report_id: int | None = None, slug: str | None = None) -> dict | None:
    ensure_private_reports_table()
    if report_id is not None:
        row = db_query_one(
            "SELECT * FROM research_documents WHERE id = %s AND status = 'private' LIMIT 1",
            (int(report_id),),
        )
        return _private_row_to_compat(row)
    clean_slug = _validate_slug(slug or "")
    row = db_query_one(
        "SELECT * FROM research_documents WHERE slug = %s AND status = 'private' LIMIT 1",
        (clean_slug,),
    )
    return _private_row_to_compat(row)


def list_private_reports_sync(limit: int = 20, keyword: str | None = None) -> list[dict]:
    ensure_private_reports_table()
    clauses: list[str] = []
    params: list[Any] = []
    if keyword:
        clauses.append("(title ILIKE %s OR slug ILIKE %s OR summary ILIKE %s OR markdown ILIKE %s)")
        q = f"%{keyword}%"
        params.extend([q, q, q, q])
    where_parts = ["status = 'private'", *clauses]
    where = "WHERE " + " AND ".join(where_parts)
    params.append(min(max(int(limit or 20), 1), 100))
    rows = db_query(
        f"""
        SELECT id, slug, title, summary, source_task_id,
               NULL::BIGINT AS published_research_id,
               content_sha256, published_at AS created_at, updated_at
          FROM research_documents
          {where}
         ORDER BY updated_at DESC, id DESC
         LIMIT %s
        """,
        tuple(params),
    )
    return [dict(row) for row in rows]


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
        raise ValueError(f"no private research document found for slug={clean_slug!r}")

    markdown_source = body if body is not None and body.strip() else private["markdown"]
    public_title, markdown = _as_public_markdown((title or private["title"] or "").strip(), markdown_source)
    filename = f"{clean_slug}.md"
    row, is_overwrite = research_store.upsert_document(
        filename=filename,
        title=public_title,
        markdown=markdown,
        summary=research_store.extract_excerpt(markdown),
        status="public",
        source_task_id=private.get("source_task_id"),
        updated_at=datetime.now(timezone.utc),
    )
    return {
        "private_research_document": private,
        "private_report": private,
        "research_document": row,
        "is_overwrite": is_overwrite,
        "public_url": _public_url(clean_slug),
        "markdown": markdown,
    }


PRIVATE_RESEARCH_DOCUMENT_TOOL = {
    "name": "private_research_document",
    "description": (
        "Manage admin-only private research documents with one action-based interface. "
        "Use action='save' to save or overwrite sensitive/unfinished research privately; "
        "action='read' to read one private research document; action='list' to list "
        "metadata/summaries; action='publish' only when the user/orchestrator explicitly "
        "asks to make a private research document public. Do not use publish_research "
        "for material that should remain private."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "read", "list", "publish"],
                "description": "save: store privately; read: full body by slug/id; list: metadata summaries; publish: make a private research document public.",
            },
            "title": {
                "type": "string",
                "description": "For action=save: private title. For action=publish: optional replacement public title.",
            },
            "slug": {"type": "string", "description": "Private research document slug."},
            "markdown_body": {"type": "string", "description": "For action=save: full Markdown body to store privately."},
            "body": {"type": "string", "description": "For action=publish: optional replacement Markdown body."},
            "document_id": {"type": "integer", "description": "Private research document id."},
            "report_id": {"type": "integer", "description": "Deprecated alias for document_id."},
            "source_task_id": {"type": "integer", "description": "For action=save: optional originating telegram_tasks id."},
            "limit": {"type": "integer", "description": "Maximum rows, 1-100.", "default": 20},
            "keyword": {"type": "string", "description": "For action=list: optional title/slug/body keyword filter."},
            "broadcast": {
                "type": "boolean",
                "description": "For action=publish: whether to broadcast the newly public document to the Telegram channel. Default true.",
                "default": True,
            },
        },
        "required": ["action"],
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
        logger.warning("save_private_research_document failed: %s", e)
        return f"Error: failed to save private research document: {type(e).__name__}: {e}"
    return (
        f"Saved private research document: id={row['id']} slug={row['slug']}\n"
        f"title: {row['title']}\n"
        f"sha256={row['content_sha256'][:12]} updated={_format_ts(row.get('updated_at'))}"
    )


async def _exec_read_private_report(
    slug: str | None = None,
    document_id: int | None = None,
    report_id: int | None = None,
) -> str:
    if document_id is None:
        document_id = report_id
    if document_id is None and not slug:
        return "Error: provide slug or document_id."
    try:
        row = await asyncio.to_thread(get_private_report_sync, report_id=document_id, slug=slug)
    except Exception as e:
        return f"Error: failed to read private research document: {type(e).__name__}: {e}"
    if not row:
        return "No private research document found."
    markdown = row.get("markdown") or ""
    return (
        f"=== PRIVATE RESEARCH DOCUMENT: {row['slug']} ===\n"
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
        return f"Error: failed to list private research documents: {type(e).__name__}: {e}"
    if not rows:
        return "No private research documents found."
    lines = [
        "=== PRIVATE RESEARCH DOCUMENTS ===",
        "Use read_private_research_document(slug='<slug>') for full detail.",
    ]
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
        return f"Error: failed to publish private research document: {type(e).__name__}: {e}"

    row = result["research_document"]
    public_url = result["public_url"]
    cache_note = ""
    try:
        from runtime_tools.research import _invalidate_cache_sync

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
                source="private research document publication",
            )
            if br.ok:
                broadcast_note = f"\nTelegram channel broadcast: sent ({br.sent_count})"
                if getattr(br, "message_ids", None):
                    await asyncio.to_thread(
                        record_publication_broadcast_sync,
                        slug=row["slug"],
                        public_url=public_url,
                        channel_message_ids=br.message_ids,
                        source="publish_private_research_document",
                    )
            else:
                broadcast_note = f"\nTelegram channel broadcast skipped/failed: {br.message}"
        except Exception as e:
            broadcast_note = f"\nTelegram channel broadcast failed: {e}"

    status = "Overwrote public research document" if result["is_overwrite"] else "Published public research document"
    return (
        f"{status}: {row['filename']}\n"
        f"Private research document slug: {slug}\n"
        f"Storage: research_documents id={row['id']} sha256={row['content_sha256'][:12]}\n"
        f"Public URL: {public_url}{cache_note}"
        f"{broadcast_note}"
    )


async def _exec_private_research_document(
    action: str,
    title: str | None = None,
    slug: str | None = None,
    markdown_body: str | None = None,
    body: str | None = None,
    document_id: int | None = None,
    report_id: int | None = None,
    source_task_id: int | None = None,
    limit: int = 20,
    keyword: str | None = None,
    broadcast: bool = True,
) -> str:
    op = (action or "").strip().lower()
    if op == "save":
        if not title or not slug or not markdown_body:
            return "Error: action='save' requires title, slug, and markdown_body."
        return await _exec_save_private_report(
            title=title,
            slug=slug,
            markdown_body=markdown_body,
            source_task_id=source_task_id,
        )
    if op == "read":
        return await _exec_read_private_report(
            slug=slug,
            document_id=document_id,
            report_id=report_id,
        )
    if op == "list":
        return await _exec_list_private_reports(limit=limit, keyword=keyword)
    if op == "publish":
        if not slug:
            return "Error: action='publish' requires slug."
        return await _exec_publish_private_report(
            slug=slug,
            body=body,
            title=title,
            broadcast=broadcast,
        )
    return "Error: action must be one of save, read, list, publish."


PRIVATE_REPORT_TOOLS = []

PRIVATE_REPORT_TOOL_HANDLERS = {
    "private_research_document": _exec_private_research_document,
    "save_private_research_document": _exec_save_private_report,
    "read_private_research_document": _exec_read_private_report,
    "list_private_research_documents": _exec_list_private_reports,
    "publish_private_research_document": _exec_publish_private_report,
    # Backward-compatible aliases for older prompts/tasks. These names are not
    # exposed in PRIVATE_REPORT_TOOLS.
    "save_private_report": _exec_save_private_report,
    "read_private_report": _exec_read_private_report,
    "list_private_reports": _exec_list_private_reports,
    "publish_private_report": _exec_publish_private_report,
}
