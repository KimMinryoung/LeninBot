"""research_tools.py — Publish, edit, and unpublish public research documents.

Public research documents are stored as Markdown rows in Supabase/Postgres and
served at https://cyber-lenin.com/reports/research/{slug}, where slug is the
filename without its .md extension. Legacy files under research/ and
output/research/ remain readable as fallback/import sources.

This module consolidates:
  * publish_research(title, content, filename?) — DB upsert + cache bust
  * edit_research(operation, filename, ...) —
      operation='edit'      → rewrite DB markdown and bust cache
      operation='unpublish' → set status=private (or move legacy file) and bust cache

`unpublish` is intentionally non-destructive: DB rows are marked private and
legacy files are relocated out of the public-listing scope.

Mirrors the post_edit_tools.py pattern (UPDATE + cache purge in one step) for
filesystem-backed artifacts instead of DB rows.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from telegram.channel_broadcast import maybe_broadcast_autonomous_publication
import research_store

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = _PROJECT_ROOT / "research"
LEGACY_RESEARCH_DIR = _PROJECT_ROOT / "output" / "research"
PRIVATE_RESEARCH_DIR = RESEARCH_DIR / "private"

KST = timezone(timedelta(hours=9))


# ── Helpers ──────────────────────────────────────────────────────────

# Filename allowlist: ASCII alnum + . _ - , must end in .md, no leading dot.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")


def _validate_filename(filename: str) -> str:
    """Normalize and validate a research filename. Raises ValueError on bad input."""
    if not filename:
        raise ValueError("filename is required")
    fname = filename.strip()
    if not fname:
        raise ValueError("filename is required")
    if not fname.endswith(".md"):
        fname += ".md"
    if "/" in fname or "\\" in fname or ".." in fname:
        raise ValueError("filename must not contain path separators or '..'")
    if not _FILENAME_RE.fullmatch(fname):
        raise ValueError(
            "filename must be ASCII letters/digits with '.', '_', '-' only "
            "(no spaces, no leading dot)"
        )
    return fname


def _slug_from_title(title: str) -> str:
    """Build an ASCII slug from a title for filename use."""
    normalized = unicodedata.normalize("NFKD", title)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
    return slug[:80] if slug else "research"


def _resolve_existing(filename: str) -> Path | None:
    """Return existing file path (preferring research/ over legacy output/research/), else None."""
    primary = RESEARCH_DIR / filename
    if primary.is_file():
        return primary
    legacy = LEGACY_RESEARCH_DIR / filename
    if legacy.is_file():
        return legacy
    return None


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically: stage to .tmp then os.replace.

    Prevents a half-written file from appearing in the public listing if the
    write is interrupted (signal, OOM, disk-full).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _extract_h1(path: Path) -> str | None:
    """Return the first markdown H1 in the file, or None."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    return stripped[2:].strip() or None
    except Exception:
        return None
    return None


_PUBLISH_DATE_RE = re.compile(r"\*\*작성일:\*\*\s*(\d{4}-\d{2}-\d{2})")


def _extract_publish_date(path: Path) -> str | None:
    """Find the existing `**작성일:** YYYY-MM-DD` line. Lets edit preserve original date."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                m = _PUBLISH_DATE_RE.match(line.strip())
                if m:
                    return m.group(1)
    except Exception:
        return None
    return None


def _build_document(title: str, content: str, publish_date: str) -> str:
    """Compose the canonical research-document layout."""
    return (
        f"# {title}\n"
        f"**작성자:** Cyber-Lenin (사이버-레닌)\n"
        f"**작성일:** {publish_date}\n\n"
        f"---\n\n"
        f"{content.strip()}\n"
    )


def _cache_safe_key(filename: str) -> str:
    """Mirror the frontend's safe-key transform: non-[A-Za-z0-9._-] → '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


def _invalidate_cache_sync(filename: str) -> dict[str, Any]:
    """Drop the per-file and list caches in Redis. Returns {ok, deleted, reason?}."""
    from redis_state import get_redis

    r = get_redis()
    if r is None:
        return {"ok": False, "deleted": 0, "reason": "redis_unavailable"}
    deleted = 0
    try:
        safe = _cache_safe_key(filename)
        deleted += int(r.delete(
            f"research:{safe}",
            f"research:{safe}:ko",
            f"research:{safe}:en",
            "report:research_list",
            "report:research_list:ko",
            "report:research_list:en",
        ) or 0)
    except Exception as e:
        logger.warning("research cache invalidation failed for %s: %s", filename, e)
        return {"ok": False, "deleted": deleted, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "deleted": deleted}


def _public_slug(filename: str) -> str:
    return filename[:-3] if filename.endswith(".md") else filename


def _public_url(filename: str) -> str:
    return f"https://cyber-lenin.com/reports/research/{_public_slug(filename)}"


def _format_cache_note(cache: dict[str, Any], filename: str, *, missing_msg: str | None = None) -> str:
    if cache["ok"]:
        return f"cache invalidated ({cache['deleted']} key(s))"
    base = f"CACHE INVALIDATION FAILED ({cache['reason']})"
    safe = _cache_safe_key(filename)
    manual = (
        f"redis-cli DEL research:{safe} research:{safe}:ko research:{safe}:en "
        "report:research_list report:research_list:ko report:research_list:en"
    )
    tail = f" — {missing_msg} Manually run: {manual}" if missing_msg else f" — readers may see stale data. Manually run: {manual}"
    return base + tail


# ── publish_research ─────────────────────────────────────────────────

PUBLISH_RESEARCH_TOOL = {
    "name": "publish_research",
    "description": (
        "Write a markdown document to the public research database. "
        "Published files are served at https://cyber-lenin.com/reports/research/{slug}, "
        "where slug is the filename without the .md extension. "
        "Use for polished analysis, forecasts, and investigative findings. "
        "Filename is auto-generated from the title with a date prefix (YYYYMMDD_slug.md) "
        "unless `filename` is passed explicitly. Writing the same filename overwrites; the "
            "frontend Redis cache is invalidated so readers see the new version immediately."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Document title. Used for both the H1 heading and the auto filename slug.",
            },
            "content": {
                "type": "string",
                "description": "Full markdown content (without the title heading — auto-prepended).",
            },
            "filename": {
                "type": "string",
                "description": "Optional. ASCII letters/digits with '.', '_', '-' only; '.md' appended if missing.",
            },
        },
        "required": ["title", "content"],
    },
}


async def _exec_publish_research(title: str, content: str, filename: str | None = None) -> str:
    if not title or not title.strip():
        return "Error: title is required."
    if not content or not content.strip():
        return "Error: content is required."

    title = title.strip()
    now = datetime.now(KST)

    if filename:
        try:
            fname = _validate_filename(filename)
        except ValueError as e:
            return f"Error: {e}."
    else:
        fname = f"{now.strftime('%Y%m%d')}_{_slug_from_title(title)}.md"

    document = _build_document(title, content, now.strftime("%Y-%m-%d"))

    try:
        row, is_overwrite = await asyncio.to_thread(
            research_store.upsert_document,
            filename=fname,
            title=title,
            markdown=document,
            summary=research_store.extract_excerpt(document),
            status="public",
        )
    except Exception as e:
        logger.error("publish_research DB write error for %s: %s", fname, e)
        return f"Error: failed to store {fname}: {type(e).__name__}: {e}"

    cache = await asyncio.to_thread(_invalidate_cache_sync, fname)
    status = "Overwrote" if is_overwrite else "Published"
    public_url = _public_url(fname)
    broadcast_note = ""
    try:
        br = await maybe_broadcast_autonomous_publication(
            title=title,
            url=public_url,
            body=content,
            source="cyber-lenin.com research",
        )
        if br.ok:
            broadcast_note = f"\nTelegram channel broadcast: sent ({br.sent_count})"
    except Exception as e:
        logger.warning("research channel broadcast failed for %s: %s", fname, e)
        broadcast_note = f"\nTelegram channel broadcast failed: {e}"
    return (
        f"{status}: {fname}\n"
        f"Storage: research_documents id={row['id']} sha256={row['content_sha256'][:12]}\n"
        f"Public URL: {public_url}\n"
        f"Size: {len(document)} chars; {_format_cache_note(cache, fname)}"
        f"{broadcast_note}"
    )


# ── edit_research ────────────────────────────────────────────────────

EDIT_RESEARCH_TOOL = {
    "name": "edit_research",
    "description": (
        "Edit or unpublish an already-published research document. "
        "operation='edit': rewrite the database Markdown with new content (and optionally a new title) "
        "and invalidate Redis cache so readers see the new version immediately. "
        "Original 작성일 is preserved when present in the existing file. "
        "operation='unpublish': mark a DB row private or move a legacy file to research/private/ "
        "and invalidate cache so it disappears from cyber-lenin.com. "
        "Use this instead of publish_research when correcting or pulling already-public content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["edit", "unpublish"],
                "description": "'edit' rewrites content; 'unpublish' archives to research/private/.",
            },
            "filename": {
                "type": "string",
                "description": "Existing research filename (e.g. '20260418_imperialism_intro.md').",
            },
            "title": {
                "type": "string",
                "description": "operation=edit only. Optional new H1; if omitted, the existing H1 is reused.",
            },
            "content": {
                "type": "string",
                "description": "operation=edit only. Required body markdown (without the H1 heading).",
            },
        },
        "required": ["operation", "filename"],
    },
}


def _unpublish_sync(existing: Path) -> Path:
    """Move file to research/private/ with a collision-safe destination. Returns new path."""
    PRIVATE_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    dest = PRIVATE_RESEARCH_DIR / existing.name
    if dest.exists():
        ts = datetime.now(KST).strftime("%Y%m%dT%H%M%S")
        dest = PRIVATE_RESEARCH_DIR / f"{existing.stem}.{ts}{existing.suffix}"
    os.replace(existing, dest)
    return dest


def _extract_publish_date_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines()[:20]:
        m = _PUBLISH_DATE_RE.match(line.strip())
        if m:
            return m.group(1)
    return None


async def _exec_edit_research(
    operation: str,
    filename: str,
    title: str | None = None,
    content: str | None = None,
) -> str:
    op = (operation or "").strip().lower()
    if op not in {"edit", "unpublish"}:
        return "Error: operation must be 'edit' or 'unpublish'."
    try:
        fname = _validate_filename(filename)
    except ValueError as e:
        return f"Error: {e}."

    existing_doc = await asyncio.to_thread(research_store.get_document, fname, include_private=True)
    existing = None if existing_doc else _resolve_existing(fname)
    if existing_doc is None and existing is None:
        return f"Error: no research file named '{fname}' under research/ or output/research/."

    if op == "edit":
        if not content or not content.strip():
            return "Error: content is required for operation='edit'."
        existing_markdown = existing_doc["markdown"] if existing_doc else existing.read_text(encoding="utf-8")
        existing_title = (
            existing_doc.get("title") if existing_doc else None
        ) or research_store.extract_title(existing_markdown, "") or (None if existing is None else _extract_h1(existing))
        new_title = (title or "").strip() or existing_title
        if not new_title:
            return "Error: existing file has no H1 — pass `title` explicitly."
        publish_date = (
            _extract_publish_date_from_markdown(existing_markdown)
            or (None if existing is None else _extract_publish_date(existing))
            or datetime.now(KST).strftime("%Y-%m-%d")
        )
        document = _build_document(new_title, content, publish_date)
        try:
            row, _ = await asyncio.to_thread(
                research_store.upsert_document,
                filename=fname,
                title=new_title,
                markdown=document,
                summary=research_store.extract_excerpt(document),
                status="public",
            )
        except Exception as e:
            logger.error("edit_research DB write error for %s: %s", fname, e)
            return f"Error: failed to rewrite {fname}: {type(e).__name__}: {e}"

        cache = await asyncio.to_thread(_invalidate_cache_sync, fname)
        return (
            f"Edited: {fname}\n"
            f"Storage: research_documents id={row['id']} sha256={row['content_sha256'][:12]}\n"
            f"Public URL: {_public_url(fname)}\n"
            f"Title: {new_title}; size: {len(document)} chars; {_format_cache_note(cache, fname)}"
        )

    # operation == "unpublish"
    backup_note = ""
    if existing_doc:
        try:
            row = await asyncio.to_thread(research_store.set_status, fname, "private")
            backup_note = f"Storage: research_documents id={row['id']} status=private"
        except Exception as e:
            logger.error("edit_research unpublish DB error for %s: %s", fname, e)
            return f"Error: failed to mark {fname} private: {type(e).__name__}: {e}"
    else:
        try:
            new_path = await asyncio.to_thread(_unpublish_sync, existing)
            backup_note = f"Backup path: {new_path}"
        except Exception as e:
            logger.error("edit_research unpublish error for %s: %s", fname, e)
            return f"Error: failed to move {fname} to private/: {type(e).__name__}: {e}"

    cache = await asyncio.to_thread(_invalidate_cache_sync, fname)
    cache_note = _format_cache_note(
        cache,
        fname,
        missing_msg="file moved but the cached copy may still be served." if not cache["ok"] else None,
    )
    return (
        f"Unpublished: {fname}\n"
        f"{backup_note}\n"
        f"Public URL (now 404): {_public_url(fname)}\n"
        f"{cache_note}"
    )


# ── Registry exports ─────────────────────────────────────────────────

RESEARCH_TOOLS = [PUBLISH_RESEARCH_TOOL, EDIT_RESEARCH_TOOL]
RESEARCH_TOOL_HANDLERS = {
    "publish_research": _exec_publish_research,
    "edit_research": _exec_edit_research,
}
