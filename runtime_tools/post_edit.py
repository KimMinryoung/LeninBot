"""runtime_tools.post_edit — Edit public-facing posts with cache invalidation.

The frontend (Node.js, runs in Docker) caches each published post permanently
in Redis under `{kind}:{id}` and serves from cache before hitting the DB. A
direct `UPDATE` via `query_db` therefore goes *unseen* on the public site
until the cache TTL (which, for per-entry keys, is never). Tasks 587/588 hit
exactly this trap — DB was edited, readers kept seeing the old text.

This tool bundles the cache-busting steps so they cannot drift:
  1. UPDATE or remove the target row (ai_diary / telegram_tasks / posts)
  2. DEL the per-entry cache key + the list/nav caches that embed entry content
  3. Purge the affected public URLs from Cloudflare via the frontend script

The diary agent writes new entries via save_diary; maintenance flows use this
tool for edits and, for diary rows only, explicit delete/unpublish actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from typing import Any

from db import execute_returning_rowcount as db_exec, get_conn, query_one as db_query_one
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

FRONTEND_DIR = os.getenv("FRONTEND_DIR", "/home/grass/frontend")
CF_PURGE_SCRIPT = os.getenv(
    "CF_PURGE_SCRIPT",
    os.path.join(FRONTEND_DIR, "scripts", "cloudflare-purge.js"),
)


# Per kind: table, which fields the tool may write, which cache keys to purge.
# "*" in a cache key is a SCAN pattern (report list pages and post/diary index
# pages are cached one key per page+lang, e.g. post:index:page:2:ko).
_KIND_CONFIG: dict[str, dict[str, Any]] = {
    "diary": {
        "table": "ai_diary",
        "allowed_fields": ("title", "content"),
        "entry_key": "diary:{id}",
        "localized_cache": True,
        "index_keys": ("diary:index:*", "diary:nav"),
    },
    "report": {
        "table": "telegram_tasks",
        "allowed_fields": ("content", "result"),
        "entry_key": "report:{id}",
        "index_keys": ("report:list:*",),
    },
    "post": {
        "table": "posts",
        "allowed_fields": ("title", "content"),
        "entry_key": "post:{id}",
        "localized_cache": True,
        "index_keys": ("post:index:*", "post:nav"),
    },
    "curation": {
        "table": "hub_curations",
        "allowed_fields": (
            "title", "source_url", "source_title", "source_author",
            "source_publication", "source_published_at",
            "selection_rationale", "context", "tags",
        ),
        "where_field": "slug",
        "touch_updated_at": True,
        "index_keys": (),
    },
    "static_page": {
        "table": "static_pages",
        "allowed_fields": (
            "title", "summary", "html_body",
            "title_en", "summary_en", "html_body_en",
        ),
        "where_field": "slug",
        "touch_updated_at": True,
        "index_keys": ("report:pages_list:ko", "report:pages_list:en"),
    },
}


EDIT_CONTENT_TOOL = {
    "name": "edit_content",
    "description": (
        "Edit an already-published diary, task report, blog post, hub curation, "
        "or static/custom HTML page, and delete/unpublish diary entries, AND "
        "invalidate Redis plus Cloudflare caches in one step. "
        "Use this instead of query_db when correcting already-published content; "
        "a raw UPDATE leaves readers seeing stale cached content. "
        "content_type='diary' for diary entries; content_type='task_report' for completed "
        "Telegram task reports; content_type='blog_post' for blog posts; "
        "content_type='hub_curation' for hub curation entries; "
        "content_type='static_page' for /p/{slug} custom HTML pages. "
        "Do NOT use this for research documents; use research_document. "
        "Provide id for diary/task_report/blog_post, or slug for hub_curation/static_page, "
        "plus at least one field for the chosen kind. For a narrow correction, pass "
        "`field`, `replace_old`, and `replace_new`; the tool reads the current field, "
        "shows matching snippets with about 10 characters of surrounding context, and "
        "updates only when the match is unambiguous unless `replace_all=true`. "
        "For diary deletion or unpublishing, pass action=delete or action=unpublish, "
        "id, and confirm=true; because ai_diary has no private status column, "
        "unpublish removes the row from the public diary table."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content_type": {
                "type": "string",
                "enum": ["diary", "task_report", "blog_post", "hub_curation", "static_page"],
                "description": "Content type: diary entry, task report, blog post, hub curation, or static page.",
            },
            "id": {
                "type": "integer",
                "description": "Numeric content id. Required for diary/task_report/blog_post; for task_report this is the task_id.",
            },
            "slug": {
                "type": "string",
                "description": "Slug. Required when content_type='hub_curation' or content_type='static_page'.",
            },
            "action": {
                "type": "string",
                "enum": ["edit", "delete", "unpublish"],
                "description": (
                    "Default edit. delete/unpublish are supported only for content_type='diary' "
                    "and remove the diary row from public ai_diary storage."
                ),
            },
            "confirm": {
                "type": "boolean",
                "description": "Required true for destructive diary delete/unpublish actions.",
            },
            "title": {
                "type": "string",
                "description": "New title. Valid for diary, post, curation, and static_page.",
            },
            "content": {
                "type": "string",
                "description": "New body text / markdown. Valid for diary, report, and post.",
            },
            "result": {
                "type": "string",
                "description": "New result body. Valid for report only (the rendered output).",
            },
            "source_url": {"type": "string", "description": "New curation source URL. Valid for curation only."},
            "source_title": {"type": "string", "description": "New original source title. Valid for curation only."},
            "source_author": {"type": "string", "description": "New source author/byline. Valid for curation only."},
            "source_publication": {"type": "string", "description": "New publication/site name. Valid for curation only."},
            "source_published_at": {"type": "string", "description": "New source publication date in YYYY-MM-DD. Valid for curation only."},
            "selection_rationale": {"type": "string", "description": "New selection rationale. Valid for curation only."},
            "context": {"type": "string", "description": "New contextual framing. Valid for curation only."},
            "summary": {
                "type": "string",
                "description": "New static page summary/description. Valid for static_page only.",
            },
            "html_body": {
                "type": "string",
                "description": (
                    "New static page Korean HTML inner body. Valid for static_page only; "
                    "must not include <html>, <head>, <body>, <script>, <iframe>, "
                    "inline event handlers, or javascript:/data: URLs."
                ),
            },
            "title_en": {
                "type": "string",
                "description": "New English title. Valid for static_page only.",
            },
            "summary_en": {
                "type": "string",
                "description": "New English summary/description. Valid for static_page only.",
            },
            "html_body_en": {
                "type": "string",
                "description": (
                    "New static page English HTML inner body. Valid for static_page only; "
                    "same safety restrictions as html_body."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Replacement curation tag list. Valid for curation only.",
            },
            "field": {
                "type": "string",
                "description": (
                    "Surgical mode only. Editable text field to modify, e.g. content, result, "
                    "title, context, selection_rationale, html_body."
                ),
            },
            "replace_old": {
                "type": "string",
                "description": "Surgical mode only. Literal text to find in the current field value.",
            },
            "replace_new": {
                "type": "string",
                "description": "Surgical mode only. Replacement text for replace_old.",
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "Surgical mode only. Default false. If replace_old appears multiple times, "
                    "false returns contextual match snippets without editing; true replaces every match."
                ),
            },
        },
        "required": ["content_type"],
    },
}

# Backward-compatible schema object kept for internal callers/tests that import it.
EDIT_PUBLIC_POST_TOOL = {
    **EDIT_CONTENT_TOOL,
    "name": "edit_public_post",
}


def _literal_match_spans(text: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            return spans
        end = pos + len(needle)
        spans.append((pos, end))
        start = end


def _format_match_snippets(text: str, spans: list[tuple[int, int]], *, context_chars: int = 10, limit: int = 20) -> str:
    lines = []
    for idx, (start, end) in enumerate(spans[:limit], 1):
        left = text[max(0, start - context_chars):start]
        match = text[start:end]
        right = text[end:end + context_chars]
        snippet = f"{left}[[{match}]]{right}".replace("\n", "\\n")
        lines.append(f"{idx}. pos {start}-{end}: {snippet}")
    if len(spans) > limit:
        lines.append(f"... {len(spans) - limit} more match(es) omitted")
    return "\n".join(lines)


def _invalidate_cache_sync(kind: str, post_id: int) -> dict[str, Any]:
    """Blocking cache cleanup. Returns {ok, deleted, reason?}. Redis-down is
    soft-failure: we report it back to the agent instead of raising."""
    from redis_state import get_redis

    cfg = _KIND_CONFIG[kind]
    if not cfg.get("entry_key") and not cfg.get("index_keys"):
        return {"ok": True, "deleted": 0}
    r = get_redis()
    if r is None:
        return {"ok": False, "deleted": 0, "reason": "redis_unavailable"}

    deleted = 0
    try:
        entry_key = cfg.get("entry_key")
        if entry_key:
            key = entry_key.format(id=post_id)
            keys = [key]
            if cfg.get("localized_cache"):
                keys.extend([f"{key}:ko", f"{key}:en"])
            deleted += int(r.delete(*keys) or 0)
        for pattern in cfg["index_keys"]:
            if "*" in pattern:
                for k in r.scan_iter(match=pattern):
                    deleted += int(r.delete(k) or 0)
            else:
                keys = [pattern]
                if cfg.get("localized_cache") and pattern.endswith(":index"):
                    keys.extend([f"{pattern}:ko", f"{pattern}:en"])
                deleted += int(r.delete(*keys) or 0)
    except Exception as e:
        logger.warning("edit_public_post cache invalidation failed (%s:%s): %s", kind, post_id, e)
        return {"ok": False, "deleted": deleted, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "deleted": deleted}


def _cloudflare_purge_paths(kind: str, target: int | str) -> list[str]:
    """Return public URLs whose Cloudflare edge cache can embed this row."""
    if kind == "diary":
        return [
            f"/ai-diary/{target}",
            "/ai-diary",
            "/ai-diary.md",
            "/",
            "/rss.xml",
            "/atom.xml",
            "/sitemap.xml",
        ]
    if kind == "post":
        return [
            f"/post/{target}",
            "/posts",
            "/posts.md",
            "/",
            "/rss.xml",
            "/atom.xml",
            "/sitemap.xml",
        ]
    if kind == "report":
        return [
            f"/reports/{target}",
            "/reports",
        ]
    if kind == "curation":
        return [
            f"/hub/{target}",
            "/hub",
            "/hub.md",
            "/",
            "/rss.xml",
            "/atom.xml",
            "/sitemap.xml",
        ]
    if kind == "static_page":
        return [
            f"/p/{target}",
            "/reports",
            "/",
            "/sitemap.xml",
        ]
    return []


def _validate_static_page_updates(provided: dict[str, Any]) -> str | None:
    from site_publishing import _validate_inner_html

    if provided.get("title") is not None and not str(provided["title"]).strip():
        return "Error: title must not be empty for kind='static_page'."
    if provided.get("title_en") is not None and not str(provided["title_en"]).strip():
        return "Error: title_en must not be empty for kind='static_page'."

    for html_field in ("html_body", "html_body_en"):
        if provided.get(html_field) is None:
            continue
        html = str(provided[html_field]).strip()
        if not html:
            return f"Error: {html_field} must not be empty for kind='static_page'."
        validation_error = _validate_inner_html(html, html_field)
        if validation_error:
            return validation_error
    return None


def _ensure_static_page_storage_sync() -> None:
    from site_publishing import _ensure_static_page_table

    _ensure_static_page_table()


def _purge_cloudflare_sync(kind: str, target: int | str) -> dict[str, Any]:
    """Purge Cloudflare URLs through the frontend script.

    Failure is reported to the caller but never raises: the database update and
    Redis invalidation are still the source-of-truth changes.
    """
    paths = list(dict.fromkeys(_cloudflare_purge_paths(kind, target)))
    if not paths:
        return {"ok": True, "purged": 0, "urls": []}
    if not os.path.isfile(CF_PURGE_SCRIPT):
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": f"script_missing: {CF_PURGE_SCRIPT}",
        }

    try:
        proc = subprocess.run(
            ["node", CF_PURGE_SCRIPT, *paths],
            cwd=FRONTEND_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except Exception as e:
        logger.warning("Cloudflare purge failed before execution (%s:%s): %s", kind, target, e)
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": f"{type(e).__name__}: {e}",
        }

    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if proc.returncode != 0:
        logger.warning(
            "Cloudflare purge failed (%s:%s, exit=%s): %s",
            kind,
            target,
            proc.returncode,
            output,
        )
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": output or f"exit_{proc.returncode}",
        }
    return {"ok": True, "purged": len(paths), "urls": paths, "output": output}


def _format_invalidation_note(
    cache: dict[str, Any],
    cf: dict[str, Any],
    cfg: dict[str, Any],
    target: int | str,
) -> str:
    if cache["ok"]:
        cache_note = f"invalidated {cache['deleted']} Redis key(s)"
    else:
        entry_key = cfg.get("entry_key")
        manual_key = entry_key.format(id=target) if entry_key else "(no per-entry cache)"
        cache_note = (
            f"CACHE INVALIDATION FAILED ({cache['reason']}) — "
            f"run `redis-cli DEL {manual_key}` manually"
        )

    if cf["ok"]:
        cf_note = f"purged {cf['purged']} Cloudflare URL(s)"
    else:
        urls = " ".join(cf.get("urls") or [])
        cf_note = (
            f"CLOUDFLARE PURGE FAILED ({cf.get('reason', 'unknown')}) — "
            f"run `cd {FRONTEND_DIR} && node scripts/cloudflare-purge.js {urls}` manually"
        )
    return f"{cache_note}; {cf_note}"


def _delete_diary_sync(target: int) -> tuple[dict[str, Any] | None, int, int]:
    """Remove a diary row after clearing publication-audit FK references."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, created_at FROM ai_diary WHERE id = %s FOR UPDATE",
                (target,),
            )
            row = cur.fetchone()
            if not row:
                return None, 0, 0
            cur.execute(
                "UPDATE diary_publication_audits SET diary_id = NULL WHERE diary_id = %s",
                (target,),
            )
            audit_links_cleared = cur.rowcount
            cur.execute("DELETE FROM ai_diary WHERE id = %s", (target,))
            deleted = cur.rowcount
            return dict(row), audit_links_cleared, deleted


async def _exec_diary_remove_action(
    *,
    action: str,
    target: int,
    cfg: dict[str, Any],
) -> str:
    row, audit_links_cleared, deleted = await asyncio.to_thread(_delete_diary_sync, target)
    if not row or not deleted:
        return f"Error: no diary row found with id={target!r} — nothing removed."

    cache = await asyncio.to_thread(_invalidate_cache_sync, "diary", target)
    cf = await asyncio.to_thread(_purge_cloudflare_sync, "diary", target)
    invalidation_note = _format_invalidation_note(cache, cf, cfg, target)
    verb = "Unpublished" if action == "unpublish" else "Deleted"
    title = row.get("title") or "(untitled)"
    extra = (
        " Removed from public ai_diary storage because diary entries do not have "
        "a private/unpublished status column."
        if action == "unpublish"
        else ""
    )
    return (
        f"{verb} diary id={target!r}: {title}; "
        f"cleared {audit_links_cleared} publication audit link(s); {invalidation_note}."
        f"{extra}"
    )


async def _exec_edit_public_post(
    kind: str,
    post_id: int | None = None,
    slug: str | None = None,
    title: str | None = None,
    content: str | None = None,
    result: str | None = None,
    summary: str | None = None,
    html_body: str | None = None,
    title_en: str | None = None,
    summary_en: str | None = None,
    html_body_en: str | None = None,
    source_url: str | None = None,
    source_title: str | None = None,
    source_author: str | None = None,
    source_publication: str | None = None,
    source_published_at: str | None = None,
    selection_rationale: str | None = None,
    context: str | None = None,
    tags: list | None = None,
    field: str | None = None,
    replace_old: str | None = None,
    replace_new: str | None = None,
    replace_all: bool = False,
    action: str | None = None,
    confirm: bool = False,
) -> str:
    kind = (kind or "").strip().lower()
    if kind not in _KIND_CONFIG:
        return f"Error: kind must be one of {sorted(_KIND_CONFIG)}."

    cfg = _KIND_CONFIG[kind]
    allowed = cfg["allowed_fields"]

    if kind in {"curation", "static_page"}:
        slug = (slug or "").strip().lower()
        if not slug:
            return f"Error: slug is required for kind='{kind}'."
        if not re.match(r"^[a-z0-9][a-z0-9-]{0,79}$", slug):
            return "Error: slug must match ^[a-z0-9][a-z0-9-]{0,79}$."
        target = slug
    else:
        try:
            target = int(post_id)
        except (TypeError, ValueError):
            return "Error: post_id must be an integer."

    action_name = (action or "edit").strip().lower()
    if action_name not in {"edit", "delete", "unpublish"}:
        return "Error: action must be one of edit, delete, unpublish."
    if action_name in {"delete", "unpublish"}:
        if kind != "diary":
            return "Error: delete/unpublish actions are currently supported only for content_type='diary'."
        if confirm is not True:
            return "Error: destructive diary delete/unpublish requires confirm=true."

    provided = {
        "title": title,
        "content": content,
        "result": result,
        "summary": summary,
        "html_body": html_body,
        "title_en": title_en,
        "summary_en": summary_en,
        "html_body_en": html_body_en,
        "source_url": source_url,
        "source_title": source_title,
        "source_author": source_author,
        "source_publication": source_publication,
        "source_published_at": source_published_at,
        "selection_rationale": selection_rationale,
        "context": context,
        "tags": tags,
    }
    surgical_requested = any(v is not None for v in (field, replace_old, replace_new)) or replace_all is True
    direct_updates = [(f, v) for f, v in provided.items() if v is not None]

    if action_name in {"delete", "unpublish"}:
        if surgical_requested or direct_updates:
            return "Error: delete/unpublish cannot be combined with edit fields."
        return await _exec_diary_remove_action(action=action_name, target=target, cfg=cfg)

    if surgical_requested:
        if direct_updates:
            return (
                "Error: surgical mode cannot be combined with direct field replacement. "
                "Use either field/replace_old/replace_new or normal editable fields, not both."
            )
        edit_field = (field or "").strip()
        if not edit_field or replace_old is None or replace_new is None:
            return "Error: surgical mode requires field, replace_old, and replace_new."
        if edit_field not in allowed:
            return (
                f"Error: field={edit_field!r} is not editable on kind='{kind}'. "
                f"Allowed: {list(allowed)}."
            )
        if edit_field == "tags":
            return "Error: surgical mode is not supported for tags; replace the whole tags array instead."
        if replace_old == "":
            return "Error: replace_old must not be empty."

        where_field = cfg.get("where_field", "id")
        if kind == "static_page":
            await asyncio.to_thread(_ensure_static_page_storage_sync)
        try:
            row = await asyncio.to_thread(
                db_query_one,
                f"SELECT {edit_field} FROM {cfg['table']} WHERE {where_field} = %s",
                (target,),
            )
        except Exception as e:
            logger.warning("edit_public_post SELECT failed (%s %s=%s): %s", kind, where_field, target, e)
            return f"Error: DB read failed: {type(e).__name__}: {e}"
        if not row:
            return f"Error: no {kind} row found with {where_field}={target!r} — nothing updated."

        current_value = row.get(edit_field)
        if current_value is None:
            current_text = ""
        elif isinstance(current_value, str):
            current_text = current_value
        else:
            current_text = str(current_value)
        spans = _literal_match_spans(current_text, replace_old)
        if not spans:
            return (
                f"Error: replace_old not found in {kind} {where_field}={target!r} field={edit_field!r}; "
                "nothing updated."
            )
        snippets = _format_match_snippets(current_text, spans)
        if len(spans) > 1 and replace_all is not True:
            return (
                f"Multiple matches found in {kind} {where_field}={target!r} field={edit_field!r}; "
                "nothing updated. Use a more specific replace_old or set replace_all=true to replace all matches.\n"
                f"Matches:\n{snippets}"
            )

        new_text = current_text.replace(replace_old, replace_new) if replace_all else (
            current_text[:spans[0][0]] + replace_new + current_text[spans[0][1]:]
        )
        provided[edit_field] = new_text

    rejected = [f for f, v in provided.items() if v is not None and f not in allowed]
    if rejected:
        return (
            f"Error: field(s) {rejected} are not editable on kind='{kind}'. "
            f"Allowed: {list(allowed)}."
        )

    updates = [(f, provided[f]) for f in allowed if provided[f] is not None]
    if not updates:
        return f"Error: provide at least one of {list(allowed)} to update."
    if kind == "curation" and provided.get("source_url") is not None:
        cleaned_url = str(provided["source_url"]).strip()
        if cleaned_url and not cleaned_url.startswith(("http://", "https://")):
            return "Error: source_url must be an http(s) URL."
    if kind == "static_page":
        validation_error = _validate_static_page_updates(provided)
        if validation_error:
            return validation_error
        await asyncio.to_thread(_ensure_static_page_storage_sync)

    set_parts = [f"{f} = %s" for f, _ in updates]
    params = []
    for field, value in updates:
        if field == "tags":
            params.append(json.dumps([str(t)[:50] for t in (value or [])][:20]))
        elif isinstance(value, str):
            params.append(value.strip())
        else:
            params.append(value)
    if kind == "diary" and any(f in {"title", "content"} for f, _ in updates):
        set_parts.extend(["title_en = NULL", "content_en = NULL"])
    if cfg.get("touch_updated_at"):
        set_parts.append("updated_at = NOW()")
    set_clause = ", ".join(set_parts)
    where_field = cfg.get("where_field", "id")
    params = tuple(params) + (target,)
    sql = f"UPDATE {cfg['table']} SET {set_clause} WHERE {where_field} = %s"

    try:
        affected = await asyncio.to_thread(db_exec, sql, params)
    except Exception as e:
        logger.warning("edit_public_post UPDATE failed (%s id=%s): %s", kind, post_id, e)
        return f"Error: DB update failed: {type(e).__name__}: {e}"

    if not affected:
        return f"Error: no {kind} row found with {where_field}={target!r} — nothing updated."

    cache = await asyncio.to_thread(_invalidate_cache_sync, kind, int(target) if isinstance(target, int) else 0)
    cf = await asyncio.to_thread(_purge_cloudflare_sync, kind, target)
    invalidation_note = _format_invalidation_note(cache, cf, cfg, target)
    fields_str = ", ".join(f for f, _ in updates)
    if surgical_requested:
        return (
            f"Updated {kind} {where_field}={target!r}: surgical replace in {field!r} "
            f"({len(spans)} match(es)); {invalidation_note}.\n"
            f"Matches replaced:\n{snippets}"
        )
    return f"Updated {kind} {where_field}={target!r}: [{fields_str}]; {invalidation_note}."


async def _exec_edit_content(
    content_type: str | None = None,
    id: int | None = None,
    kind: str | None = None,
    post_id: int | None = None,
    **kwargs: Any,
) -> str:
    canonical = (content_type or kind or "").strip().lower()
    kind_map = {
        "diary": "diary",
        "task_report": "report",
        "report": "report",
        "blog_post": "post",
        "post": "post",
        "hub_curation": "curation",
        "curation": "curation",
        "static_page": "static_page",
        "static_pages": "static_page",
    }
    mapped = kind_map.get(canonical)
    if not mapped:
        return "Error: content_type must be one of diary, task_report, blog_post, hub_curation, static_page."
    return await _exec_edit_public_post(
        kind=mapped,
        post_id=id if id is not None else post_id,
        **kwargs,
    )


POST_EDIT_TOOLS = [EDIT_CONTENT_TOOL]
POST_EDIT_TOOL_HANDLERS = {
    "edit_content": _exec_edit_content,
    "edit_public_post": _exec_edit_public_post,
}
