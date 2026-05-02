"""post_edit_tools.py — Edit public-facing posts with cache invalidation.

The frontend (Node.js, runs in Docker) caches each published post permanently
in Redis under `{kind}:{id}` and serves from cache before hitting the DB. A
direct `UPDATE` via `query_db` therefore goes *unseen* on the public site
until the cache TTL (which, for per-entry keys, is never). Tasks 587/588 hit
exactly this trap — DB was edited, readers kept seeing the old text.

This tool bundles the two steps so they cannot drift:
  1. UPDATE the target row (ai_diary / telegram_tasks / posts)
  2. DEL the per-entry cache key + the list/nav caches that embed entry content

Kept programmer-only (same blast-radius class as query_db) — the diary agent
writes new entries via save_diary; only maintenance flows need to *edit*.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from db import execute_returning_rowcount as db_exec, query_one as db_query_one

logger = logging.getLogger(__name__)


# Per kind: table, which fields the tool may write, which cache keys to purge.
# "*" in a cache key is a SCAN pattern (used for report list pages, one key per page).
_KIND_CONFIG: dict[str, dict[str, Any]] = {
    "diary": {
        "table": "ai_diary",
        "allowed_fields": ("title", "content"),
        "entry_key": "diary:{id}",
        "localized_cache": True,
        "index_keys": ("diary:index", "diary:nav"),
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
        "index_keys": ("post:index", "post:nav"),
    },
    "curation": {
        "table": "hub_curations",
        "allowed_fields": (
            "title", "source_url", "source_title", "source_author",
            "source_publication", "source_published_at",
            "selection_rationale", "context", "tags",
        ),
        "where_field": "slug",
        "index_keys": (),
    },
}


EDIT_PUBLIC_POST_TOOL = {
    "name": "edit_public_post",
    "description": (
        "Edit a public-facing post AND invalidate its Redis cache in one step. "
        "Use this instead of query_db when correcting already-published content; "
        "a raw UPDATE leaves the per-entry cache stale so readers keep seeing the "
        "old version (tasks 587/588). "
        "kind='diary' (ai_diary, fields: title, content), "
        "kind='report' (telegram_tasks, fields: content, result), "
        "kind='post' (posts, fields: title, content), "
        "kind='curation' (hub_curations, fields: title, source_url, source_title, "
        "source_author, source_publication, source_published_at, selection_rationale, "
        "context, tags). Provide post_id for diary/report/post, or slug for curation, "
        "plus at least one field for the chosen kind. For a narrow correction, pass "
        "`field`, `replace_old`, and `replace_new`; the tool reads the current field, "
        "shows matching snippets with about 10 characters of surrounding context, and "
        "updates only when the match is unambiguous unless `replace_all=true`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["diary", "report", "post", "curation"],
                "description": "Which public collection the row belongs to.",
            },
            "post_id": {
                "type": "integer",
                "description": "Primary-key id of the row to update. Required unless kind='curation'.",
            },
            "slug": {
                "type": "string",
                "description": "Hub curation slug. Required when kind='curation'.",
            },
            "title": {
                "type": "string",
                "description": "New title. Valid for diary, post, and curation.",
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
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Replacement curation tag list. Valid for curation only.",
            },
            "field": {
                "type": "string",
                "description": (
                    "Surgical mode only. Editable text field to modify, e.g. content, result, "
                    "title, context, selection_rationale."
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
        "required": ["kind"],
    },
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


async def _exec_edit_public_post(
    kind: str,
    post_id: int | None = None,
    slug: str | None = None,
    title: str | None = None,
    content: str | None = None,
    result: str | None = None,
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
) -> str:
    kind = (kind or "").strip().lower()
    if kind not in _KIND_CONFIG:
        return f"Error: kind must be one of {sorted(_KIND_CONFIG)}."

    cfg = _KIND_CONFIG[kind]
    allowed = cfg["allowed_fields"]

    if kind == "curation":
        slug = (slug or "").strip().lower()
        if not slug:
            return "Error: slug is required for kind='curation'."
        if not re.match(r"^[a-z0-9][a-z0-9-]{0,79}$", slug):
            return "Error: slug must match ^[a-z0-9][a-z0-9-]{0,79}$."
        target = slug
    else:
        try:
            target = int(post_id)
        except (TypeError, ValueError):
            return "Error: post_id must be an integer."

    provided = {
        "title": title,
        "content": content,
        "result": result,
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
    if kind == "curation":
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
    fields_str = ", ".join(f for f, _ in updates)
    if cache["ok"]:
        cache_note = f"invalidated {cache['deleted']} Redis key(s)"
    else:
        entry_key = cfg.get("entry_key")
        manual_key = entry_key.format(id=target) if entry_key else "(no per-entry cache)"
        cache_note = (
            f"CACHE INVALIDATION FAILED ({cache['reason']}) — "
            f"run `redis-cli DEL {manual_key}` manually"
        )
    if surgical_requested:
        return (
            f"Updated {kind} {where_field}={target!r}: surgical replace in {field!r} "
            f"({len(spans)} match(es)); {cache_note}.\n"
            f"Matches replaced:\n{snippets}"
        )
    return f"Updated {kind} {where_field}={target!r}: [{fields_str}]; {cache_note}."


POST_EDIT_TOOLS = [EDIT_PUBLIC_POST_TOOL]
POST_EDIT_TOOL_HANDLERS = {"edit_public_post": _exec_edit_public_post}
