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
import logging
from typing import Any

from db import execute_returning_rowcount as db_exec

logger = logging.getLogger(__name__)


# Per kind: table, which fields the tool may write, which cache keys to purge.
# "*" in a cache key is a SCAN pattern (used for report list pages, one key per page).
_KIND_CONFIG: dict[str, dict[str, Any]] = {
    "diary": {
        "table": "ai_diary",
        "allowed_fields": ("title", "content"),
        "entry_key": "diary:{id}",
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
        "index_keys": ("post:index", "post:nav"),
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
        "kind='post' (posts, fields: title, content). "
        "Provide post_id plus at least one field for the chosen kind."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["diary", "report", "post"],
                "description": "Which public collection the row belongs to.",
            },
            "post_id": {
                "type": "integer",
                "description": "Primary-key id of the row to update.",
            },
            "title": {
                "type": "string",
                "description": "New title. Valid for diary and post.",
            },
            "content": {
                "type": "string",
                "description": "New body text / markdown. Valid for all three kinds.",
            },
            "result": {
                "type": "string",
                "description": "New result body. Valid for report only (the rendered output).",
            },
        },
        "required": ["kind", "post_id"],
    },
}


def _invalidate_cache_sync(kind: str, post_id: int) -> dict[str, Any]:
    """Blocking cache cleanup. Returns {ok, deleted, reason?}. Redis-down is
    soft-failure: we report it back to the agent instead of raising."""
    from redis_state import get_redis

    cfg = _KIND_CONFIG[kind]
    r = get_redis()
    if r is None:
        return {"ok": False, "deleted": 0, "reason": "redis_unavailable"}

    deleted = 0
    try:
        deleted += int(r.delete(cfg["entry_key"].format(id=post_id)) or 0)
        for pattern in cfg["index_keys"]:
            if "*" in pattern:
                for k in r.scan_iter(match=pattern):
                    deleted += int(r.delete(k) or 0)
            else:
                deleted += int(r.delete(pattern) or 0)
    except Exception as e:
        logger.warning("edit_public_post cache invalidation failed (%s:%s): %s", kind, post_id, e)
        return {"ok": False, "deleted": deleted, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "deleted": deleted}


async def _exec_edit_public_post(
    kind: str,
    post_id: int,
    title: str | None = None,
    content: str | None = None,
    result: str | None = None,
) -> str:
    kind = (kind or "").strip().lower()
    if kind not in _KIND_CONFIG:
        return f"Error: kind must be one of {sorted(_KIND_CONFIG)}."
    try:
        post_id = int(post_id)
    except (TypeError, ValueError):
        return "Error: post_id must be an integer."

    cfg = _KIND_CONFIG[kind]
    allowed = cfg["allowed_fields"]

    provided = {"title": title, "content": content, "result": result}
    rejected = [f for f, v in provided.items() if v is not None and f not in allowed]
    if rejected:
        return (
            f"Error: field(s) {rejected} are not editable on kind='{kind}'. "
            f"Allowed: {list(allowed)}."
        )

    updates = [(f, provided[f]) for f in allowed if provided[f] is not None]
    if not updates:
        return f"Error: provide at least one of {list(allowed)} to update."

    set_clause = ", ".join(f"{f} = %s" for f, _ in updates)
    params = tuple(v for _, v in updates) + (post_id,)
    sql = f"UPDATE {cfg['table']} SET {set_clause} WHERE id = %s"

    try:
        affected = await asyncio.to_thread(db_exec, sql, params)
    except Exception as e:
        logger.warning("edit_public_post UPDATE failed (%s id=%s): %s", kind, post_id, e)
        return f"Error: DB update failed: {type(e).__name__}: {e}"

    if not affected:
        return f"Error: no {kind} row found with id={post_id} — nothing updated."

    cache = await asyncio.to_thread(_invalidate_cache_sync, kind, post_id)
    fields_str = ", ".join(f for f, _ in updates)
    if cache["ok"]:
        cache_note = f"invalidated {cache['deleted']} Redis key(s)"
    else:
        cache_note = (
            f"CACHE INVALIDATION FAILED ({cache['reason']}) — "
            f"run `redis-cli DEL {cfg['entry_key'].format(id=post_id)}` manually"
        )
    return f"Updated {kind} id={post_id}: [{fields_str}]; {cache_note}."


POST_EDIT_TOOLS = [EDIT_PUBLIC_POST_TOOL]
POST_EDIT_TOOL_HANDLERS = {"edit_public_post": _exec_edit_public_post}
