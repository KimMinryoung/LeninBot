"""publication_records.py — Track Telegram channel posts for public reports."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from db import execute as db_execute, query as db_query, query_one as db_query_one

logger = logging.getLogger(__name__)

_ready = False


def ensure_publish_record_table() -> None:
    global _ready
    if _ready:
        return
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS publish_record (
          id BIGSERIAL PRIMARY KEY,
          slug TEXT NOT NULL,
          public_url TEXT,
          channel_message_id BIGINT,
          channel_message_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
          source TEXT,
          broadcast_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          deleted_at TIMESTAMPTZ,
          delete_ok BOOLEAN,
          delete_error TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    for ddl in (
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS public_url TEXT",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS channel_message_id BIGINT",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS channel_message_ids JSONB NOT NULL DEFAULT '[]'::jsonb",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS source TEXT",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS broadcast_time TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS delete_ok BOOLEAN",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS delete_error TEXT",
        "ALTER TABLE publish_record ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    ):
        db_execute(ddl)
    db_execute(
        """
        CREATE INDEX IF NOT EXISTS publish_record_slug_idx
        ON publish_record(slug, broadcast_time DESC)
        """
    )
    _ready = True


def _normalize_message_ids(channel_message_ids: list[int] | tuple[int, ...] | None) -> list[int]:
    out: list[int] = []
    for value in channel_message_ids or []:
        try:
            msg_id = int(value)
        except (TypeError, ValueError):
            continue
        if msg_id > 0 and msg_id not in out:
            out.append(msg_id)
    return out


def record_publication_broadcast_sync(
    *,
    slug: str,
    public_url: str,
    channel_message_ids: list[int] | tuple[int, ...],
    source: str = "publish_research",
) -> dict | None:
    ensure_publish_record_table()
    clean_slug = (slug or "").strip()
    ids = _normalize_message_ids(channel_message_ids)
    if not clean_slug or not ids:
        return None
    row = db_query_one(
        """
        INSERT INTO publish_record (
          slug, public_url, channel_message_id, channel_message_ids, source, broadcast_time
        )
        VALUES (%s, %s, %s, %s::jsonb, %s, NOW())
        RETURNING *
        """,
        (clean_slug, public_url, ids[0], json.dumps(ids), source),
    )
    return dict(row) if row else None


def list_active_publish_records_sync(slug: str) -> list[dict]:
    ensure_publish_record_table()
    clean_slug = (slug or "").strip()
    if not clean_slug:
        return []
    return db_query(
        """
        SELECT *
          FROM publish_record
         WHERE slug = %s
           AND deleted_at IS NULL
         ORDER BY broadcast_time DESC, id DESC
        """,
        (clean_slug,),
    )


def mark_publish_record_deleted_sync(
    record_id: int,
    *,
    ok: bool,
    error: str | None = None,
) -> None:
    ensure_publish_record_table()
    db_execute(
        """
        UPDATE publish_record
           SET deleted_at = CASE WHEN %s THEN NOW() ELSE deleted_at END,
               delete_ok = %s,
               delete_error = %s
         WHERE id = %s
        """,
        (bool(ok), bool(ok), (error or "")[:1000] if error else None, int(record_id)),
    )


def message_ids_from_record(row: dict[str, Any]) -> list[int]:
    raw = row.get("channel_message_ids")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    ids = _normalize_message_ids(raw if isinstance(raw, list) else [])
    if not ids and row.get("channel_message_id"):
        ids = _normalize_message_ids([row.get("channel_message_id")])
    return ids


async def delete_broadcasts_for_slug(slug: str) -> dict[str, Any]:
    """Best-effort delete of all tracked Telegram channel posts for a slug."""
    import asyncio
    from telegram.channel_broadcast import delete_channel_messages

    records = await asyncio.to_thread(list_active_publish_records_sync, slug)
    if not records:
        return {"attempted": 0, "deleted": 0, "failed": 0, "message": "no tracked channel messages"}

    attempted = 0
    deleted = 0
    failed = 0
    errors: list[str] = []
    for row in records:
        ids = message_ids_from_record(row)
        if not ids:
            await asyncio.to_thread(mark_publish_record_deleted_sync, row["id"], ok=False, error="no message ids")
            failed += 1
            continue
        attempted += len(ids)
        result = await delete_channel_messages(ids)
        deleted += int(result.get("deleted", 0))
        failed += int(result.get("failed", 0))
        ok = bool(result.get("ok"))
        error = None if ok else str(result.get("message") or result.get("errors") or "delete failed")
        if error:
            errors.append(error)
        await asyncio.to_thread(mark_publish_record_deleted_sync, row["id"], ok=ok, error=error)

    return {
        "attempted": attempted,
        "deleted": deleted,
        "failed": failed,
        "message": "; ".join(errors[:3]) if errors else "ok",
    }
