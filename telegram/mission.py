"""Mission context system for Telegram bot — shared timeline between chat and tasks.

Uses PostgreSQL (via db.py) instead of SQLite. Missions are per-user.
"""

import logging
from db import query as _query, execute as _execute, get_conn as _get_conn
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


_MISSION_STALE_HOURS = 24


def get_active_mission(user_id: int) -> dict | None:
    """Return the most recent active mission for a user, or None.

    Auto-closes missions with no events in the last 24 hours.
    """
    rows = _query(
        "SELECT * FROM telegram_missions WHERE user_id = %s AND status = 'active' "
        "ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    if not rows:
        return None
    mission = rows[0]
    # Check staleness: last event time or mission creation
    last_event = _query(
        "SELECT created_at FROM telegram_mission_events WHERE mission_id = %s "
        "ORDER BY created_at DESC LIMIT 1",
        (mission["id"],),
    )
    last_activity = last_event[0]["created_at"] if last_event else mission["created_at"]
    try:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        # Ensure last_activity is timezone-aware
        if hasattr(last_activity, 'tzinfo') and last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=timezone.utc)
        elif isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
        hours_since = (now - last_activity).total_seconds() / 3600
        if hours_since >= _MISSION_STALE_HOURS:
            # Don't auto-close if there are pending/processing tasks linked to this mission
            active_tasks = _query(
                "SELECT COUNT(*) AS cnt FROM telegram_tasks "
                "WHERE mission_id = %s AND status IN ('pending', 'processing', 'queued')",
                (mission["id"],),
            )
            if active_tasks and active_tasks[0]["cnt"] > 0:
                logger.debug(
                    "Mission #%d stale (%.1fh) but has %d active tasks — keeping alive",
                    mission["id"], hours_since, active_tasks[0]["cnt"],
                )
            else:
                close_mission(mission["id"])
                logger.info("Mission #%d auto-closed (stale %.1fh, no active tasks)", mission["id"], hours_since)
                return None
    except Exception as e:
        logger.debug("Mission staleness check failed: %s", e)
    return mission


def get_mission_events(mission_id: int, limit: int = 20) -> list[dict]:
    """Return events for a mission in chronological order."""
    return _query(
        "SELECT * FROM telegram_mission_events WHERE mission_id = %s "
        "ORDER BY created_at ASC LIMIT %s",
        (mission_id, limit),
    )


def create_mission(user_id: int, title: str, task_id: int | None = None) -> dict:
    """Create a new mission, capture recent chat context, and optionally link a task."""
    closed_mission_ids: list[int] = []
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"telegram_mission:{user_id}",))

            # Close any existing active mission first. The partial unique index is the
            # durable invariant; this transaction lock keeps normal creation orderly.
            cur.execute(
                "SELECT id FROM telegram_missions WHERE user_id = %s AND status = 'active'",
                (user_id,),
            )
            closed_mission_ids = [int(row["id"]) for row in cur.fetchall()]
            cur.execute(
                "UPDATE telegram_missions "
                "SET status = 'done', closed_at = COALESCE(closed_at, NOW()) "
                "WHERE user_id = %s AND status = 'active'",
                (user_id,),
            )

            cur.execute(
                "INSERT INTO telegram_missions (user_id, title) VALUES (%s, %s) RETURNING id",
                (user_id, title),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("mission insert returned no id")
            mission_id = row["id"]

            # Capture recent 5 chat turns as context.
            cur.execute(
                "SELECT role, content, created_at FROM telegram_chat_history "
                "WHERE user_id = %s ORDER BY id DESC LIMIT 5",
                (user_id,),
            )
            recent = [dict(r) for r in cur.fetchall()]
            if recent:
                recent.reverse()
                context_lines = []
                for msg in recent:
                    role_label = "사용자" if msg["role"] == "user" else "에이전트"
                    text = str(msg["content"] or "")[:300]
                    context_lines.append(f"[{msg['created_at']}] {role_label}: {text}")
                context = "\n".join(context_lines)
                cur.execute(
                    "INSERT INTO telegram_mission_events (mission_id, source, event_type, content) "
                    "VALUES (%s, %s, %s, %s)",
                    (mission_id, "system", "context_capture", context),
                )

            # Link task if provided.
            if task_id:
                cur.execute(
                    "UPDATE telegram_tasks SET mission_id = %s WHERE id = %s",
                    (mission_id, task_id),
                )
                cur.execute(
                    "INSERT INTO telegram_mission_events (mission_id, source, event_type, content) "
                    "VALUES (%s, %s, %s, %s)",
                    (mission_id, "system", "task_created", f"Task #{task_id} linked to mission"),
                )

    for old_mission_id in closed_mission_ids:
        try:
            task_rows = _query("SELECT id FROM telegram_tasks WHERE mission_id = %s", (old_mission_id,))
            task_ids = [r["id"] for r in task_rows] if task_rows else []
            from redis_state import cleanup_mission
            cleanup_mission(old_mission_id, task_ids)
        except Exception:
            logger.debug("Redis cleanup failed for superseded mission #%d", old_mission_id, exc_info=True)

    logger.info("Mission #%d created for user %d: %s", mission_id, user_id, title)
    return {"id": mission_id, "title": title, "status": "active"}


def add_mission_event(
    mission_id: int, source: str, event_type: str, content: str,
) -> None:
    """Add an event to a mission timeline. Only active missions accept events."""
    _execute(
        "INSERT INTO telegram_mission_events (mission_id, source, event_type, content) "
        "SELECT id, %s, %s, %s FROM telegram_missions "
        "WHERE id = %s AND status = 'active'",
        (source, event_type, content, mission_id),
    )


def close_mission(mission_id: int) -> str:
    """Close a mission and clean up associated Redis state."""
    _execute(
        "UPDATE telegram_missions SET status = 'done', closed_at = NOW() WHERE id = %s",
        (mission_id,),
    )
    # Clean up Redis board and ephemeral task state for this mission
    try:
        task_rows = _query(
            "SELECT id FROM telegram_tasks WHERE mission_id = %s",
            (mission_id,),
        )
        task_ids = [r["id"] for r in task_rows] if task_rows else []
        from redis_state import cleanup_mission
        cleanup_mission(mission_id, task_ids)
    except Exception:
        pass  # best-effort cleanup
    return f"Mission #{mission_id} closed."


def build_mission_context(user_id: int, provider: str = "claude") -> str:
    """Build mission context string for system prompt injection.

    `provider` selects the output structure so the injected block matches
    the surrounding system prompt style: "claude" → `<active-mission>` XML
    tag, anything else → `### Active Mission` Markdown heading. Default
    "claude" preserves legacy callers that don't yet pass provider.
    """
    try:
        mission = get_active_mission(user_id)
        if not mission:
            return ""
        events = get_mission_events(mission["id"], limit=20)

        if provider == "claude":
            lines = [
                f"<active-mission id=\"{mission['id']}\" title=\"{mission['title']}\">",
                f"Started: {mission['created_at']}",
            ]
            if events:
                lines.append("Timeline:")
                for e in events:
                    content_preview = str(e["content"] or "")[:500]
                    lines.append(
                        f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {content_preview}"
                    )
            lines.append("</active-mission>")
            return "\n".join(lines)

        # Markdown (OpenAI / Qwen)
        lines = [
            f"### Active Mission #{mission['id']} — \"{mission['title']}\"",
            f"Started: {mission['created_at']}",
        ]
        if events:
            lines.append("")
            lines.append("**Timeline:**")
            for e in events:
                content_preview = str(e["content"] or "")[:500]
                lines.append(
                    f"- [{e['created_at']}] ({e['source']}) {e['event_type']}: {content_preview}"
                )
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Mission context build failed: %s", e)
        return ""
