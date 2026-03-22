"""Mission context system for Telegram bot — shared timeline between chat and tasks.

Uses PostgreSQL (via db.py) instead of SQLite. Missions are per-user.
"""

import logging
from db import query as _query, execute as _execute

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
            # Parse ISO format string
            last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
        hours_since = (now - last_activity).total_seconds() / 3600
        if hours_since >= _MISSION_STALE_HOURS:
            close_mission(mission["id"])
            logger.info("Mission #%d auto-closed (stale %.1fh)", mission["id"], hours_since)
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
    # Close any existing active mission first
    active = get_active_mission(user_id)
    if active:
        close_mission(active["id"])

    rows = _query(
        "INSERT INTO telegram_missions (user_id, title) VALUES (%s, %s) RETURNING id",
        (user_id, title),
    )
    mission_id = rows[0]["id"]

    # Capture recent 5 chat turns as context
    recent = _query(
        "SELECT role, content, created_at FROM telegram_chat_history "
        "WHERE user_id = %s ORDER BY id DESC LIMIT 5",
        (user_id,),
    )
    if recent:
        recent.reverse()
        context_lines = []
        for msg in recent:
            role_label = "사용자" if msg["role"] == "user" else "에이전트"
            text = str(msg["content"] or "")[:300]
            context_lines.append(f"[{msg['created_at']}] {role_label}: {text}")
        context = "\n".join(context_lines)
        _execute(
            "INSERT INTO telegram_mission_events (mission_id, source, event_type, content) "
            "VALUES (%s, %s, %s, %s)",
            (mission_id, "system", "context_capture", context),
        )

    # Link task if provided
    if task_id:
        _execute(
            "UPDATE telegram_tasks SET mission_id = %s WHERE id = %s",
            (mission_id, task_id),
        )
        _execute(
            "INSERT INTO telegram_mission_events (mission_id, source, event_type, content) "
            "VALUES (%s, %s, %s, %s)",
            (mission_id, "system", "task_created", f"Task #{task_id} linked to mission"),
        )

    logger.info("Mission #%d created for user %d: %s", mission_id, user_id, title)
    return {"id": mission_id, "title": title, "status": "active"}


def add_mission_event(
    mission_id: int, source: str, event_type: str, content: str,
) -> None:
    """Add an event to a mission timeline. Only active missions accept events."""
    # Guard: don't write to closed missions
    rows = _query("SELECT status FROM telegram_missions WHERE id = %s", (mission_id,))
    if not rows or rows[0]["status"] != "active":
        return
    _execute(
        "INSERT INTO telegram_mission_events (mission_id, source, event_type, content) "
        "VALUES (%s, %s, %s, %s)",
        (mission_id, source, event_type, content),
    )


def close_mission(mission_id: int) -> str:
    """Close a mission."""
    _execute(
        "UPDATE telegram_missions SET status = 'done', closed_at = NOW() WHERE id = %s",
        (mission_id,),
    )
    return f"Mission #{mission_id} closed."


def build_mission_context(user_id: int) -> str:
    """Build mission context string for system prompt injection."""
    try:
        mission = get_active_mission(user_id)
        if not mission:
            return ""
        events = get_mission_events(mission["id"], limit=10)
        lines = [
            f"\n<active-mission id=\"{mission['id']}\" title=\"{mission['title']}\">",
            f"Started: {mission['created_at']}",
        ]
        if events:
            lines.append("Timeline:")
            for e in events:
                content_preview = str(e["content"] or "")[:200]
                lines.append(
                    f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {content_preview}"
                )
        lines.append("</active-mission>")
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Mission context build failed: %s", e)
        return ""
