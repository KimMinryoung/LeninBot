"""Mission context system for Telegram bot — shared timeline between chat and tasks.

Uses PostgreSQL (via db.py) instead of SQLite. Missions are per-user.
"""

import logging
from db import query as _query, execute as _execute

logger = logging.getLogger(__name__)


def get_active_mission(user_id: int) -> dict | None:
    """Return the most recent active mission for a user, or None."""
    rows = _query(
        "SELECT * FROM telegram_missions WHERE user_id = %s AND status = 'active' "
        "ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    return rows[0] if rows else None


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
    """Add an event to a mission timeline."""
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
            f"\n\n## Active Mission: #{mission['id']} — {mission['title']}",
            f"Started: {mission['created_at']}",
        ]
        if events:
            lines.append("Timeline:")
            for e in events:
                content_preview = str(e["content"] or "")[:200]
                lines.append(
                    f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {content_preview}"
                )
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Mission context build failed: %s", e)
        return ""
