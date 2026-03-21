"""Mission context system — shared timeline between chat and tasks."""

import logging
from local_agent.local_db import query, execute

logger = logging.getLogger(__name__)


def get_active_mission() -> dict | None:
    """Return the most recent active mission, or None."""
    rows = query(
        "SELECT * FROM missions WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
    )
    return rows[0] if rows else None


def get_mission_events(mission_id: int, limit: int = 20) -> list[dict]:
    """Return events for a mission in chronological order."""
    return query(
        "SELECT * FROM mission_events WHERE mission_id = ? ORDER BY created_at ASC LIMIT ?",
        (mission_id, limit),
    )


def create_mission_from_chat(title: str, task_id: int | None = None) -> dict:
    """Create a new mission, capture recent chat context, and optionally link a task."""
    # Close any existing active mission first
    active = get_active_mission()
    if active:
        close_mission(active["id"])

    mission_id = execute(
        "INSERT INTO missions (title) VALUES (?)", (title,)
    )

    # Capture recent 5 chat turns as context
    recent = query(
        "SELECT role, content, created_at FROM conversations ORDER BY id DESC LIMIT 5"
    )
    if recent:
        # Reverse to chronological order
        recent.reverse()
        context_lines = []
        for msg in recent:
            role_label = "사용자" if msg["role"] == "user" else "에이전트"
            text = msg["content"][:300]
            context_lines.append(f"[{msg['created_at']}] {role_label}: {text}")
        context = "\n".join(context_lines)
        execute(
            "INSERT INTO mission_events (mission_id, source, event_type, content) VALUES (?, ?, ?, ?)",
            (mission_id, "system", "context_capture", context),
        )

    # Link task if provided
    if task_id:
        execute("UPDATE tasks SET mission_id = ? WHERE id = ?", (mission_id, task_id))
        execute(
            "INSERT INTO mission_events (mission_id, source, event_type, content) VALUES (?, ?, ?, ?)",
            (mission_id, "system", "task_created", f"Task #{task_id} linked to mission"),
        )

    logger.info("Mission #%d created: %s", mission_id, title)
    return {"id": mission_id, "title": title, "status": "active"}


def add_mission_event(
    mission_id: int, source: str, event_type: str, content: str
) -> int | None:
    """Add an event to a mission timeline. Only active missions accept events."""
    rows = query("SELECT status FROM missions WHERE id = ?", (mission_id,))
    if not rows or rows[0]["status"] != "active":
        return None
    return execute(
        "INSERT INTO mission_events (mission_id, source, event_type, content) VALUES (?, ?, ?, ?)",
        (mission_id, source, event_type, content),
    )


def close_mission(mission_id: int) -> str:
    """Close a mission."""
    execute(
        "UPDATE missions SET status = 'done', closed_at = datetime('now', 'localtime') WHERE id = ?",
        (mission_id,),
    )
    return f"Mission #{mission_id} closed."


async def handle_mission(action: str, **kwargs) -> str:
    """Tool handler for the mission tool."""
    try:
        if action == "status":
            mission = get_active_mission()
            if not mission:
                return "No active mission."
            events = get_mission_events(mission["id"], limit=5)
            lines = [f"Mission #{mission['id']}: {mission['title']} [{mission['status']}]"]
            lines.append(f"Created: {mission['created_at']}")
            if events:
                lines.append(f"\nRecent events ({len(events)}):")
                for e in events:
                    lines.append(f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {e['content'][:200]}")
            return "\n".join(lines)

        elif action == "add_event":
            mission = get_active_mission()
            if not mission:
                return "No active mission to add event to."
            event_type = kwargs.get("event_type", "finding")
            content = kwargs.get("content", "")
            source = kwargs.get("source", "chat")
            if not content:
                return "Error: 'content' required for add_event."
            add_mission_event(mission["id"], source, event_type, content)
            return f"Event added to mission #{mission['id']}: [{event_type}] {content[:100]}"

        elif action == "list_events":
            mission = get_active_mission()
            if not mission:
                return "No active mission."
            limit = kwargs.get("limit", 50)
            events = get_mission_events(mission["id"], limit=limit)
            if not events:
                return f"Mission #{mission['id']}: no events yet."
            lines = [f"Mission #{mission['id']} timeline ({len(events)} events):"]
            for e in events:
                lines.append(f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {e['content'][:200]}")
            return "\n".join(lines)

        elif action == "close":
            mission = get_active_mission()
            if not mission:
                return "No active mission to close."
            result = close_mission(mission["id"])
            return result

        elif action == "create":
            title = kwargs.get("title", "")
            if not title:
                return "Error: 'title' required for create."
            mission = create_mission_from_chat(title)
            return f"Mission #{mission['id']} created: {mission['title']}"

        return f"Unknown mission action: {action}"
    except Exception as e:
        return f"Mission error: {e}"
