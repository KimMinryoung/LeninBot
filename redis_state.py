"""redis_state.py — Redis-backed live state for orchestrator-worker coordination.

Provides incremental task progress persistence (survives process restarts),
active task registry, and live task state monitoring.

All operations are fail-safe: Redis unavailability never crashes the bot.
PostgreSQL remains the system of record; Redis is the live state layer.
"""

import json
import logging
import os
import time

from prompt_context import format_agent_board, format_task_chain

logger = logging.getLogger(__name__)

_KEY_TTL = 604800  # 7 days — all ephemeral keys (progress, state, board)

# ── Connection ────────────────────────────────────────────────────────

_redis_client = None


def get_redis():
    """Lazy singleton Redis connection."""
    global _redis_client
    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            _redis_client = None

    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
        )
        _redis_client.ping()
        logger.info("Redis connected: %s", url)
        return _redis_client
    except Exception as e:
        logger.warning("Redis unavailable: %s", e)
        _redis_client = None
        return None


def redis_available() -> bool:
    """Check if Redis is reachable."""
    r = get_redis()
    return r is not None


# ── Task Execution Progress (survives restart) ────────────────────────

def save_task_progress(
    task_id: int,
    round_num: int,
    tool_name: str,
    input_summary: str,
    result_snippet: str,
    is_error: bool = False,
):
    """Append a tool execution record to the task's progress log."""
    try:
        r = get_redis()
        if not r:
            return
        key = f"task:{task_id}:progress"
        entry = json.dumps({
            "round": round_num,
            "tool": tool_name,
            "input": input_summary[:300],
            "result": result_snippet[:500],
            "error": is_error,
            "ts": time.time(),
        }, ensure_ascii=False)
        r.rpush(key, entry)
        r.expire(key, _KEY_TTL)
    except Exception as e:
        logger.debug("save_task_progress failed (task %d): %s", task_id, e)


def get_task_progress(task_id: int) -> list[dict]:
    """Retrieve all progress entries for a task."""
    try:
        r = get_redis()
        if not r:
            return []
        key = f"task:{task_id}:progress"
        entries = r.lrange(key, 0, -1)
        return [json.loads(e) for e in entries]
    except Exception as e:
        logger.debug("get_task_progress failed (task %d): %s", task_id, e)
        return []


def clear_task_progress(task_id: int):
    """Remove progress log after task completes (PG has the record)."""
    try:
        r = get_redis()
        if r:
            r.delete(f"task:{task_id}:progress")
    except Exception as e:
        logger.debug("clear_task_progress failed (task %d): %s", task_id, e)


_MAX_PROGRESS_ENTRIES = 30  # cap injected entries to prevent context bloat
_MAX_PROGRESS_CHARS = 8000  # hard limit on total formatted output


def format_progress_for_context(task_id: int) -> str:
    """Format Redis progress entries as a readable context block for child tasks.

    Applies truncation:
    - Caps at _MAX_PROGRESS_ENTRIES (keeps newest)
    - Hard limit on total output chars
    - Condenses read_file results to action-only
    """
    entries = get_task_progress(task_id)
    if not entries:
        return ""

    filtered = entries
    if len(filtered) > _MAX_PROGRESS_ENTRIES:
        skipped = len(filtered) - _MAX_PROGRESS_ENTRIES
        filtered = filtered[-_MAX_PROGRESS_ENTRIES:]
    else:
        skipped = 0

    lines = []
    for e in filtered:
        tool = e.get("tool", "?")
        status = "ERROR" if e.get("error") else "OK"
        input_str = e.get("input", "")
        result_str = e.get("result", "")

        # Condense bulky read results — the child will re-read if needed
        if tool in ("read_file", "list_directory") and not e.get("error"):
            result_str = f"({len(result_str)} chars)"

        lines.append(
            f"  [{e.get('round', '?')}] {tool}({input_str}) → [{status}] {result_str}"
        )

    body = "\n".join(lines)
    if len(body) > _MAX_PROGRESS_CHARS:
        body = body[-_MAX_PROGRESS_CHARS:]
        body = "  ...(truncated)\n" + body[body.index("\n") + 1:]

    skipped_note = f" (earliest {skipped} entries omitted)" if skipped else ""
    return (
        f"<parent-execution-log task_id=\"{task_id}\" entries=\"{len(filtered)}\"{skipped_note}>\n"
        "Below is the list of actions this task already performed before the service restart. "
        "Do not repeat completed work — resume from the point of interruption.\n\n"
        + body
        + "\n</parent-execution-log>"
    )


# ── Live Task State (real-time monitoring) ────────────────────────────

def set_task_state(
    task_id: int,
    round_num: int = 0,
    total_cost: float = 0.0,
    status: str = "running",
    agent_type: str = "",
):
    """Update live task state hash."""
    try:
        r = get_redis()
        if not r:
            return
        key = f"task:{task_id}:state"
        r.hset(key, mapping={
            "round": str(round_num),
            "cost": f"{total_cost:.4f}",
            "status": status,
            "agent_type": agent_type,
            "updated_at": f"{time.time():.0f}",
        })
        r.expire(key, _KEY_TTL)
    except Exception as e:
        logger.debug("set_task_state failed (task %d): %s", task_id, e)


def get_task_state(task_id: int) -> dict | None:
    """Get live task state."""
    try:
        r = get_redis()
        if not r:
            return None
        key = f"task:{task_id}:state"
        data = r.hgetall(key)
        return data if data else None
    except Exception as e:
        logger.debug("get_task_state failed (task %d): %s", task_id, e)
        return None


def get_all_active_tasks() -> list[dict]:
    """Get state of all currently active tasks."""
    try:
        r = get_redis()
        if not r:
            return []
        task_ids = r.smembers("active_tasks")
        result = []
        for tid_str in task_ids:
            state = get_task_state(int(tid_str))
            if state:
                state["task_id"] = tid_str
                result.append(state)
        return result
    except Exception as e:
        logger.debug("get_all_active_tasks failed: %s", e)
        return []


def clear_task_state(task_id: int):
    """Remove live state after task completes."""
    try:
        r = get_redis()
        if r:
            r.delete(f"task:{task_id}:state")
    except Exception as e:
        logger.debug("clear_task_state failed (task %d): %s", task_id, e)


# ── Active Task Registry ─────────────────────────────────────────────

def register_active_task(task_id: int, agent_type: str = "", user_id: int = 0):
    """Register a task as actively processing."""
    try:
        r = get_redis()
        if not r:
            return
        r.sadd("active_tasks", str(task_id))
        set_task_state(task_id, status="starting", agent_type=agent_type)
    except Exception as e:
        logger.debug("register_active_task failed (task %d): %s", task_id, e)


def unregister_active_task(task_id: int):
    """Remove task from active registry. Progress and summary are preserved
    until mission close — they're cheap text and useful for chain context."""
    try:
        r = get_redis()
        if not r:
            return
        r.srem("active_tasks", str(task_id))
        clear_task_state(task_id)
    except Exception as e:
        logger.debug("unregister_active_task failed (task %d): %s", task_id, e)


def get_active_task_ids() -> set[int]:
    """Get set of currently active task IDs."""
    try:
        r = get_redis()
        if not r:
            return set()
        members = r.smembers("active_tasks")
        return {int(m) for m in members}
    except Exception as e:
        logger.debug("get_active_task_ids failed: %s", e)
        return set()


# ── Mission Bulletin Board (inter-agent messaging) ───────────────────

def post_to_board(mission_id: int, from_task_id: int, agent_type: str, message: str):
    """Post a message to the mission bulletin board, visible to all sibling agents."""
    try:
        r = get_redis()
        if not r:
            return
        key = f"board:{mission_id}"
        entry = json.dumps({
            "task_id": from_task_id,
            "agent": agent_type,
            "message": message[:2000],
            "ts": time.time(),
        }, ensure_ascii=False)
        r.rpush(key, entry)
        r.expire(key, _KEY_TTL)
    except Exception as e:
        logger.debug("post_to_board failed (mission %d): %s", mission_id, e)


def read_board(mission_id: int, since_ts: float = 0.0) -> list[dict]:
    """Read messages from the mission bulletin board, optionally filtering by timestamp."""
    try:
        r = get_redis()
        if not r:
            return []
        key = f"board:{mission_id}"
        entries = r.lrange(key, 0, -1)
        result = []
        for e in entries:
            parsed = json.loads(e)
            if parsed.get("ts", 0) > since_ts:
                result.append(parsed)
        return result
    except Exception as e:
        logger.debug("read_board failed (mission %d): %s", mission_id, e)
        return []


def format_board_for_context(mission_id: int, *, provider: str = "claude") -> str:
    """Format board messages as an injectable context block."""
    return format_agent_board(read_board(mission_id), provider)


# ── Task Chain Memory (parent chain context) ─────────────────────────

_CHAIN_TTL = 2592000  # 30 days — task chain history persists until mission cleanup


def save_task_summary(
    task_id: int,
    parent_task_id: int | None,
    agent_type: str,
    content_excerpt: str,
    result_excerpt: str,
    tool_log_excerpt: str = "",
):
    """Save a completed task's summary to Redis for chain context retrieval."""
    try:
        r = get_redis()
        if not r:
            return
        key = f"task_result:{task_id}"
        r.hset(key, mapping={
            "parent_task_id": str(parent_task_id or 0),
            "agent_type": agent_type or "",
            "content": content_excerpt[:500],
            "result": result_excerpt[:1000],
            "tool_log": tool_log_excerpt[:2000],
            "ts": f"{time.time():.0f}",
        })
        r.expire(key, _CHAIN_TTL)
    except Exception as e:
        logger.debug("save_task_summary failed (task %d): %s", task_id, e)


def get_task_summary(task_id: int) -> dict | None:
    """Get a task's cached summary from Redis."""
    try:
        r = get_redis()
        if not r:
            return None
        data = r.hgetall(f"task_result:{task_id}")
        return data if data else None
    except Exception as e:
        logger.debug("get_task_summary failed (task %d): %s", task_id, e)
        return None


def get_task_chain(task_id: int, max_depth: int = 5) -> list[dict]:
    """Walk the parent_task_id chain, loading each ancestor's summary.

    Returns list from oldest ancestor to immediate parent (chronological order).
    Falls back to PostgreSQL if a summary is missing from Redis.
    """
    chain = []
    current_id = task_id
    for _ in range(max_depth):
        summary = get_task_summary(current_id)
        if summary:
            summary["task_id"] = str(current_id)
            chain.append(summary)
            parent = int(summary.get("parent_task_id", 0))
            if parent <= 0:
                break
            current_id = parent
        else:
            # Fall back to PG
            try:
                from db import query_one
                row = query_one(
                    "SELECT id, parent_task_id, agent_type, content, result, tool_log "
                    "FROM telegram_tasks WHERE id = %s",
                    (current_id,),
                )
                if not row:
                    break
                chain.append({
                    "task_id": str(current_id),
                    "parent_task_id": str(row.get("parent_task_id") or 0),
                    "agent_type": row.get("agent_type") or "",
                    "content": (str(row.get("content") or ""))[:500],
                    "result": (str(row.get("result") or ""))[:1000],
                    "tool_log": (str(row.get("tool_log") or ""))[:2000],
                })
                parent = row.get("parent_task_id")
                if not parent:
                    break
                current_id = parent
            except Exception:
                break

    chain.reverse()  # oldest first
    return chain


def format_task_chain_for_context(task_id: int, *, provider: str = "claude") -> str:
    """Format the parent task chain as an injectable context block."""
    return format_task_chain(get_task_chain(task_id), provider)


# ── Mission-scoped Cleanup ────────────────────────────────────────────

def cleanup_mission(mission_id: int, task_ids: list[int] | None = None):
    """Clean up all Redis state for a completed mission.

    Called when a mission is closed. Removes board messages and
    optionally task progress/state/summaries for associated tasks.
    """
    try:
        r = get_redis()
        if not r:
            return
        # Board messages
        r.delete(f"board:{mission_id}")
        # Task-level keys
        if task_ids:
            keys_to_delete = []
            for tid in task_ids:
                keys_to_delete.extend([
                    f"task:{tid}:progress",
                    f"task:{tid}:state",
                    # task_result kept — chain history is cheap and useful for future reference
                ])
            if keys_to_delete:
                r.delete(*keys_to_delete)
        logger.info("Cleaned up Redis state for mission #%d (%d tasks)", mission_id, len(task_ids or []))
    except Exception as e:
        logger.debug("cleanup_mission failed (mission %d): %s", mission_id, e)
