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

logger = logging.getLogger(__name__)

_KEY_TTL = 86400  # 24h auto-cleanup for orphaned keys

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


def format_progress_for_context(task_id: int) -> str:
    """Format Redis progress entries as a readable context block for child tasks."""
    entries = get_task_progress(task_id)
    if not entries:
        return ""
    lines = []
    for e in entries:
        status = "ERROR" if e.get("error") else "OK"
        lines.append(
            f"  [{e.get('round', '?')}] {e.get('tool', '?')}({e.get('input', '')}) "
            f"→ [{status}] {e.get('result', '')}"
        )
    return (
        f"<parent-execution-log task_id=\"{task_id}\" entries=\"{len(entries)}\">\n"
        "아래는 서비스 재시작 전에 이 태스크가 이미 수행한 작업 목록이다. "
        "이미 완료된 작업을 반복하지 말고, 중단된 지점부터 이어서 진행하라.\n\n"
        + "\n".join(lines)
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
    """Remove task from active registry."""
    try:
        r = get_redis()
        if not r:
            return
        r.srem("active_tasks", str(task_id))
        clear_task_state(task_id)
        clear_task_progress(task_id)
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
