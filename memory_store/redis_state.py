"""redis_state.py — Redis-backed short-lived shared state.

Holds incremental task progress (survives process restarts), in-flight web
chat markers, the owner alert queue, task-chain summaries, and the keys other
modules keep here (tool-gateway rate windows, Jev route exhaustion flags).

All operations are fail-safe: Redis unavailability never crashes the bot.
PostgreSQL remains the system of record; Redis is the live state layer.

Leninbot uses Redis database 1. Database 0 of the same server belongs to the
frontend (sessions), so the two never share a keyspace.
"""

import json
import logging
import os
import time

import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from llm.prompt_context import format_task_chain

logger = logging.getLogger(__name__)

_KEY_TTL = 604800  # 7 days — ephemeral keys (progress, web chat markers)
_DEFAULT_URL = "redis://localhost:6379/1"

# ── Connection ────────────────────────────────────────────────────────
#
# Callers run on the Telegram/API event loop, so a hung or absent Redis must
# not cost seconds per call. One client is built once (no per-call PING); any
# connection or timeout error opens a circuit, and for _DOWN_COOLDOWN seconds
# get_redis() returns None at once instead of retrying the connection.

_SOCKET_TIMEOUT = 1.0
_DOWN_COOLDOWN = 30.0

_redis_client = None
_down_until = 0.0


def _open_circuit(exc: Exception) -> None:
    global _down_until
    if time.monotonic() >= _down_until:
        logger.warning("Redis unavailable, skipping it for %.0fs: %s", _DOWN_COOLDOWN, exc)
    _down_until = time.monotonic() + _DOWN_COOLDOWN


class _GuardedRedis:
    """Delegates to a redis client; connection or timeout errors open the circuit."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except (RedisConnectionError, RedisTimeoutError) as e:
                _open_circuit(e)
                raise

        return call


def get_redis():
    """Shared Redis client, or None while Redis is unreachable."""
    global _redis_client
    if time.monotonic() < _down_until:
        return None
    if _redis_client is not None:
        return _redis_client
    url = os.getenv("REDIS_URL", _DEFAULT_URL)
    try:
        client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=_SOCKET_TIMEOUT,
            socket_timeout=_SOCKET_TIMEOUT,
            health_check_interval=30,
        )
        client.ping()
    except Exception as e:
        _open_circuit(e)
        return None
    _redis_client = _GuardedRedis(client)
    logger.info("Redis connected: %s", url)
    return _redis_client


def redis_available() -> bool:
    """Check if Redis is reachable (one PING)."""
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(r.ping())
    except Exception:
        return False


# ── Owner alerts from any process ─────────────────────────────────────
#
# Timer jobs and the roleplay bot have no Telegram token. They queue alerts
# here; the Telegram bot's system_monitor drains the list and DMs the owner.

_OWNER_ALERTS_KEY = "owner_alerts"
_OWNER_ALERTS_MAX = 50


def push_owner_alert(text: str) -> bool:
    """Queue ``text`` for the owner's Telegram DM. False when Redis is down."""
    r = get_redis()
    if r is None:
        return False
    try:
        r.rpush(_OWNER_ALERTS_KEY, text)
        r.ltrim(_OWNER_ALERTS_KEY, -_OWNER_ALERTS_MAX, -1)
        return True
    except Exception as e:
        logger.warning("push_owner_alert failed: %s", e)
        return False


def pop_owner_alerts(limit: int = 10) -> list[str]:
    """Remove and return up to ``limit`` queued owner alerts, oldest first."""
    r = get_redis()
    if r is None:
        return []
    alerts = []
    try:
        for _ in range(limit):
            item = r.lpop(_OWNER_ALERTS_KEY)
            if item is None:
                break
            alerts.append(item)
    except Exception as e:
        logger.warning("pop_owner_alerts failed: %s", e)
    return alerts


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


# ── Active Web Chat Registry ─────────────────────────────────────────

def register_active_web_chat(request_id: str, session_id: str = "", fingerprint: str = ""):
    """Register an in-flight web chat answer generation."""
    try:
        r = get_redis()
        if not r:
            return
        key = f"web_chat:{request_id}:state"
        r.sadd("active_web_chats", request_id)
        r.hset(key, mapping={
            "session_id": session_id,
            "fingerprint": fingerprint[:16],
            "started_at": f"{time.time():.0f}",
            "updated_at": f"{time.time():.0f}",
        })
        r.expire(key, _KEY_TTL)
        r.expire("active_web_chats", _KEY_TTL)
    except Exception as e:
        logger.debug("register_active_web_chat failed (%s): %s", request_id, e)


def unregister_active_web_chat(request_id: str):
    """Remove an in-flight web chat marker."""
    try:
        r = get_redis()
        if not r:
            return
        r.srem("active_web_chats", request_id)
        r.delete(f"web_chat:{request_id}:state")
    except Exception as e:
        logger.debug("unregister_active_web_chat failed (%s): %s", request_id, e)


def get_active_web_chats(max_age_sec: int = 1800) -> list[dict]:
    """Return web chat generations still considered active.

    Stale markers are ignored and removed so a crashed API process does not
    block restarts forever.
    """
    try:
        r = get_redis()
        if not r:
            return []
        now = time.time()
        result = []
        for request_id in r.smembers("active_web_chats"):
            key = f"web_chat:{request_id}:state"
            state = r.hgetall(key)
            if not state:
                r.srem("active_web_chats", request_id)
                continue
            started = float(state.get("started_at") or 0)
            if started and now - started > max_age_sec:
                r.srem("active_web_chats", request_id)
                r.delete(key)
                continue
            state["request_id"] = request_id
            result.append(state)
        return result
    except Exception as e:
        logger.debug("get_active_web_chats failed: %s", e)
        return []


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
    """Drop the progress logs of a closed mission's tasks.

    task_result summaries are kept: chain history is cheap and useful later.
    """
    if not task_ids:
        return
    try:
        r = get_redis()
        if not r:
            return
        r.delete(*[f"task:{tid}:progress" for tid in task_ids])
        logger.info("Cleaned up Redis state for mission #%d (%d tasks)", mission_id, len(task_ids))
    except Exception as e:
        logger.debug("cleanup_mission failed (mission %d): %s", mission_id, e)
