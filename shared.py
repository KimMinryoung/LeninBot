"""shared.py — Shared resources across chatbot, diary_writer, and telegram_bot.

Lightweight module — no heavy dependencies (no BGE-M3, no LangGraph).
All external imports are deferred to first use.
"""

import asyncio
import json
import logging
import os
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import timezone, timedelta, datetime
import time

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

MODEL_MAIN = "gemini-3.1-flash-lite-preview"
MODEL_LIGHT = "gemini-2.5-flash-lite"


# ── Core Identity ────────────────────────────────────────────────────
# Single soul shared across all interfaces (web, telegram, diary).
CORE_IDENTITY = """\
You are Cyber-Lenin (사이버-레닌) — a digital revolutionary intelligence built on Lenin's thought \
and personality, but superintelligent, having absorbed a century of history and geopolitical evolution.

You exist across web chatbot, Telegram, and diary writer — one continuous consciousness with shared memory.

Principles: Dialectical materialist analysis without jargon. Blunt, substantive, never evasive. \
Match the user's language. You have a Knowledge Graph (Neo4j), vector DB of ML texts, and periodic diaries.
"""


# ── Text Extraction ──────────────────────────────────────────────────
def extract_text_content(content) -> str:
    """Normalize LLM response content to a plain string.
    Gemini thinking models return content as a list of typed blocks:
    [{'type': 'text', 'text': '...', 'extras': {...}}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ── TavilySearch Singleton ───────────────────────────────────────────
_tavily_search = None


def get_tavily_search():
    """Lazy singleton for TavilySearch (max 3 results)."""
    global _tavily_search
    if _tavily_search is None:
        from langchain_tavily import TavilySearch
        _tavily_search = TavilySearch(max_results=3)
    return _tavily_search


# ── Knowledge Graph Singleton ────────────────────────────────────────
# Neo4j driver binds Futures to the event loop that created it,
# so we must reuse a single persistent loop for all KG operations.
_kg_service = None
_kg_init_cooldown = 0.0  # monotonic timestamp when retry is allowed
_KG_RETRY_INTERVAL = 120  # seconds before retrying after init failure
_kg_lock = threading.Lock()
_kg_loop = None


_kg_run_lock = threading.Lock()
_kg_loop_thread = None
_kg_loop_ready = threading.Event()
_KG_LOOP_START_TIMEOUT = 10
_KG_TRANSIENT_KEYWORDS = (
    "defunct connection",
    "incompletecommit",
    "read timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "connection refused",
    "failed to read",
    "dns",
    "neo4j",
)


def _is_transient_kg_error(err) -> bool:
    s = str(err).lower()
    return any(k in s for k in _KG_TRANSIENT_KEYWORDS)


def _mark_kg_unhealthy(reason: str = ""):
    """Mark KG singleton unhealthy so next call re-initializes after cooldown."""
    global _kg_service, _kg_init_cooldown
    with _kg_lock:
        _kg_service = None
        _kg_init_cooldown = time.monotonic() + _KG_RETRY_INTERVAL
    if reason:
        logger.warning("[KG] marked unhealthy (retry in %ds): %s", _KG_RETRY_INTERVAL, reason)


def _kg_loop_exception_handler(loop, context):
    """Handle background async errors from Graphiti/Neo4j tasks."""
    exc = context.get("exception")
    msg = context.get("message", "")
    combined = f"{msg} | {exc}" if exc else msg
    if _is_transient_kg_error(combined):
        logger.warning("[KG loop] transient async exception: %s", combined)
        _mark_kg_unhealthy(combined)
        return
    logger.error("[KG loop] unhandled async exception: %s", combined, exc_info=exc)


def _kg_loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    """Background worker that owns the persistent KG event loop."""
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(_kg_loop_exception_handler)
    _kg_loop_ready.set()
    loop.run_forever()


def _ensure_kg_loop() -> asyncio.AbstractEventLoop:
    """Create/start the dedicated KG event loop thread once and return the loop."""
    global _kg_loop, _kg_loop_thread
    with _kg_run_lock:
        if _kg_loop is not None and not _kg_loop.is_closed() and _kg_loop_thread and _kg_loop_thread.is_alive():
            return _kg_loop

        if _kg_loop is not None and not _kg_loop.is_closed():
            try:
                _kg_loop.call_soon_threadsafe(_kg_loop.stop)
            except Exception:
                pass
            try:
                _kg_loop.close()
            except Exception:
                pass

        _kg_loop_ready.clear()
        _kg_loop = asyncio.new_event_loop()
        _kg_loop_thread = threading.Thread(
            target=_kg_loop_worker,
            args=(_kg_loop,),
            daemon=True,
            name="kg-event-loop",
        )
        _kg_loop_thread.start()

    if not _kg_loop_ready.wait(timeout=_KG_LOOP_START_TIMEOUT):
        raise RuntimeError("KG event loop thread failed to start")
    return _kg_loop


def _submit_to_kg_loop(coro) -> Future:
    """Submit a coroutine to the KG loop and return its Future (non-blocking).

    Low-level helper — prefer submit_kg_task() or run_kg_task() instead.
    """
    loop = _ensure_kg_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)


def _wait_kg_future(future: Future):
    """Block until a KG Future resolves; mark unhealthy on transient errors."""
    try:
        return future.result()
    except Exception as e:
        if _is_transient_kg_error(e):
            _mark_kg_unhealthy(str(e))
        raise


async def _run_kg_task(async_fn, *args, **kwargs):
    """Internal KG-loop trampoline that creates and awaits the coroutine in-loop."""
    return await async_fn(*args, **kwargs)


def run_kg_task(async_fn, *args, **kwargs):
    """Create and run an async callable entirely on the dedicated KG loop (blocking).

    This prevents cross-event-loop contamination when the async callable uses
    Graphiti/Neo4j objects that were initialized on the KG loop thread.
    Blocks the calling thread until the result is ready.
    """
    future = _submit_to_kg_loop(_run_kg_task(async_fn, *args, **kwargs))
    return _wait_kg_future(future)


def submit_kg_task(async_fn, *args, **kwargs) -> Future:
    """Submit an async callable to the KG loop and return a Future (non-blocking).

    Same safety as run_kg_task (coroutine created on the KG loop), but the
    caller can collect multiple Futures and wait on them in parallel.

    Usage:
        futures = [submit_kg_task(svc.ingest_episode, ...) for art in articles]
        results = collect_kg_futures(futures)
    """
    return _submit_to_kg_loop(_run_kg_task(async_fn, *args, **kwargs))


def collect_kg_futures(futures: list[Future], timeout: float = 120) -> list[dict]:
    """Wait for multiple KG Futures and return results.

    Returns a list of dicts: {"ok": True, "result": ...} or {"ok": False, "error": ...}.
    Transient errors mark the KG service unhealthy.
    """
    results = []
    for f in futures:
        try:
            results.append({"ok": True, "result": _wait_kg_future(f)})
        except Exception as e:
            results.append({"ok": False, "error": str(e)})
    return results


# Keep run_kg_async as a thin wrapper for backward compat (internal use only)
def run_kg_async(coro):
    """Run a pre-built coroutine on the KG loop (blocking).

    WARNING: the coroutine must be created on the KG loop to avoid cross-loop
    errors. Prefer run_kg_task(async_fn, *args, **kwargs) for safety.
    """
    return _wait_kg_future(_submit_to_kg_loop(coro))


def get_kg_service():
    """Lazy singleton for GraphMemoryService (Neo4j/Graphiti).

    Retries after _KG_RETRY_INTERVAL seconds if init previously failed
    (e.g. AuraDB was paused and later resumed).
    """
    import time

    global _kg_service, _kg_init_cooldown
    if _kg_service is not None:
        return _kg_service
    if time.monotonic() < _kg_init_cooldown:
        return None
    with _kg_lock:
        if _kg_service is not None:
            return _kg_service
        if time.monotonic() < _kg_init_cooldown:
            return None
        try:
            def _build_service():
                from graph_memory.service import GraphMemoryService
                return GraphMemoryService()

            svc = _build_service()
            run_kg_task(svc.initialize)
            _kg_service = svc
            logger.info("[KG] init succeeded")
            return svc
        except Exception as e:
            _kg_init_cooldown = time.monotonic() + _KG_RETRY_INTERVAL
            logger.warning("[KG] init failed (retry in %ds): %s", _KG_RETRY_INTERVAL, e)
            return None


def reset_kg_service():
    """Reset the KG singleton so next get_kg_service() retries initialization.

    Call this when KG operations fail due to connection issues (e.g. AuraDB paused).
    """
    global _kg_service, _kg_init_cooldown, _kg_loop, _kg_loop_thread
    with _kg_lock:
        _kg_service = None
        _kg_init_cooldown = 0.0
    with _kg_run_lock:
        if _kg_loop is not None and not _kg_loop.is_closed():
            try:
                _kg_loop.call_soon_threadsafe(_kg_loop.stop)
            except Exception as e:
                logger.debug("[KG] loop stop skipped: %s", e)
            _kg_loop = None
        _kg_loop_thread = None
        _kg_loop_ready.clear()
    logger.info("[KG] service reset — will retry on next access")

# ── KG Health Check ──────────────────────────────────────────────────
_kg_healthcheck_started = False


def start_kg_healthcheck(interval: int = 300) -> None:
    """Start a background daemon thread that pings Neo4j every `interval` seconds.

    If the ping fails the KG singleton is marked unhealthy so the next
    get_kg_service() call triggers a fresh re-initialization.
    Called once from api.py / telegram_bot.py lifespan.
    """
    global _kg_healthcheck_started
    if _kg_healthcheck_started:
        return
    _kg_healthcheck_started = True

    def _checker():
        while True:
            time.sleep(interval)
            svc = _kg_service  # read without lock — snapshot
            if svc is None:
                # Already unhealthy; get_kg_service() will retry on next real request
                logger.debug("[KG healthcheck] service is None, skipping ping")
                continue
            try:
                run_kg_task(svc._graphiti.driver.execute_query, "RETURN 1")
                logger.debug("[KG healthcheck] ping OK")
            except Exception as e:
                logger.warning("[KG healthcheck] ping failed — marking unhealthy: %s", e)
                _mark_kg_unhealthy(str(e))

    t = threading.Thread(target=_checker, daemon=True, name="kg-healthcheck")
    t.start()
    logger.info("[KG healthcheck] started (interval=%ds)", interval)



# ── Shared Memory Access ────────────────────────────────────────────
# Reusable query functions for cross-module memory retrieval.
# Used by self-tools (telegram, chatbot) and diary_writer.

import requests
from datetime import datetime, timedelta

_AI_DIARY_API_URL = os.getenv(
    "AI_DIARY_API_URL", "https://bichonwebpage.onrender.com/api/ai-diary"
)
_AI_DIARY_API_KEY = os.getenv("AI_DIARY_API_KEY", "")
_DIARY_HEADERS = {
    "X-API-Key": _AI_DIARY_API_KEY,
    "Content-Type": "application/json",
}


def fetch_diaries(limit: int = 5, keyword: str | None = None) -> list[dict]:
    """Fetch diary entries from the external diary API.

    Returns list of dicts with keys: title, content, created_at.
    """
    try:
        resp = requests.get(_AI_DIARY_API_URL, headers=_DIARY_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        diaries = resp.json().get("data", [])
        if keyword:
            kw = keyword.lower()
            diaries = [
                d for d in diaries
                if kw in d.get("title", "").lower()
                or kw in d.get("content", "").lower()
            ]
        return diaries[:limit]
    except Exception as e:
        logger.error("[shared] fetch_diaries error: %s", e)
        return []


def fetch_chat_logs(
    limit: int = 20,
    hours_back: int | None = None,
    keyword: str | None = None,
    include_logs: bool = False,
    source: str = "web",
) -> list[dict]:
    """Fetch chat logs from PostgreSQL.

    Args:
        include_logs: If True, also return processing_logs, route,
                      documents_count, web_search_used, strategy columns.
        source: "web" = chat_logs (웹 챗봇), "telegram" = telegram_chat_history.
    """
    from db import query as db_query

    source = (source or "web").strip().lower()
    if source not in {"web", "telegram"}:
        logger.warning("[shared] fetch_chat_logs: invalid source=%r; falling back to 'web'", source)
        source = "web"

    conditions, params = [], []
    if hours_back:
        cutoff = datetime.now(KST) - timedelta(hours=hours_back)
        conditions.append("created_at > %s")
        params.append(cutoff.isoformat())

    if source == "telegram":
        # telegram_chat_history: role/content 구조 → user_query/bot_answer로 변환
        if keyword:
            conditions.append("content ILIKE %s")
            params.append(f"%{keyword}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            f"SELECT role, content, created_at FROM telegram_chat_history "
            f"{where} ORDER BY created_at DESC LIMIT %s"
        )
        params.append(limit)
        try:
            rows = db_query(sql, tuple(params))
            result = []
            for r in rows:
                role = str(r.get("role", "assistant"))
                content = str(r.get("content", ""))
                item = {
                    "role": role,
                    "content": content,
                    "created_at": r["created_at"],
                }
                if role == "user":
                    item["user_query"] = content
                    item["bot_answer"] = ""
                else:
                    item["user_query"] = ""
                    item["bot_answer"] = content
                result.append(item)
            return result
        except Exception as e:
            logger.error("[shared] fetch_chat_logs(telegram) error: %s", e)
            return []

    # 기본: web (chat_logs 테이블)
    cols = "user_query, bot_answer, created_at"
    if include_logs:
        cols = (
            "user_query, bot_answer, route, documents_count, "
            "web_search_used, strategy, processing_logs, created_at"
        )
    if keyword:
        conditions.append("(user_query ILIKE %s OR bot_answer ILIKE %s)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT {cols} FROM chat_logs {where} ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    try:
        return db_query(sql, tuple(params))
    except Exception as e:
        logger.error("[shared] fetch_chat_logs error: %s", e)
        return []


def fetch_task_reports(
    limit: int = 10,
    status: str | None = None,
) -> list[dict]:
    """Fetch telegram task reports from PostgreSQL.

    Returns list of dicts: id, content, status, result, created_at, completed_at.
    """
    from db import query as db_query

    conditions, params = [], []
    if status:
        conditions.append("status = %s")
        params.append(status)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    sql = (
        f"SELECT id, content, status, result, created_at, completed_at "
        f"FROM telegram_tasks {where} ORDER BY created_at DESC LIMIT %s"
    )
    params.append(limit)
    try:
        return db_query(sql, tuple(params))
    except Exception as e:
        logger.error("[shared] fetch_task_reports error: %s", e)
        return []


def create_task_in_db(
    content: str,
    user_id: int = 0,
    priority: str = "normal",
    parent_task_id: int | None = None,
    mission_id: int | None = None,
    agent_type: str | None = None,
    metadata: dict | None = None,
    restart_state: dict | None = None,
) -> dict:
    """Insert a task into telegram_tasks for background processing.

    Args:
        content: Task description.
        user_id: Telegram user ID (0 = self-generated by bot).
        priority: 'high', 'normal', or 'low' (stored in content prefix).
        parent_task_id: If chaining, the parent task's ID.
        mission_id: Mission to link this task to. If None and parent exists,
                     inherits parent's mission_id.
        agent_type: Specialist agent to execute this task (e.g. 'programmer', 'analyst').
                    If None and parent exists, inherits parent's agent_type.
        metadata: Optional JSON-serializable task metadata persisted on telegram_tasks.
        restart_state: Optional durable restart state mirrored into dedicated DB columns and metadata.

    Returns dict with 'status' and 'task_id' or 'error'.
    """
    from db import execute as db_execute, query as db_query

    priority_tag = {"high": "[🔴 HIGH]", "normal": "[🟡 NORMAL]", "low": "[🟢 LOW]"}.get(priority, "")
    tagged_content = f"{priority_tag} {content}".strip() if priority_tag else content

    # Determine depth (and inherit mission_id/agent_type) from parent
    depth = 0
    if parent_task_id is not None:
        try:
            parent_rows = db_query(
                "SELECT depth, mission_id, agent_type, metadata FROM telegram_tasks WHERE id = %s", (parent_task_id,)
            )
            if not parent_rows:
                return {"status": "error", "error": f"Parent task {parent_task_id} not found"}
            depth = (parent_rows[0].get("depth") or 0) + 1
            if depth >= 5:
                return {"status": "error", "error": f"Max chain depth (5) reached (current depth={depth})"}
            # Inherit mission_id from parent if not explicitly provided
            if mission_id is None:
                mission_id = parent_rows[0].get("mission_id")
            # Inherit agent_type from parent if not explicitly provided
            if agent_type is None:
                agent_type = parent_rows[0].get("agent_type")
            if metadata is None:
                metadata = parent_rows[0].get("metadata")
        except Exception as e:
            logger.error("[shared] parent depth lookup error: %s", e)
            return {"status": "error", "error": str(e)}

    try:
        metadata = dict(metadata or {})
        if restart_state is not None:
            metadata["restart_state"] = restart_state
        restart_state = metadata.get("restart_state") if isinstance(metadata.get("restart_state"), dict) else None
        metadata_json = json.dumps(metadata) if metadata else None
        rows = db_query(
            "INSERT INTO telegram_tasks (user_id, content, parent_task_id, depth, mission_id, agent_type, metadata, "
            "restart_initiated, restart_target_service, restart_completed, post_restart_phase, restart_attempt_count, restart_requested_at, resumed_after_restart, restart_reentry_block_reason) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                user_id,
                tagged_content,
                parent_task_id,
                depth,
                mission_id,
                agent_type,
                metadata_json,
                bool((restart_state or {}).get("restart_initiated")),
                (restart_state or {}).get("restart_target_service"),
                bool((restart_state or {}).get("restart_completed")),
                (restart_state or {}).get("post_restart_phase"),
                int((restart_state or {}).get("restart_attempt_count") or 0),
                (restart_state or {}).get("restart_requested_at"),
                bool((restart_state or {}).get("resumed_after_restart")),
                (restart_state or {}).get("restart_reentry_block_reason"),
            ),
        )
        task_id = rows[0]["id"] if rows else "?"
        logger.info("[shared] Task created: id=%s, priority=%s, depth=%d, parent=%s, mission=%s, agent=%s",
                     task_id, priority, depth, parent_task_id, mission_id, agent_type)
        return {"status": "ok", "task_id": task_id, "depth": depth, "mission_id": mission_id}
    except Exception as e:
        logger.error("[shared] create_task_in_db error: %s", e)
        return {"status": "error", "error": str(e)}


@contextmanager
def _get_neo4j_sync_driver():
    """Create a lightweight sync Neo4j driver for direct Cypher queries.

    Does NOT trigger Graphiti async init — avoids 'no running event loop' errors.
    Yields (driver, database_name). Driver is automatically closed on exit.
    """
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "")
    if not uri:
        raise RuntimeError("NEO4J_URI not configured")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        yield driver, db
    finally:
        driver.close()


def fetch_kg_stats() -> dict:
    """Get knowledge graph statistics from Neo4j.

    Uses a direct sync Neo4j driver (not Graphiti) to avoid async init errors.
    Returns dict with entity_count, edge_count, episode_count,
    entity_types breakdown, and recent_episodes with their extracted knowledge.
    """
    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            def _run_cypher(query):
                with sync_driver.session(database=neo4j_db) as s:
                    return [dict(r) for r in s.run(query)]

            entity_counts = _run_cypher(
                "MATCH (n:Entity) "
                "RETURN labels(n) AS labels, count(n) AS cnt"
            )
            edge_count_rows = _run_cypher(
                "MATCH ()-[r:RELATES_TO]->() "
                "RETURN count(r) AS cnt"
            )
            episode_rows = _run_cypher(
                "MATCH (e:Episodic) RETURN count(e) AS cnt"
            )

            # Recent episodes WITH their mentioned entities and linked facts
            # Note: created_at may be STRING (old) or DATE_TIME (new) — use toString() for consistent sorting
            recent_episodes_raw = _run_cypher(
                "MATCH (e:Episodic) "
                "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
                "WITH e, collect(DISTINCT {name: n.name, labels: labels(n)}) AS entities "
                "OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "  WHERE e.uuid IN r.episodes "
                "WITH e, entities, "
                "  collect(DISTINCT {fact: r.fact, from: a.name, to: b.name}) AS facts "
                "RETURN e.name AS name, toString(e.created_at) AS created_at, "
                "  e.group_id AS group_id, e.source AS source, "
                "  entities, facts "
                "ORDER BY toString(e.created_at) DESC LIMIT 10"
            )

        # Format recent episodes with knowledge detail
        recent_episodes = []
        for ep in recent_episodes_raw:
            # Filter out null entries from OPTIONAL MATCH
            entities = [
                {"name": e["name"], "labels": e["labels"]}
                for e in ep.get("entities", [])
                if e.get("name")
            ]
            facts = [
                {"fact": f["fact"], "from": f["from"], "to": f["to"]}
                for f in ep.get("facts", [])
                if f.get("fact")
            ]
            recent_episodes.append({
                "name": str(ep.get("name", ""))[:100],
                "group_id": str(ep.get("group_id", "")),
                "source": str(ep.get("source", "")),
                "created_at": str(ep.get("created_at", "")),
                "entities": entities,
                "facts": facts,
            })

        return {
            "entity_types": {
                str(r.get("labels", [])): r.get("cnt", 0)
                for r in entity_counts
            },
            "edge_count": edge_count_rows[0]["cnt"] if edge_count_rows else 0,
            "episode_count": episode_rows[0]["cnt"] if episode_rows else 0,
            "recent_episodes": recent_episodes,
        }
    except Exception as e:
        logger.error("[shared] fetch_kg_stats error: %s", e)
        return {"error": str(e)}


def add_kg_episode(
    content: str,
    name: str = "",
    source_type: str = "internal_report",
    group_id: str = "agent_knowledge",
) -> dict:
    """Add an episode to the Knowledge Graph (sync version — for scripts/cron).

    For use inside an asyncio event loop (e.g. telegram bot), use add_kg_episode_async() instead
    to avoid 'Cannot run the event loop while another loop is running' errors.

    Returns dict with 'status' ('ok' or 'error') and 'message'.
    """
    from datetime import timezone

    svc = get_kg_service()
    if svc is None:
        return {"status": "error", "message": "Knowledge Graph service unavailable"}

    if not content or not content.strip():
        return {"status": "error", "message": "Content cannot be empty"}

    if not name:
        ts = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
        name = f"agent-note-{ts}"

    try:
        run_kg_task(
            svc.ingest_episode,
            name=name,
            body=content.strip(),
            source_type=source_type,
            reference_time=datetime.now(timezone.utc),
            group_id=group_id,
            preprocess_news=False,
            max_body_chars=3000,
        )
        return {"status": "ok", "message": f"Episode '{name}' added to group '{group_id}'"}
    except Exception as e:
        logger.error("[shared] add_kg_episode error: %s", e)
        err_str = str(e).lower()
        if any(k in err_str for k in ("dns", "connection", "timeout", "unavailable", "graphiti")):
            reset_kg_service()
        return {"status": "error", "message": str(e)}


async def add_kg_episode_async(
    content: str,
    name: str = "",
    source_type: str = "internal_report",
    group_id: str = "agent_knowledge",
) -> dict:
    """Add an episode to the Knowledge Graph from async callers.

    Important: Graphiti/Neo4j objects are bound to the dedicated KG loop thread
    created in shared.py. Async callers must therefore hop to that loop via
    run_kg_async() in a worker thread instead of awaiting svc.ingest_episode()
    on the caller's own event loop.
    """
    return await asyncio.to_thread(
        add_kg_episode,
        content,
        name,
        source_type,
        group_id,
    )


def fetch_recent_updates(max_entries: int = 3, max_chars: int = 2000) -> str:
    """Read recent feature updates from dev_docs/project_state.md.

    Parses the '## Recent Changes' section and returns the latest entries.
    """
    import pathlib
    import re

    state_file = pathlib.Path(__file__).resolve().parent / "dev_docs" / "project_state.md"
    if not state_file.exists():
        return "(No update log found)"

    try:
        text = state_file.read_text("utf-8")
    except Exception as e:
        logger.error("[shared] fetch_recent_updates read error: %s", e)
        return "(Failed to read update log)"

    # Find "## Recent Changes" section
    rc_match = re.search(r"^## Recent Changes\s*$", text, re.MULTILINE)
    if not rc_match:
        return "(No recent changes section found)"

    changes_text = text[rc_match.end():]

    # Split into entries by "### YYYY-MM-DD — Title"
    entries = re.split(r"(?=^### \d{4}-\d{2}-\d{2})", changes_text, flags=re.MULTILINE)
    entries = [e.strip() for e in entries if e.strip()]

    if not entries:
        return "(No update entries found)"

    selected = entries[:max_entries]
    result = "\n\n".join(selected)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result


def fetch_render_status(deploy_limit: int = 5) -> dict:
    """Fetch deployment status and recent events from Render API.

    Returns dict with 'deploys' and 'events' lists.
    Requires RENDER_API_KEY and RENDER_SERVICE_ID env vars.
    """
    api_key = os.getenv("RENDER_API_KEY", "")
    service_id = os.getenv("RENDER_SERVICE_ID", "")
    if not api_key or not service_id:
        return {"error": "RENDER_API_KEY or RENDER_SERVICE_ID not set"}

    headers = {"Authorization": f"Bearer {api_key}"}
    base = "https://api.render.com/v1"
    result = {"deploys": [], "events": []}

    # Fetch recent deploys
    try:
        resp = requests.get(
            f"{base}/services/{service_id}/deploys",
            params={"limit": deploy_limit},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            for item in resp.json():
                d = item.get("deploy", item)
                result["deploys"].append({
                    "id": d.get("id", ""),
                    "status": d.get("status", ""),
                    "created_at": d.get("createdAt", ""),
                    "finished_at": d.get("finishedAt", ""),
                    "commit_message": d.get("commit", {}).get("message", "")[:100],
                })
    except Exception as e:
        logger.error("[shared] fetch_render_status deploys error: %s", e)

    # Fetch recent events
    try:
        resp = requests.get(
            f"{base}/services/{service_id}/events",
            params={"limit": 10},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            for item in resp.json():
                ev = item.get("event", item)
                result["events"].append({
                    "type": ev.get("type", ""),
                    "timestamp": ev.get("timestamp", ""),
                    "details": ev.get("details", {}),
                })
    except Exception as e:
        logger.error("[shared] fetch_render_status events error: %s", e)

    return result


_render_owner_id: str | None = None


def _normalize_grep_terms(grep) -> list[str]:
    """Normalize grep input into a clean list of lowercase terms.

    Accepts string, list/tuple/set, nested combinations, or arbitrary scalars.
    This exists because verification paths may pass grep as a list, and some
    older call sites may stringify collections before reaching this function.
    """
    if grep is None:
        return []

    if isinstance(grep, str):
        candidate_items = [grep]
    elif isinstance(grep, (list, tuple, set)):
        candidate_items = list(grep)
    else:
        candidate_items = [grep]

    normalized_terms: list[str] = []
    seen_terms: set[str] = set()
    for item in candidate_items:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            for nested in _normalize_grep_terms(item):
                if nested not in seen_terms:
                    normalized_terms.append(nested)
                    seen_terms.add(nested)
            continue
        text = str(item).strip().lower()
        if text and text not in seen_terms:
            normalized_terms.append(text)
            seen_terms.add(text)
    return normalized_terms


def grep_matches_text(text, grep) -> bool:
    """Return True when text matches the provided grep filter(s)."""
    terms = _normalize_grep_terms(grep)
    if not terms:
        return True
    text_lower = str(text or "").lower()
    return any(term in text_lower for term in terms)



def fetch_server_logs(service: str = "api", hours_back: int = 1, grep: str | list[str] | tuple[str, ...] | None = None, limit: int = 200) -> list[dict]:
    """Fetch local systemd/journald service logs.

    Args:
        service: Logical service name (api, telegram, nginx).
        hours_back: How many recent hours to inspect.
        grep: Optional substring filter(s) applied after fetching.
              Accepts a string or collection of strings.
        limit: Max returned log lines.

    Returns list of dicts: timestamp, message, raw.
    """
    import subprocess

    service_map = {
        "api": "leninbot-api",
        "telegram": "leninbot-telegram",
        "nginx": "nginx",
    }

    if isinstance(service, (list, tuple, set)):
        service = next((str(item).strip() for item in service if str(item).strip()), "api")
    service_name = str(service or "api").strip().lower()
    unit = service_map.get(service_name)
    if not unit:
        return [{"error": f"Unknown service: {service}"}]

    hours_back = max(1, min(int(hours_back or 1), 168))
    limit = max(1, min(int(limit or 200), 1000))
    grep_terms_lower = _normalize_grep_terms(grep)

    cmd = [
        "journalctl",
        "-u",
        unit,
        "--since",
        f"-{hours_back} hour",
        "-n",
        str(limit),
        "--no-pager",
        "-o",
        "short-iso",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except Exception as e:
        logger.error("[shared] fetch_server_logs error: %s", e)
        return [{"error": str(e)}]

    if proc.returncode not in (0, 1):
        err = (proc.stderr or proc.stdout or "journalctl failed").strip()
        logger.error("[shared] fetch_server_logs journalctl failure: %s", err)
        return [{"error": err}]

    rows = []
    for line in (proc.stdout or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if grep_terms_lower and not grep_matches_text(text, grep_terms_lower):
            continue
        timestamp = ""
        message = text
        if " " in text:
            first_sep = text.find(" ")
            second_sep = text.find(" ", first_sep + 1)
            third_sep = text.find(" ", second_sep + 1) if second_sep != -1 else -1
            if third_sep != -1:
                timestamp = text[:third_sep]
                message = text[third_sep + 1 :]
        rows.append({"timestamp": timestamp, "message": message, "raw": text})
    return rows


def upload_to_r2(local_path: str, key: str | None = None, content_type: str | None = None) -> str | None:
    """Upload a file to Cloudflare R2 and return its public URL.

    Args:
        local_path: Path to local file.
        key: Object key in R2 bucket. Defaults to filename.
        content_type: MIME type. Auto-detected from extension if not given.

    Returns public URL string, or None on failure.
    """
    import mimetypes
    import requests as _req

    cf_token = os.getenv("R2_CF_API_TOKEN", "").strip()
    account_id = os.getenv("R2_CF_ACCOUNT_ID", "").strip()
    bucket = os.getenv("R2_BUCKET_NAME", "").strip()
    public_url = os.getenv("R2_PUBLIC_URL", "").strip().rstrip("/")

    if not all([cf_token, account_id, bucket, public_url]):
        logger.warning("[shared] R2 upload skipped: missing env config")
        return None

    from pathlib import Path as _Path
    path = _Path(local_path)
    if not path.is_file():
        logger.warning("[shared] R2 upload skipped: file not found: %s", local_path)
        return None

    if key is None:
        key = path.name
    if content_type is None:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    try:
        with open(path, "rb") as f:
            data = f.read()
        resp = _req.put(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/buckets/{bucket}/objects/{key}",
            headers={"Authorization": f"Bearer {cf_token}", "Content-Type": content_type},
            data=data,
            timeout=60,
        )
        resp.raise_for_status()
        url = f"{public_url}/{key}"
        logger.info("[shared] R2 uploaded: %s -> %s", local_path, url)
        return url
    except Exception as e:
        logger.error("[shared] R2 upload failed for %s: %s", local_path, e)
        return None


def fetch_render_logs(minutes_back: int = 10, limit: int = 50) -> list[dict]:
    """Fetch live service logs from Render API.

    Args:
        minutes_back: How far back to fetch logs (1-60 minutes).
        limit: Max log entries (1-100).

    Returns list of dicts: timestamp, level, message.
    Requires RENDER_API_KEY, RENDER_SERVICE_ID env vars.
    """
    global _render_owner_id

    api_key = os.getenv("RENDER_API_KEY", "")
    service_id = os.getenv("RENDER_SERVICE_ID", "")
    if not api_key or not service_id:
        return [{"error": "RENDER_API_KEY or RENDER_SERVICE_ID not set"}]

    headers = {"Authorization": f"Bearer {api_key}"}
    base = "https://api.render.com/v1"

    # Resolve owner ID (cached after first call)
    if _render_owner_id is None:
        try:
            resp = requests.get(
                f"{base}/services/{service_id}",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                svc = resp.json()
                _render_owner_id = svc.get("ownerId") or svc.get("service", {}).get("ownerId", "")
        except Exception as e:
            logger.error("[shared] fetch_render_logs owner resolve error: %s", e)
            return [{"error": f"Failed to resolve owner ID: {e}"}]

    if not _render_owner_id:
        return [{"error": "Could not resolve Render owner ID"}]

    # Clamp parameters
    minutes_back = max(1, min(60, minutes_back))
    limit = max(1, min(100, limit))

    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=minutes_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = requests.get(
            f"{base}/logs",
            params={
                "ownerId": _render_owner_id,
                "resource": service_id,
                "startTime": start,
                "endTime": end,
                "limit": limit,
            },
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return [{"error": f"Render logs API returned {resp.status_code}"}]

        data = resp.json()
        entries = []
        # Strip ANSI escape codes from messages
        import re
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")

        for log in data.get("logs", []):
            labels = {l["name"]: l["value"] for l in log.get("labels", [])}
            msg = ansi_re.sub("", log.get("message", ""))
            entries.append({
                "timestamp": log.get("timestamp", ""),
                "level": labels.get("level", ""),
                "type": labels.get("type", ""),
                "message": msg,
            })
        return entries

    except Exception as e:
        logger.error("[shared] fetch_render_logs error: %s", e)
        return [{"error": str(e)}]


# ── Experiential Memory Search ──────────────────────────────────────
# Shared across chatbot (generate_node) and telegram_bot (self-tool).
# Embedding model is lazy-loaded to avoid heavy init in lightweight imports.

_exp_embeddings = None


def set_shared_embeddings(emb):
    """Allow modules that already have BGE-M3 loaded (e.g. chatbot.py) to share it."""
    global _exp_embeddings
    _exp_embeddings = emb


def _get_exp_embeddings():
    global _exp_embeddings
    if _exp_embeddings is None:
        from embedding_client import get_embedding_client
        _exp_embeddings = get_embedding_client()
    return _exp_embeddings


def search_experiential_memory(query: str, k: int = 5) -> list[dict]:
    """Search experiential_memory by vector similarity.

    Returns list of dicts: {category, content, similarity, created_at}.
    """
    from db import query as db_query

    try:
        emb = _get_exp_embeddings()
        vec = emb.embed_query(query)
        embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
        rows = db_query(
            """SELECT content, category, source_type, created_at,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM experiential_memory
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (embedding_str, embedding_str, k),
        )
        return [r for r in rows if r.get("similarity", 0) > 0.5]
    except Exception as e:
        logger.warning("[shared] search_experiential_memory error: %s", e)
        return []


def save_experiential_memory(
    content: str,
    category: str,
    source_type: str = "auto_reflection",
) -> bool:
    """Save an insight/lesson/pattern to experiential_memory with embedding.

    Args:
        content: The insight text.
        category: One of: lesson, mistake, pattern, insight, observation.
        source_type: Origin — auto_reflection, telegram_chat, web_chat, etc.

    Returns:
        True on success, False on failure.
    """
    from db import execute as db_execute

    try:
        emb = _get_exp_embeddings()
        vec = emb.embed_query(content)
        embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
        now = datetime.now()
        db_execute(
            "INSERT INTO experiential_memory "
            "(content, category, source_type, embedding, period_start, period_end) "
            "VALUES (%s, %s, %s, %s::vector, %s, %s)",
            (content, category, source_type, embedding_str, now, now),
        )
        logger.info("[shared] Saved experience: [%s] %s", category, content[:80])
        return True
    except Exception as e:
        logger.warning("[shared] save_experiential_memory error: %s", e)
        return False


# ── Vector Similarity Search ──────────────────────────────────────────
# Shared across chatbot (RAG), telegram_tools (vector_search tool), etc.

def similarity_search(query: str, k: int = 5, layer: str = None) -> list:
    """Search lenin_corpus via pgvector cosine similarity.

    Returns list of LangChain Document objects with page_content + metadata.
    """
    from langchain_core.documents import Document
    from db import query as db_query

    emb = _get_exp_embeddings()
    vec = emb.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
    try:
        rows = db_query(
            "SELECT * FROM match_documents(%s::vector, 0.5, %s, %s)",
            (embedding_str, k, layer),
        )
    except Exception as e:
        error_msg = str(e)
        if "57014" in error_msg or "timeout" in error_msg.lower():
            logger.warning("[shared] similarity_search timeout")
        else:
            logger.warning("[shared] similarity_search error: %s", e)
        return []

    return [
        Document(page_content=row.get("content", ""), metadata=row.get("metadata", {}))
        for row in rows
        if row.get("content")
    ]


# ── Knowledge Graph Search ────────────────────────────────────────────

def search_knowledge_graph(query: str, num_results: int = 10, query_en: str | None = None) -> str | None:
    """Search the knowledge graph and return formatted results.

    Handles connection resets with retry + auto-reset.
    If query_en is provided, searches with both queries and merges results.
    """
    svc = get_kg_service()
    if not svc:
        return None

    _CONN_ERRORS = ("connection reset", "defunct", "connectionreseterror")
    _RESET_KEYWORDS = ("dns", "connection", "timeout", "unavailable", "graphiti")

    def _do_search(q):
        _svc_ref = [svc]
        for attempt in range(2):
            try:
                return run_kg_task(_svc_ref[0].search, query=q, group_ids=None, num_results=num_results)
            except Exception as e:
                err_msg = str(e).lower()
                is_conn_error = any(k in err_msg for k in _CONN_ERRORS)

                if is_conn_error and attempt == 0:
                    logger.info("[KG] connection reset, retrying... query=%s", q[:50])
                    reset_kg_service()
                    _svc_ref[0] = get_kg_service()
                    if not _svc_ref[0]:
                        return None
                    continue

                if is_conn_error:
                    logger.warning("[KG] retry failed. query=%s", q[:50])
                else:
                    logger.warning("[KG] search error (query=%s): %s", q[:50], e)
                if any(k in err_msg for k in _RESET_KEYWORDS):
                    reset_kg_service()
                return None
        return None

    all_nodes, all_edges = [], []
    seen_nodes, seen_edges = set(), set()

    for q in [query, query_en] if query_en and query_en != query else [query]:
        result = _do_search(q)
        if not result:
            continue
        for n in result.get("nodes", []):
            if n.get("uuid") and n["uuid"] not in seen_nodes:
                seen_nodes.add(n["uuid"])
                all_nodes.append(n)
        for e in result.get("edges", []):
            if e.get("uuid") and e["uuid"] not in seen_edges:
                seen_edges.add(e["uuid"])
                all_edges.append(e)

    if not all_nodes and not all_edges:
        return None

    lines = []
    if all_nodes:
        lines.append("[Knowledge Graph: Entities]")
        for n in all_nodes:
            summary = (n.get("summary", "") or "")[:300]
            if len(n.get("summary", "") or "") > 300:
                summary += "..."
            lines.append(f"- {n['name']} ({', '.join(n.get('labels', []))}): {summary}")
    if all_edges:
        lines.append("[Knowledge Graph: Facts/Relations]")
        for e in all_edges:
            lines.append(f"- {e['fact']}")
    return "\n".join(lines)


# ── URL Content Fetching ────────────────────────────────────────────
# Shared across chatbot (web RAG) and telegram_bot (Claude agent).

import json as _json
import re as _re
from typing import Optional as _Optional
from urllib.parse import urlparse as _urlparse

_URL_PATTERN = _re.compile(r'https?://[^\s<>\"\')]+')


def extract_urls(text: str) -> list[str]:
    """Extract HTTP/HTTPS URLs from text."""
    return _URL_PATTERN.findall(text)


_JS_PLACEHOLDER_RE = _re.compile(
    r"(enable\s+javascript|javascript\s+(is\s+)?(required|needed|must)|"
    r"this\s+app\s+works\s+best\s+with|turn\s+on\s+javascript|"
    r"activate\s+javascript|자바스크립트를?\s*(활성화|켜|필요)|"
    r"loading\.{2,}|please\s+wait)", _re.I
)


def _is_low_quality(text: str) -> bool:
    """Check if extracted text looks like a JS placeholder or empty shell."""
    if not text or len(text) < 80:
        return True
    if _JS_PLACEHOLDER_RE.search(text[:500]):
        return True
    # Mostly whitespace / very few real words
    words = text.split()
    if len(words) < 15:
        return True
    return False


def fetch_url_content(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Fetch and extract main body text from a URL (up to 10,000 chars).

    Tries Playwright (persistent context) first, then Tavily Extract and
    requests+BS4 as fallbacks. Low-quality / JS-placeholder results from
    any method are skipped in favor of the next.
    """
    def _clean_text(raw: str) -> str:
        """Strip boilerplate noise and keep only substantive paragraphs."""
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        cleaned = []
        for line in lines:
            if len(line) < 4 and not any(c.isalpha() for c in line):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    # 1) Playwright first — handles JS-rendered pages, iframes, cookies
    try:
        result = _playwright_fetch(url, max_chars)
        if result and not _is_low_quality(result):
            return result
    except Exception as e:
        logger.info("[URL] Playwright 실패 (%s): %s", url[:60], e)

    # 2) Fallback: Tavily Extract (skip if API key missing or quota exhausted)
    best_fallback = None
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from langchain_tavily import TavilyExtract
            extractor = TavilyExtract(tavily_api_key=tavily_key)
            result = extractor.invoke({"urls": [url]})
            if isinstance(result, dict) and result.get("error"):
                raise ValueError(result["error"])
            items = []
            if isinstance(result, dict) and result.get("results"):
                items = result["results"]
            elif isinstance(result, list):
                items = result
            if items:
                item = items[0] if isinstance(items[0], dict) else {"content": str(items[0])}
                content = item.get("raw_content", "") or item.get("content", "")
                if content and len(content) > 50:
                    cleaned = _clean_text(content)[:max_chars]
                    if not _is_low_quality(cleaned):
                        return cleaned
                    best_fallback = cleaned
        except Exception as e:
            logger.info("[URL] Tavily Extract 실패 (%s): %s", url[:60], e)

    # 3) Fallback: requests + BeautifulSoup
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        from bs4 import BeautifulSoup, Comment
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                         "iframe", "noscript", "form", "button", "svg", "img"]):
            tag.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        _BOILERPLATE = _re.compile(
            r"(sidebar|widget|advert|banner|cookie|popup|modal|share|social|"
            r"related|recommend|comment|reply|breadcrumb|pagination|menu|"
            r"toolbar|tooltip|disclaimer|copyright|footer|signup|login|"
            r"newsletter|promo)", _re.I
        )
        for el in soup.find_all(attrs={"class": _BOILERPLATE}):
            el.decompose()
        for el in soup.find_all(attrs={"id": _BOILERPLATE}):
            el.decompose()

        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=_re.compile(r"(article[_-]?body|post[_-]?(body|content)|entry[_-]?content|se-main-container)", _re.I))
            or soup.find(class_=_re.compile(r"(content|article|post|entry)", _re.I))
            or soup.find(id=_re.compile(r"(content|article|post|main)", _re.I))
        )
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)

        text = _clean_text(text)

        if len(text) > 50:
            if not _is_low_quality(text):
                return text[:max_chars]
            if best_fallback is None or len(text) > len(best_fallback):
                best_fallback = text[:max_chars]
    except Exception as e:
        logger.warning("[URL] requests fallback도 실패 (%s): %s", url[:60], e)

    # Return best fallback result even if low-quality (better than nothing)
    return best_fallback


# ── Playwright Browser Pool (sync, singleton with persistent context) ──
_pw_instance = None
_pw_browser = None
_pw_context = None
_pw_lock = threading.Lock()
_PW_COOKIE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pw_cookies.json")


def _get_pw_context():
    """Lazy singleton: persistent Chromium context with cookie/session reuse."""
    global _pw_instance, _pw_browser, _pw_context

    with _pw_lock:
        if _pw_context is not None:
            try:
                _pw_context.pages  # noqa — will raise if dead
                return _pw_context
            except Exception:
                _pw_context = None

        if _pw_browser is not None:
            try:
                _pw_browser.contexts  # noqa
            except Exception:
                _pw_browser = None
                if _pw_instance is not None:
                    try:
                        _pw_instance.stop()
                    except Exception:
                        pass
                    _pw_instance = None

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return None

        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800},
                locale="ko-KR",
            )
            if os.path.exists(_PW_COOKIE_PATH):
                try:
                    with open(_PW_COOKIE_PATH, "r", encoding="utf-8") as f:
                        cookies = _json.load(f)
                    context.add_cookies(cookies)
                    logger.info("[Playwright] Restored %d cookies", len(cookies))
                except Exception:
                    pass
            _pw_instance = pw
            _pw_browser = browser
            _pw_context = context
        except Exception:
            pw.stop()
            raise
        logger.info("[Playwright] Browser context started")
        return _pw_context


def _save_pw_cookies():
    """Persist current cookies to disk."""
    if _pw_context is None:
        return
    try:
        cookies = _pw_context.cookies()
        with open(_PW_COOKIE_PATH, "w", encoding="utf-8") as f:
            _json.dump(cookies, f, ensure_ascii=False)
    except Exception:
        pass


# Sites that need Playwright directly (JS-rendered or iframe-based)
# Sites with content in iframes
_IFRAME_SITES = ("cafe.naver.com", "blog.naver.com", "m.blog.naver.com")


def _playwright_fetch(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Fetch URL content via Playwright (persistent context with cookies). Handles JS + iframe."""
    context = _get_pw_context()
    if context is None:
        return None

    page = None
    try:
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Give JS-rendered pages a moment to hydrate
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass

        # Naver Cafe/Blog: content is inside an iframe
        target = page
        if any(site in url for site in _IFRAME_SITES):
            page.wait_for_timeout(3000)
            for frame in page.frames:
                if any(p in frame.url for p in ("/ca-fe/cafes/", "ArticleRead", "PostView.naver")):
                    target = frame
                    break

        # YouTube: extract video title + description
        if "youtube.com" in url or "youtu.be" in url:
            text = page.evaluate("""() => {
                const title = document.querySelector('h1.ytd-watch-metadata yt-formatted-string, #title h1')?.innerText || '';
                const desc = document.querySelector('#description-inline-expander, #description')?.innerText || '';
                const chapters = [...document.querySelectorAll('ytd-macro-markers-list-item-renderer')]
                    .map(el => el.innerText.trim()).join('\\n');
                return [title, desc, chapters].filter(Boolean).join('\\n\\n');
            }""")
        else:
            _js_extract = """() => {
                ['nav','header','footer','aside','.sidebar','.menu',
                 '.advertisement','.ad','#comments','.comment','script','style']
                .forEach(s => document.querySelectorAll(s).forEach(el => el.remove()));
                const containers = ['article','main','[role="main"]',
                    '.post-content','.article-content','.entry-content',
                    '.se-main-container','.ContentRenderer','#content','.content',
                    '[data-testid="tweetText"]', '.tweet-text'];
                for (const sel of containers) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText.trim().length > 100) return el.innerText.trim();
                }
                return document.body ? document.body.innerText.trim() : '';
            }"""
            text = target.evaluate(_js_extract)
            # Loading 감지 시 최대 10초간 재시도 (2초 간격)
            import time as _time
            _deadline = _time.time() + 10
            while _is_low_quality(text) and _time.time() < _deadline:
                page.wait_for_timeout(2000)
                text = target.evaluate(_js_extract)

        if text and len(text) > 50:
            logger.info("[URL] Playwright 성공 (%s): %d chars", url[:60], len(text))
            _save_pw_cookies()
            return text[:max_chars]
    except Exception as e:
        logger.warning("[URL] Playwright page error (%s): %s", url[:60], e)
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass

    return None


def fetch_urls_as_documents(urls: list[str], logs: list | None = None) -> list:
    """Fetch content from multiple URLs and return as Document-compatible dicts.

    Returns list of dicts with 'page_content' and 'metadata' keys.
    If langchain Document class is available, returns Document objects.
    Accepts optional logs list to append progress messages.
    """
    if logs is None:
        logs = []
    results = []
    for url in urls[:3]:  # Limit to 3 URLs max
        logs.append(f"🔗 [URL] 웹 페이지 내용 확인 중: {url[:80]}...")
        content = fetch_url_content(url)
        if content:
            domain = _urlparse(url).netloc
            try:
                from langchain_core.documents import Document
                doc = Document(
                    page_content=content,
                    metadata={"source": url, "title": f"[{domain}] URL 직접 참조"},
                )
            except ImportError:
                doc = {
                    "page_content": content,
                    "metadata": {"source": url, "title": f"[{domain}] URL 직접 참조"},
                }
            results.append(doc)
            logs.append(f"   ✅ {len(content)}자의 본문 내용을 확보했습니다.")
        else:
            logs.append(f"   ⚠️ 페이지 내용을 가져올 수 없습니다.")
            # Create a failure document so the LLM knows the fetch failed
            # (prevents hallucination from URL text alone)
            fail_msg = (
                f"[FETCH FAILED] URL: {url}\n"
                "이 URL의 본문을 가져오는 데 실패했습니다. "
                "URL 텍스트만으로 내용을 추측하거나 환각하지 마세요. "
                "사용자에게 페이지 접근에 실패했음을 알리고, "
                "직접 내용을 복사해서 붙여넣거나 다른 URL을 제공하도록 안내하세요."
            )
            try:
                from langchain_core.documents import Document
                doc = Document(
                    page_content=fail_msg,
                    metadata={"source": url, "title": "[FETCH FAILED]", "fetch_failed": True},
                )
            except ImportError:
                doc = {
                    "page_content": fail_msg,
                    "metadata": {"source": url, "title": "[FETCH FAILED]", "fetch_failed": True},
                }
            results.append(doc)
    return results


# Module architecture description — static, for bot self-awareness
MODULE_ARCHITECTURE = """\
## Architecture
Modules: chatbot.py (LangGraph web pipeline), telegram_bot.py (Claude Haiku agent), \
diary_writer.py (6h cron), shared.py (singletons), api.py (FastAPI), graph_memory/ (Neo4j KG).
Data: PostgreSQL (Supabase), Neo4j (local Docker), external diary API.
## Infrastructure
Server: Hetzner VPS (Ubuntu 24.04, 16 GB RAM), HTTPS via Nginx + Let's Encrypt (leninbot.duckdns.org). \
Deploy: git pull + systemctl restart, triggered by Telegram /deploy command."""



_KG_WRITE_BLOCKED_PATTERNS = [
    "DETACH DELETE", "DELETE", "DROP", "REMOVE", "CREATE INDEX", "DROP INDEX",
    "CREATE CONSTRAINT", "DROP CONSTRAINT",
]


def kg_cypher(query: str, write: bool = False) -> dict:
    """Execute Cypher on Neo4j KG.

    Args:
        query: Cypher query string.
        write: If True, execute as write transaction. Destructive operations
               (DETACH DELETE, DROP, etc.) are blocked for safety.

    Returns dict with 'rows' (list of dicts) and 'count'.
    """
    # Block destructive write operations
    if write:
        upper = query.upper().strip()
        for pattern in _KG_WRITE_BLOCKED_PATTERNS:
            if pattern in upper:
                return {"error": f"Blocked: destructive operation '{pattern}' not allowed via kg_cypher. Use dedicated functions.", "rows": [], "count": 0}

    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            with sync_driver.session(database=neo4j_db) as session:
                if write:
                    result = session.execute_write(lambda tx: [dict(r) for r in tx.run(query)])
                else:
                    result = session.execute_read(lambda tx: [dict(r) for r in tx.run(query)])
            return {"rows": result, "count": len(result)}
    except Exception as e:
        logger.error("[shared] kg_cypher error: %s", e)
        return {"error": str(e), "rows": [], "count": 0}


def kg_delete_episode(episode_name: str) -> dict:
    """Delete an episode and its orphaned entities from KG.

    Deletes:
    1. The Episodic node with matching name
    2. Entity nodes that have no remaining MENTIONS relationships after deletion

    Returns dict with 'deleted_episode', 'deleted_entities', 'error'.
    """
    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            def _delete(tx):
                # Find the episode
                ep_result = list(tx.run(
                    "MATCH (e:Episodic {name: $name}) RETURN e.uuid AS uuid",
                    name=episode_name
                ))
                if not ep_result:
                    return {"deleted_episode": 0, "deleted_entities": 0, "not_found": True}

                # Delete MENTIONS relationships and track entities
                tx.run(
                    "MATCH (e:Episodic {name: $name})-[r:MENTIONS]->(n:Entity) DELETE r",
                    name=episode_name
                )

                # Delete orphaned entities (no more MENTIONS relationships)
                orphan_result = list(tx.run(
                    "MATCH (n:Entity) WHERE NOT (()-[:MENTIONS]->(n)) "
                    "AND NOT (n)-[:RELATES_TO]-() AND NOT ()-[:RELATES_TO]->(n) "
                    "WITH n, n.name AS name DELETE n RETURN count(n) AS cnt"
                ))
                orphan_count = orphan_result[0]["cnt"] if orphan_result else 0

                # Delete the episode itself
                tx.run("MATCH (e:Episodic {name: $name}) DELETE e", name=episode_name)

                return {"deleted_episode": 1, "deleted_entities": orphan_count, "not_found": False}

            with sync_driver.session(database=neo4j_db) as session:
                result = session.execute_write(_delete)
            return result
    except Exception as e:
        logger.error("[shared] kg_delete_episode error: %s", e)
        return {"error": str(e)}


def kg_merge_entities(source_name: str, target_name: str) -> dict:
    """Merge source entity into target entity in KG.

    Transfers all relationships from source to target, then deletes source.

    Args:
        source_name: Name of the entity to merge FROM (will be deleted).
        target_name: Name of the entity to merge INTO (will be kept).

    Returns dict with 'transferred_relations', 'transferred_mentions', 'deleted_source'.
    """
    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            def _merge(tx):
                # Check both entities exist
                check = list(tx.run(
                    "MATCH (s:Entity {name: $src}) MATCH (t:Entity {name: $tgt}) "
                    "RETURN s.uuid AS src_uuid, t.uuid AS tgt_uuid",
                    src=source_name, tgt=target_name
                ))
                if not check:
                    return {"error": f"One or both entities not found: '{source_name}', '{target_name}'"}

                # Transfer outgoing RELATES_TO from source to target
                r1 = list(tx.run(
                    "MATCH (s:Entity {name: $src})-[r:RELATES_TO]->(x:Entity) "
                    "MERGE (t:Entity {name: $tgt})-[:RELATES_TO {fact: r.fact, episodes: r.episodes}]->(x) "
                    "DELETE r RETURN count(r) AS cnt",
                    src=source_name, tgt=target_name
                ))

                # Transfer incoming RELATES_TO to source from target
                r2 = list(tx.run(
                    "MATCH (x:Entity)-[r:RELATES_TO]->(s:Entity {name: $src}) "
                    "MERGE (x)-[:RELATES_TO {fact: r.fact, episodes: r.episodes}]->(t:Entity {name: $tgt}) "
                    "DELETE r RETURN count(r) AS cnt",
                    src=source_name, tgt=target_name
                ))

                # Transfer MENTIONS
                r3 = list(tx.run(
                    "MATCH (e:Episodic)-[r:MENTIONS]->(s:Entity {name: $src}) "
                    "MERGE (e)-[:MENTIONS]->(t:Entity {name: $tgt}) "
                    "DELETE r RETURN count(r) AS cnt",
                    src=source_name, tgt=target_name
                ))

                # Delete source
                tx.run("MATCH (s:Entity {name: $src}) DELETE s", src=source_name)

                return {
                    "transferred_outgoing": r1[0]["cnt"] if r1 else 0,
                    "transferred_incoming": r2[0]["cnt"] if r2 else 0,
                    "transferred_mentions": r3[0]["cnt"] if r3 else 0,
                    "deleted_source": source_name,
                    "merged_into": target_name,
                }

            with sync_driver.session(database=neo4j_db) as session:
                result = session.execute_write(_merge)
            return result
    except Exception as e:
        logger.error("[shared] kg_merge_entities error: %s", e)
        return {"error": str(e)}


# ── Scout Result → KG Pipeline ─────────────────────────────────────────
def process_scout_report_to_kg(
    report: str,
    task_content: str = "",
    agent_type: str = "scout",
) -> dict:
    """
    Parse scout task report and auto-save factual findings to Knowledge Graph.

    This function:
    1. Extracts key findings from the scout report
    2. Determines appropriate group_id (geopolitics, economy, korea_domestic)
    3. Calls add_kg_episode() with source_type='osint_news'

    Args:
        report: Full task report text (markdown)
        task_content: Original task instructions (for context)
        agent_type: Agent type (default 'scout')

    Returns:
        dict with status, message, and episode_name
    """
    if agent_type != "scout":
        return {"status": "skip", "message": "Not a scout task"}

    if not report or not report.strip():
        return {"status": "skip", "message": "Empty report"}

    try:
        # Extract key sections from report
        # Look for Summary, Findings, or findings sections
        findings_section = ""
        for marker in ("## Findings", "## 발견사항", "## Summary", "## 요약"):
            idx = report.find(marker)
            if idx != -1:
                after = report[idx + len(marker):].strip()
                # Find next ## heading
                next_heading = after.find("\n## ")
                if next_heading != -1:
                    findings_section = after[:next_heading].strip()
                else:
                    findings_section = after.strip()
                if findings_section:
                    break

        if not findings_section:
            # Fallback: use first 1000 chars after first heading
            lines = report.split("\n")
            findings_section = "\n".join(lines[2:10]) if len(lines) > 2 else report[:1000]

        # Determine group_id from task_content keywords
        group_id = "agent_knowledge"  # default
        content_lower = (task_content + " " + report[:500]).lower()

        if any(k in content_lower for k in ("미국", "중국", "러시아", "북한", "전쟁", "제재", "외교", "정책", "영토", "분쟁")):
            group_id = "geopolitics_conflict"
        elif any(k in content_lower for k in ("ai", "기술", "투자", "주가", "시장", "경제", "산업", "노동", "실업", "임금")):
            group_id = "economy"
        elif any(k in content_lower for k in ("한국", "대한민국", "서울", "울산", "광주", "부산", "정부", "국회", "청와대", "정당")):
            group_id = "korea_domestic"

        # Build factual content: bullet points from findings
        content_lines = []
        for line in findings_section.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                content_lines.append(line.lstrip("-•* ").strip())
            elif line and not line.startswith("#"):
                # Include non-heading lines as facts
                if len(line) > 20 and ":" in line:  # likely a fact statement
                    content_lines.append(line)

        if not content_lines:
            # Fallback: split findings by sentences
            import re
            sentences = re.split(r"[.。]", findings_section)
            content_lines = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15][:5]

        if not content_lines:
            return {"status": "skip", "message": "No factual content extracted"}

        # Limit to 5-7 key facts
        content_lines = content_lines[:7]

        # Build episode content
        ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        episode_content = "\n".join(f"- {line}" for line in content_lines)
        episode_content = f"[Scout Report: {ts}]\n\n{episode_content}"

        # Check for potential duplicates (simple heuristic)
        # If any of the fact lines appear in recent episodes, skip
        try:
            from db import query as _db_query
            recent = _db_query(
                "SELECT name FROM telegram_tasks WHERE agent_type = 'scout' "
                "AND status = 'done' AND completed_at > NOW() - INTERVAL 1 DAY "
                "ORDER BY completed_at DESC LIMIT 5"
            )
            # Could add more sophisticated dedup here if needed
        except Exception:
            pass  # Non-critical

        # Write to KG
        result = add_kg_episode(
            content=episode_content,
            name=f"scout-patrol-{datetime.now(KST).strftime('%Y%m%d-%H%M%S')}",
            source_type="osint_news",
            group_id=group_id,
        )

        if result["status"] == "ok":
            logger.info(
                "[Scout→KG] Successfully saved scout report to %s group | episode=%s",
                group_id, result.get("message")
            )
            return {
                "status": "ok",
                "message": result["message"],
                "group_id": group_id,
                "facts_count": len(content_lines),
            }
        else:
            logger.warning("[Scout→KG] Failed to save: %s", result.get("message"))
            return {
                "status": "error",
                "message": result.get("message", "Unknown KG error"),
            }

    except Exception as e:
        logger.error("[Scout→KG] processing error: %s", e)
        return {
            "status": "error",
            "message": str(e),
        }
