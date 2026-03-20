"""shared.py — Shared resources across chatbot, diary_writer, and telegram_bot.

Lightweight module — no heavy dependencies (no BGE-M3, no LangGraph).
All external imports are deferred to first use.
"""

import asyncio
import logging
import threading
from contextlib import contextmanager
from datetime import timezone, timedelta
import time

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


def run_kg_async(coro):
    """Run a coroutine on the persistent KG event loop.

    Thread-safe: serialized via _kg_run_lock because asyncio event loops
    are NOT thread-safe and concurrent run_until_complete() calls crash.
    """
    global _kg_loop
    with _kg_run_lock:
        if _kg_loop is None or _kg_loop.is_closed():
            _kg_loop = asyncio.new_event_loop()
            _kg_loop.set_exception_handler(_kg_loop_exception_handler)
        try:
            return _kg_loop.run_until_complete(coro)
        except Exception as e:
            if _is_transient_kg_error(e):
                _mark_kg_unhealthy(str(e))
            raise


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
            from graph_memory.service import GraphMemoryService
            svc = GraphMemoryService()
            run_kg_async(svc.initialize())
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
    global _kg_service, _kg_init_cooldown, _kg_loop
    with _kg_lock:
        _kg_service = None
        _kg_init_cooldown = 0.0
    with _kg_run_lock:
        if _kg_loop is not None and not _kg_loop.is_closed() and not _kg_loop.is_running():
            try:
                _kg_loop.close()
            except Exception as e:
                logger.debug("[KG] loop close skipped: %s", e)
            finally:
                _kg_loop = None
    logger.info("[KG] service reset — will retry on next access")


# ── Shared Memory Access ────────────────────────────────────────────
# Reusable query functions for cross-module memory retrieval.
# Used by self-tools (telegram, chatbot) and diary_writer.

import os
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
        source: "web" = chat_logs (웹 챗봇), "telegram" = telegram_chat_history,
                "all" = 두 소스 합산 (created_at 기준 정렬).
    """
    from db import query as db_query

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
                if r.get("role") == "user":
                    result.append({"user_query": r["content"], "bot_answer": "", "created_at": r["created_at"]})
                else:
                    result.append({"user_query": "", "bot_answer": r["content"], "created_at": r["created_at"]})
            return result
        except Exception as e:
            logger.error("[shared] fetch_chat_logs(telegram) error: %s", e)
            return []

    elif source == "all":
        # 웹 + 텔레그램 합산
        web = fetch_chat_logs(limit=limit, hours_back=hours_back, keyword=keyword, include_logs=include_logs, source="web")
        tg = fetch_chat_logs(limit=limit, hours_back=hours_back, keyword=keyword, source="telegram")
        combined = web + tg
        combined.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return combined[:limit]

    else:
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
    scratchpad: str = "",
) -> dict:
    """Insert a task into telegram_tasks for background processing.

    Args:
        content: Task description.
        user_id: Telegram user ID (0 = self-generated by bot).
        priority: 'high', 'normal', or 'low' (stored in content prefix).
        parent_task_id: If chaining, the parent task's ID.
        scratchpad: Inherited context from parent task.

    Returns dict with 'status' and 'task_id' or 'error'.
    """
    from db import execute as db_execute, query as db_query

    priority_tag = {"high": "[🔴 HIGH]", "normal": "[🟡 NORMAL]", "low": "[🟢 LOW]"}.get(priority, "")
    tagged_content = f"{priority_tag} {content}".strip() if priority_tag else content

    # Determine depth from parent
    depth = 0
    if parent_task_id is not None:
        try:
            parent_rows = db_query(
                "SELECT depth FROM telegram_tasks WHERE id = %s", (parent_task_id,)
            )
            if not parent_rows:
                return {"status": "error", "error": f"Parent task {parent_task_id} not found"}
            depth = (parent_rows[0].get("depth") or 0) + 1
            if depth >= 5:
                return {"status": "error", "error": f"Max chain depth (5) reached (current depth={depth})"}
        except Exception as e:
            logger.error("[shared] parent depth lookup error: %s", e)
            return {"status": "error", "error": str(e)}

    try:
        rows = db_query(
            "INSERT INTO telegram_tasks (user_id, content, parent_task_id, scratchpad, depth) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (user_id, tagged_content, parent_task_id, scratchpad, depth),
        )
        task_id = rows[0]["id"] if rows else "?"
        logger.info("[shared] Task created: id=%s, priority=%s, depth=%d, parent=%s, content=%s",
                     task_id, priority, depth, parent_task_id, content[:100])
        return {"status": "ok", "task_id": task_id, "depth": depth}
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
            recent_episodes_raw = _run_cypher(
                "MATCH (e:Episodic) "
                "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
                "WITH e, collect(DISTINCT {name: n.name, labels: labels(n)}) AS entities "
                "OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "  WHERE e.uuid IN r.episodes "
                "WITH e, entities, "
                "  collect(DISTINCT {fact: r.fact, from: a.name, to: b.name}) AS facts "
                "RETURN e.name AS name, e.created_at AS created_at, "
                "  e.group_id AS group_id, e.source AS source, "
                "  entities, facts "
                "ORDER BY e.created_at DESC LIMIT 10"
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
    """Add an episode to the Knowledge Graph.

    Args:
        content: The text content to ingest (facts, profiles, observations).
        name: Episode name/label. Auto-generated if empty.
        source_type: One of EPISODE_SOURCE_MAP keys. Default 'internal_report'.
        group_id: Logical grouping. Default 'agent_knowledge'.

    Returns dict with 'status' ('ok' or 'error') and 'message'.
    """
    from datetime import timezone

    svc = get_kg_service()
    if svc is None:
        return {"status": "error", "message": "Knowledge Graph service unavailable"}

    if not content or not content.strip():
        return {"status": "error", "message": "Content cannot be empty"}

    # Auto-generate name if not provided
    if not name:
        ts = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
        name = f"agent-note-{ts}"

    try:
        run_kg_async(svc.ingest_episode(
            name=name,
            body=content.strip(),
            source_type=source_type,
            reference_time=datetime.now(timezone.utc),
            group_id=group_id,
            preprocess_news=False,  # Already curated by the agent
            max_body_chars=3000,
        ))
        return {"status": "ok", "message": f"Episode '{name}' added to group '{group_id}'"}
    except Exception as e:
        logger.error("[shared] add_kg_episode error: %s", e)
        err_str = str(e).lower()
        if any(k in err_str for k in ("dns", "connection", "timeout", "unavailable", "graphiti")):
            reset_kg_service()
        return {"status": "error", "message": str(e)}


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
        from langchain_huggingface import HuggingFaceEmbeddings
        _exp_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
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


# ── URL Content Fetching ────────────────────────────────────────────
# Shared across chatbot (web RAG) and telegram_bot (Claude agent).

import re as _re
from typing import Optional as _Optional
from urllib.parse import urlparse as _urlparse

_URL_PATTERN = _re.compile(r'https?://[^\s<>\"\')]+')


def extract_urls(text: str) -> list[str]:
    """Extract HTTP/HTTPS URLs from text."""
    return _URL_PATTERN.findall(text)


def fetch_url_content(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Fetch and extract main body text from a URL (up to 10,000 chars).

    Tries Tavily Extract first (handles JS-rendered pages),
    falls back to requests + BeautifulSoup with aggressive boilerplate removal.
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

    # Sites that need Playwright directly (JS-rendered, iframe, or API-gated)
    if any(site in url for site in _PLAYWRIGHT_DIRECT_SITES):
        try:
            result = _playwright_fetch(url, max_chars)
            if result and len(result) > 50:
                return result
        except Exception as e:
            logger.info("[URL] Playwright direct 실패 (%s): %s", url[:60], e)
        return None

    # Try Tavily Extract first
    try:
        from langchain_tavily import TavilyExtract
        extractor = TavilyExtract()
        result = extractor.invoke({"urls": [url]})
        items = []
        if isinstance(result, dict) and result.get("results"):
            items = result["results"]
        elif isinstance(result, list):
            items = result
        if items:
            item = items[0] if isinstance(items[0], dict) else {"content": str(items[0])}
            content = item.get("raw_content", "") or item.get("content", "")
            if content and len(content) > 50:
                return _clean_text(content)[:max_chars]
    except Exception as e:
        logger.info("[URL] Tavily Extract 실패 (%s), fallback 시도: %s", url[:60], e)

    # Fallback: requests + BeautifulSoup
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

        # Remove non-content elements aggressively
        for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                         "iframe", "noscript", "form", "button", "svg", "img"]):
            tag.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        # Remove common boilerplate classes/ids
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

        # Try to find the main content container (priority order)
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
            return text[:max_chars]
    except Exception as e:
        logger.warning("[URL] requests fallback도 실패 (%s): %s", url[:60], e)

    # Final fallback: Playwright (handles JS-rendered pages, iframes like Naver Cafe)
    try:
        result = _playwright_fetch(url, max_chars)
        if result and len(result) > 50:
            return result
    except Exception as e:
        logger.info("[URL] Playwright fallback 실패 (%s): %s", url[:60], e)

    return None


# ── Playwright Browser Pool (async, singleton) ──────────────────────
_pw_instance = None
_pw_browser = None
_pw_lock = threading.Lock()


def _get_pw_browser():
    """Lazy singleton: one persistent Chromium instance, reused across calls."""
    global _pw_instance, _pw_browser
    if _pw_browser is not None:
        try:
            # Check if browser is still alive
            _pw_browser.contexts  # noqa — will raise if dead
            return _pw_browser
        except Exception:
            _pw_browser = None
            _pw_instance = None

    with _pw_lock:
        if _pw_browser is not None:
            return _pw_browser
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return None
        _pw_instance = sync_playwright().start()
        _pw_browser = _pw_instance.chromium.launch(headless=True)
        logger.info("[Playwright] Browser pool started")
        return _pw_browser


# Sites that need Playwright directly (JS-rendered or iframe-based)
_PLAYWRIGHT_DIRECT_SITES = (
    "cafe.naver.com", "blog.naver.com", "m.blog.naver.com",
    "twitter.com", "x.com",
    "instagram.com",
    "youtube.com", "youtu.be",
    "threads.net",
)

# Sites with content in iframes
_IFRAME_SITES = ("cafe.naver.com", "blog.naver.com", "m.blog.naver.com")


def _playwright_fetch(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Fetch URL content via Playwright (persistent browser pool). Handles JS + iframe."""
    browser = _get_pw_browser()
    if browser is None:
        return None

    page = None
    try:
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            locale="ko-KR",
        )
        page.goto(url, wait_until="networkidle", timeout=30000)

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
            text = target.evaluate("""() => {
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
            }""")

        if text and len(text) > 50:
            logger.info("[URL] Playwright 성공 (%s): %d chars", url[:60], len(text))
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
Data: PostgreSQL (Supabase), Neo4j AuraDB, external diary API.
## Infrastructure
Server: Hetzner VPS (Ubuntu 24.04, 16 GB RAM), HTTPS via Nginx + Let's Encrypt (leninbot.duckdns.org). \
Deploy: git pull + systemctl restart, triggered by Telegram /deploy command."""



def kg_cypher(query: str, write: bool = False) -> dict:
    """Execute arbitrary Cypher on Neo4j KG.

    Args:
        query: Cypher query string.
        write: If True, execute as write transaction.

    Returns dict with 'rows' (list of dicts) and 'count'.
    """
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
