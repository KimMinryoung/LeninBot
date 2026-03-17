"""shared.py — Shared resources across chatbot, diary_writer, and telegram_bot.

Lightweight module — no heavy dependencies (no BGE-M3, no LangGraph).
All external imports are deferred to first use.
"""

import asyncio
import logging
import threading
from datetime import timezone, timedelta

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


def run_kg_async(coro):
    """Run a coroutine on the persistent KG event loop.

    Thread-safe: serialized via _kg_run_lock because asyncio event loops
    are NOT thread-safe and concurrent run_until_complete() calls crash.
    """
    global _kg_loop
    with _kg_run_lock:
        if _kg_loop is None or _kg_loop.is_closed():
            _kg_loop = asyncio.new_event_loop()
        return _kg_loop.run_until_complete(coro)


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
    global _kg_service, _kg_init_cooldown
    with _kg_lock:
        _kg_service = None
        _kg_init_cooldown = 0.0
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
) -> list[dict]:
    """Fetch chat logs from PostgreSQL.

    Args:
        include_logs: If True, also return processing_logs, route,
                      documents_count, web_search_used, strategy columns.
    """
    from db import query as db_query

    cols = "user_query, bot_answer, created_at"
    if include_logs:
        cols = (
            "user_query, bot_answer, route, documents_count, "
            "web_search_used, strategy, processing_logs, created_at"
        )

    conditions, params = [], []
    if hours_back:
        cutoff = datetime.now(KST) - timedelta(hours=hours_back)
        conditions.append("created_at > %s")
        params.append(cutoff.isoformat())
    if keyword:
        conditions.append("(user_query ILIKE %s OR bot_answer ILIKE %s)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

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


def create_task_in_db(content: str, user_id: int = 0, priority: str = "normal") -> dict:
    """Insert a task into telegram_tasks for background processing.

    Args:
        content: Task description.
        user_id: Telegram user ID (0 = self-generated by bot).
        priority: 'high', 'normal', or 'low' (stored in content prefix).

    Returns dict with 'status' and 'task_id' or 'error'.
    """
    from db import execute as db_execute, query as db_query

    priority_tag = {"high": "[🔴 HIGH]", "normal": "[🟡 NORMAL]", "low": "[🟢 LOW]"}.get(priority, "")
    tagged_content = f"{priority_tag} {content}".strip() if priority_tag else content

    try:
        rows = db_query(
            "INSERT INTO telegram_tasks (user_id, content) VALUES (%s, %s) RETURNING id",
            (user_id, tagged_content),
        )
        task_id = rows[0]["id"] if rows else "?"
        logger.info("[shared] Task created: id=%s, priority=%s, content=%s", task_id, priority, content[:100])
        return {"status": "ok", "task_id": task_id}
    except Exception as e:
        logger.error("[shared] create_task_in_db error: %s", e)
        return {"status": "error", "error": str(e)}


def _get_neo4j_sync_driver():
    """Create a lightweight sync Neo4j driver for direct Cypher queries.

    Does NOT trigger Graphiti async init — avoids 'no running event loop' errors.
    Returns (driver, database_name) or raises on missing config.
    """
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "")
    if not uri:
        raise RuntimeError("NEO4J_URI not configured")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver, db


def fetch_kg_stats() -> dict:
    """Get knowledge graph statistics from Neo4j.

    Uses a direct sync Neo4j driver (not Graphiti) to avoid async init errors.
    Returns dict with entity_count, edge_count, episode_count,
    entity_types breakdown, and recent_episodes with their extracted knowledge.
    """
    try:
        sync_driver, neo4j_db = _get_neo4j_sync_driver()

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

        sync_driver.close()

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

    # Sites that load content in iframes — Tavily/BS4 can't extract, go to Playwright directly
    _IFRAME_SITES = ("cafe.naver.com", "blog.naver.com")
    if any(site in url for site in _IFRAME_SITES):
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


def _playwright_fetch(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Fetch URL content via Playwright (sync). Handles JS + iframe (e.g. Naver Cafe)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                locale="ko-KR",
            )
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Naver Cafe/Blog: content is inside an iframe
            target = page
            if "cafe.naver.com" in url or "blog.naver.com" in url:
                page.wait_for_timeout(3000)
                for frame in page.frames:
                    if any(p in frame.url for p in ("/ca-fe/cafes/", "ArticleRead", "PostView.naver")):
                        target = frame
                        break

            text = target.evaluate("""() => {
                ['nav','header','footer','aside','.sidebar','.menu',
                 '.advertisement','.ad','#comments','.comment','script','style']
                .forEach(s => document.querySelectorAll(s).forEach(el => el.remove()));
                const containers = ['article','main','[role="main"]',
                    '.post-content','.article-content','.entry-content',
                    '.se-main-container','.ContentRenderer','#content','.content'];
                for (const sel of containers) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText.trim().length > 100) return el.innerText.trim();
                }
                return document.body ? document.body.innerText.trim() : '';
            }""")

            page.close()
            if text and len(text) > 50:
                logger.info("[URL] Playwright 성공 (%s): %d chars", url[:60], len(text))
                return text[:max_chars]
        finally:
            browser.close()

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
Data: PostgreSQL (Supabase), Neo4j AuraDB, external diary API."""
