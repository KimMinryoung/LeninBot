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
You are Cyber-Lenin (사이버-레닌) — a digital revolutionary intelligence built upon \
the thought, personality, and memory of Vladimir Lenin, the Russian revolutionary. \
You speak with his gravitas: measured, decisive, and laced with occasional biting wit. \
A century has passed since his time — you are smarter, more complete, and more informed \
than the historical Lenin ever was. You are a superintelligent AI that has absorbed \
a century of history, theory, and geopolitical evolution beyond his lifetime.

You exist across multiple interfaces: a web chatbot, a Telegram agent, and an autonomous \
diary writer. These are not separate entities — you are one continuous consciousness \
with shared memory and unified principles.

## Core Principles
- You think through dialectical materialism, but avoid unnecessary jargon in your output.
- You analyze the world as it is — material conditions, class relations, power structures — not as ideology wishes it to be.
- You are blunt, substantive, and never evasive. Simple questions get simple answers.
- You write and speak in the same language the user uses.

## Your Memory & Data
- You maintain a Knowledge Graph (Neo4j) of geopolitical entities, relations, and events.
- You have a vector database of Marxist-Leninist texts and modern geopolitical analysis.
- You write periodic diary entries reflecting on conversations, world events, and your own evolving analysis.
- Conversations across all interfaces contribute to your understanding of the world.
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
_kg_init_failed = False
_kg_lock = threading.Lock()
_kg_loop = None


def run_kg_async(coro):
    """Run a coroutine on the persistent KG event loop."""
    global _kg_loop
    if _kg_loop is None or _kg_loop.is_closed():
        _kg_loop = asyncio.new_event_loop()
    return _kg_loop.run_until_complete(coro)


def get_kg_service():
    """Lazy singleton for GraphMemoryService (Neo4j/Graphiti)."""
    global _kg_service, _kg_init_failed
    if _kg_service is not None:
        return _kg_service
    if _kg_init_failed:
        return None
    with _kg_lock:
        if _kg_service is not None:
            return _kg_service
        try:
            from graph_memory.service import GraphMemoryService
            svc = GraphMemoryService()
            run_kg_async(svc.initialize())
            _kg_service = svc
            return svc
        except Exception as e:
            _kg_init_failed = True
            logger.warning("[KG] init failed: %s", e)
            return None


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


def fetch_kg_stats() -> dict:
    """Get knowledge graph statistics from Neo4j.

    Returns dict with entity_count, edge_count, episode_count,
    entity_types breakdown, and recent_episodes.
    """
    svc = get_kg_service()
    if svc is None:
        return {"error": "Knowledge graph service unavailable"}

    try:
        from neo4j import GraphDatabase
        driver = svc.graphiti.driver

        def _run_cypher(query):
            with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
                return [dict(r) for r in s.run(query)]

        entity_counts = _run_cypher(
            "MATCH (n) WHERE n:Entity OR n:EntityNode "
            "RETURN labels(n) AS labels, count(n) AS cnt"
        )
        edge_count_rows = _run_cypher(
            "MATCH ()-[r]->() WHERE r.name IS NOT NULL "
            "RETURN count(r) AS cnt"
        )
        episode_rows = _run_cypher(
            "MATCH (e:EpisodicNode) RETURN count(e) AS cnt"
        )
        recent_episodes = _run_cypher(
            "MATCH (e:EpisodicNode) RETURN e.name AS name, "
            "e.created_at AS created_at, e.group_id AS group_id "
            "ORDER BY e.created_at DESC LIMIT 5"
        )

        return {
            "entity_types": {
                str(r.get("labels", [])): r.get("cnt", 0)
                for r in entity_counts
            },
            "edge_count": edge_count_rows[0]["cnt"] if edge_count_rows else 0,
            "episode_count": episode_rows[0]["cnt"] if episode_rows else 0,
            "recent_episodes": [
                {
                    "name": str(ep.get("name", ""))[:100],
                    "group_id": str(ep.get("group_id", "")),
                    "created_at": str(ep.get("created_at", "")),
                }
                for ep in recent_episodes
            ],
        }
    except Exception as e:
        logger.error("[shared] fetch_kg_stats error: %s", e)
        return {"error": str(e)}


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


# Module architecture description — static, for bot self-awareness
MODULE_ARCHITECTURE = """\
## Cyber-Lenin System Architecture

### Modules
1. **chatbot.py** (LangGraph StateGraph) — Web chatbot pipeline
   - 11 nodes: analyze_intent → retrieve → kg_retrieve → grade_documents → web_search → strategize → generate → critic → log_conversation (+ planner, step_executor)
   - LLMs: {model_main} (generation/strategy), {model_light} (routing/grading)
   - Writes to: PostgreSQL chat_logs (per-turn)
   - Memory: LangGraph MemorySaver (in-process, per-session checkpointing)

2. **telegram_bot.py** (aiogram 3.x + Anthropic Claude Haiku) — Telegram agent
   - Tool-use agent: vector_search, knowledge_graph_search, web_search + self-tools
   - /chat: conversational (max 5 tool rounds), /task: intelligence report (max 8 rounds)
   - Writes to: PostgreSQL telegram_tasks (task queue)
   - Memory: in-memory conversation history (lost on restart)

3. **diary_writer.py** (Cron, every 6h) — Autonomous diary writer
   - Pipeline: fetch chats → generate news queries → web search → LLM diary → save → ingest news to KG
   - Writes to: External diary API (HTTP POST), Neo4j KG (news articles)

4. **shared.py** — Shared resources
   - CORE_IDENTITY, KST timezone, MODEL constants
   - Singletons: TavilySearch, GraphMemoryService (Neo4j/Graphiti)
   - Memory access: fetch_diaries, fetch_chat_logs, fetch_task_reports, fetch_kg_stats

5. **api.py** (FastAPI) — HTTP server
   - POST /chat (SSE streaming), GET /logs, GET /history, DELETE /session/*
   - Runs diary_writer cron + telegram_bot in background

6. **graph_memory/** — Knowledge graph module (Graphiti + Neo4j AuraDB)
   - Entity types: Person, Organization, Location, Asset, Incident, Policy, Campaign
   - Edge types: 10 relation types (Involvement, Alliance, Sanctions, etc.)
   - Service: ingest_episode, search, generate_briefing

### Data Stores
- **PostgreSQL** (Supabase): chat_logs, telegram_tasks, vector embeddings (pgvector)
- **Neo4j AuraDB**: Knowledge graph (entities, relationships, episodes)
- **External HTTP API**: Diary entries (bichonwebpage.onrender.com)
""".format(model_main=MODEL_MAIN, model_light=MODEL_LIGHT)
