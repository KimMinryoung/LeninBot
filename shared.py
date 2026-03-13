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
