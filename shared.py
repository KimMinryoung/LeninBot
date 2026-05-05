"""shared.py — Shared resources across web_chat, telegram_bot, and agents.

Lightweight module — no heavy dependencies (no BGE-M3, no LangGraph).
All external imports are deferred to first use.
"""

import asyncio
import json
import logging
import os
import socket
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import timezone, timedelta, datetime
import time

from secrets_loader import get_secret

logger = logging.getLogger(__name__)


def get_github_token() -> str:
    """Return configured GitHub token.

    Preferred key: GITHUB_TOKEN. Legacy fallback: GH_TOKEN.
    """
    return (get_secret("GITHUB_TOKEN") or get_secret("GH_TOKEN") or "").strip()

# ── Constants ─────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

MODEL_MAIN = "gemini-3.1-flash-lite-preview"
MODEL_LIGHT = "gemini-2.5-flash-lite"


# ── Broadcast tool compatibility re-exports ────────────────────────
# Compatibility re-exports. New code should import from runtime_tools.broadcast.
from runtime_tools.broadcast import BROADCAST_TO_CHANNEL_TOOL, broadcast_to_channel

# ── Identity and provenance compatibility re-exports ────────────────
# Compatibility re-exports. New code should import from identity.prompts and provenance.runtime.
from identity.prompts import AGENT_CONTEXT, CORE_IDENTITY, EXTERNAL_SOURCE_RULE
from provenance.runtime import (
    ProvenanceBuffer,
    _wrap_external,
    get_provenance_buffer,
    init_provenance_buffer,
)



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


# ── Knowledge Graph runtime compatibility re-exports ────────────────
# Compatibility re-exports. New code should import from kg_runtime.service_runtime.
from kg_runtime.service_runtime import (
    collect_kg_futures,
    get_kg_service,
    reset_kg_service,
    run_kg_async,
    run_kg_task,
    start_kg_healthcheck,
    submit_kg_task,
)

# ── Memory query and task-store compatibility re-exports ────────────
# Compatibility re-exports. New code should import from memory_store.queries and task_store.
from memory_store.queries import fetch_chat_logs, fetch_diaries, fetch_task_reports
from task_store import create_task_in_db

# ── KG stats/write compatibility re-exports ────────────────────────
# Compatibility re-exports. New code should import from kg_runtime.search and kg_runtime.writes.
from kg_runtime.search import _get_neo4j_sync_driver, fetch_kg_stats
from kg_runtime.writes import (
    add_kg_episode,
    add_kg_episode_async,
    add_kg_structured,
    add_kg_structured_async,
)

def fetch_recent_updates(max_entries: int = 3, max_chars: int = 2000) -> str:
    """Deprecated: do not inject dev_docs/project_state.md into agents.

    That file is a human-maintained snapshot and routinely becomes stale. Agents
    should use live state tools, DB-backed task reports, or targeted source files
    instead of treating the snapshot as runtime context.
    """
    return "(Disabled: dev_docs/project_state.md is stale and excluded from agent context.)"


# ── Ops log helper compatibility re-exports ────────────────────────
# Compatibility re-exports. New code should import from ops.logs.
from ops.logs import _normalize_grep_terms, fetch_server_logs, grep_matches_text

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

    cf_token = (get_secret("R2_CF_API_TOKEN", "") or "").strip()
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


# ── Corpus and experiential memory compatibility re-exports ─────────
# Compatibility re-exports. New code should import from corpus.* and memory_store.experiential.
from corpus.embeddings import _get_exp_embeddings, set_shared_embeddings
from corpus.public_index import (
    delete_public_self_analysis_index,
    index_public_self_analysis,
    public_self_analysis_source,
    save_self_produced_analysis,
)
from corpus.store import (
    _chunk_text,
    delete_corpus_source,
    fetch_corpus_source_context,
    ingest_to_corpus,
    similarity_search,
)
from memory_store.experiential import search_experiential_memory, save_experiential_memory

# ── KG search compatibility re-export ──────────────────────────────
# Compatibility re-export. New code should import from kg_runtime.search.
from kg_runtime.search import search_knowledge_graph

# ── Content fetch compatibility re-exports ──────────────────────────
# Compatibility re-exports. New code should import from content_fetch.urls,
# content_fetch.documents, and content_fetch.browser_pool.
from content_fetch.browser_pool import PW_COOKIE_PATH, _PW_COOKIE_PATH
from content_fetch.documents import convert_document
from content_fetch.urls import (
    diagnose_url_fetch_failure,
    extract_urls,
    fetch_url_content,
    fetch_url_content_async,
    fetch_urls_as_documents,
)

# Module architecture description — static, for bot self-awareness
MODULE_ARCHITECTURE = """\
## Architecture
Modules: web_chat.py (claude_loop web pipeline), telegram_bot.py (multi-agent orchestrator), \
agents/ (diary, analyst, scout, programmer, browser, visualizer), shared.py (singletons), api.py (FastAPI), graph_memory/ (Neo4j KG).
Data: PostgreSQL (Supabase), Neo4j (local Docker), Redis (live state).
## Infrastructure
Server: Hetzner VPS (Ubuntu 24.04, 16 GB RAM), HTTPS via Nginx + Let's Encrypt (leninbot.duckdns.org). \
Deploy: git pull + systemctl restart, triggered by Telegram /deploy command."""



# ── KG admin and scout ingest compatibility re-exports ──────────────
# Compatibility re-exports. New code should import from kg_runtime.admin and kg_runtime.scout_ingest.
from kg_runtime.admin import kg_cypher, kg_delete_episode, kg_merge_entities
from kg_runtime.scout_ingest import process_scout_report_to_kg
