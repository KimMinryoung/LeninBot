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


# ── Telegram channel broadcast tool ─────────────────────────────────
BROADCAST_TO_CHANNEL_TOOL = {
    "name": "broadcast_to_channel",
    "description": (
        "Post a formatted message to Cyber-Lenin's Telegram channel. "
        "Use for public channel announcements only. Message format is fixed: "
        "bold title, 2-3 sentence summary preview, then a plain full-text URL."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Broadcast title. Do not include Markdown link syntax.",
            },
            "summary": {
                "type": "string",
                "description": "Two- or three-sentence preview of the article's core argument.",
            },
            "url": {
                "type": "string",
                "description": "Plain URL where the full article/report can be read.",
            },
            "slug": {
                "type": "string",
                "description": "Optional public research slug for message-id tracking. Usually inferred from url.",
            },
        },
        "required": ["title", "summary", "url"],
    },
}


async def broadcast_to_channel(title: str, summary: str, url: str, **_kw) -> str:
    """Tool handler: send a formatted post to the configured Telegram channel."""
    try:
        from telegram.channel_broadcast import send_broadcast

        result = await send_broadcast(title=title, summary=summary, url=url)
    except Exception as e:
        logger.error("broadcast_to_channel failed: %s", e)
        return f"채널 브로드캐스트 실패: {e}"

    tracking_note = ""
    if result.ok and getattr(result, "message_ids", None):
        try:
            import re as _re
            from publication_records import record_publication_broadcast_sync

            slug = str(_kw.get("slug") or "").strip()
            if not slug:
                m = _re.search(r"/(?:reports/)?research/([^/?#\s]+)", url or "")
                if m:
                    slug = m.group(1).removesuffix(".md")
            if slug:
                await asyncio.to_thread(
                    record_publication_broadcast_sync,
                    slug=slug,
                    public_url=url,
                    channel_message_ids=result.message_ids,
                    source="broadcast_to_channel",
                )
                tracking_note = f"\n추적된 채널 message_id: {len(result.message_ids)}개"
        except Exception as e:
            logger.warning("broadcast_to_channel message-id tracking failed: %s", e)
            tracking_note = f"\n채널 message_id 추적 실패: {e}"

    status = "성공" if result.ok else "실패"
    return (
        f"채널 브로드캐스트 {status}: {result.message}\n"
        f"전송 메시지 수: {result.sent_count}"
        f"{tracking_note}"
    )


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

# ── Shared Memory Access ────────────────────────────────────────────
# Reusable query functions for cross-module memory retrieval.
# Used by self-tools, web_chat, and agents.

from datetime import datetime, timedelta


def fetch_diaries(
    limit: int = 5,
    keyword: str | None = None,
    diary_id: int | None = None,
) -> list[dict]:
    """Fetch diary entries directly from the database.

    Returns list of dicts with keys: id, title, content, created_at, updated_at.
    """
    from db import query as db_query
    try:
        if diary_id is not None:
            return db_query(
                """SELECT id, title, content, created_at, updated_at
                   FROM ai_diary
                   WHERE id = %s
                   LIMIT 1""",
                (diary_id,),
            )
        if keyword:
            kw = f"%{keyword}%"
            return db_query(
                """SELECT id, title, content, created_at, updated_at
                   FROM ai_diary
                   WHERE title ILIKE %s OR content ILIKE %s
                   ORDER BY created_at DESC LIMIT %s""",
                (kw, kw, limit),
            )
        return db_query(
            """SELECT id, title, content, created_at, updated_at
               FROM ai_diary
               ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
    except Exception as e:
        logger.error("[shared] fetch_diaries error: %s", e)
        return []


def fetch_chat_logs(
    limit: int = 20,
    hours_back: int | None = None,
    keyword: str | None = None,
    include_logs: bool = False,
    source: str = "web",
    group_web_contexts: bool = False,
    per_context_limit: int = 10,
) -> list[dict]:
    """Fetch chat logs from PostgreSQL.

    Args:
        include_logs: If True, also return processing_logs, route,
                      documents_count, web_search_used, strategy columns.
        source: "web" = chat_logs (웹 챗봇), "telegram" = telegram_chat_history.
        group_web_contexts: For web logs, fetch recent fingerprint/session
                            contexts, then several turns inside each context.
        per_context_limit: Rows per fingerprint/session context when
                           group_web_contexts=True.
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
    cols = "session_id, fingerprint, user_query, bot_answer, created_at"
    if include_logs:
        cols = (
            "session_id, fingerprint, user_query, bot_answer, route, documents_count, "
            "web_search_used, strategy, processing_logs, created_at"
        )
    if keyword:
        conditions.append("(user_query ILIKE %s OR bot_answer ILIKE %s)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    if group_web_contexts:
        try:
            context_limit = max(1, min(50, int(limit or 20)))
        except (TypeError, ValueError):
            context_limit = 20
        try:
            row_limit = max(1, min(20, int(per_context_limit or 10)))
        except (TypeError, ValueError):
            row_limit = 10
        sql = (
            "WITH filtered AS ("
            f"  SELECT {cols},"
            "         MAX(created_at) OVER (PARTITION BY fingerprint, session_id) AS context_latest,"
            "         ROW_NUMBER() OVER (PARTITION BY fingerprint, session_id ORDER BY created_at DESC) AS context_rank "
            f"  FROM chat_logs {where}"
            "), contexts AS ("
            "  SELECT fingerprint, session_id, MAX(created_at) AS latest "
            "  FROM filtered "
            "  GROUP BY fingerprint, session_id "
            "  ORDER BY latest DESC "
            "  LIMIT %s"
            ") "
            f"SELECT {', '.join('f.' + c.strip() for c in cols.split(','))}, c.latest AS context_latest "
            "FROM filtered f "
            "JOIN contexts c "
            "  ON f.fingerprint IS NOT DISTINCT FROM c.fingerprint "
            " AND f.session_id IS NOT DISTINCT FROM c.session_id "
            "WHERE f.context_rank <= %s "
            "ORDER BY c.latest DESC, f.fingerprint, f.session_id, f.created_at ASC"
        )
        try:
            return db_query(sql, tuple(params) + (context_limit, row_limit))
        except Exception as e:
            logger.error("[shared] fetch_chat_logs grouped web error: %s", e)
            return []

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
    plan_id: int | None = None,
    plan_role: str | None = None,
    status: str = "pending",
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

    tagged_content = content

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
            "INSERT INTO telegram_tasks (user_id, content, status, parent_task_id, depth, mission_id, agent_type, metadata, "
            "plan_id, plan_role, "
            "restart_initiated, restart_target_service, restart_completed, post_restart_phase, restart_attempt_count, restart_requested_at, resumed_after_restart, restart_reentry_block_reason) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                user_id,
                tagged_content,
                status,
                parent_task_id,
                depth,
                mission_id,
                agent_type,
                metadata_json,
                plan_id,
                plan_role,
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


# ── Experiential Memory Search ──────────────────────────────────────
# Shared across chatbot (generate_node) and telegram_bot (self-tool).
# Embedding model is lazy-loaded to avoid heavy init in lightweight imports.

_exp_embeddings = None


def set_shared_embeddings(emb):
    """Allow modules that already have BGE-M3 loaded (e.g. embedding_server.py) to share it."""
    global _exp_embeddings
    _exp_embeddings = emb


def _get_exp_embeddings():
    global _exp_embeddings
    if _exp_embeddings is None:
        from llm.embedding_client import get_embedding_client
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

def similarity_search(query: str, k: int = 5, layer: str = None, rerank: bool = False) -> list:
    """Search lenin_corpus via pgvector cosine similarity.

    Returns list of LangChain Document objects with page_content + metadata.
    When rerank=True, re-scores results with a cross-encoder for better relevance.

    Note on recall: when a layer filter is present, the default HNSW ef_search
    is too small to find enough candidates that also satisfy the layer filter —
    top-k by distance may all be on the wrong layer and get dropped, returning
    an empty result even when the DB has plenty of matches. We bump ef_search
    per-transaction to mitigate this (the SET clause on the SP itself is
    forbidden to non-superusers on managed Postgres).
    """
    from langchain_core.documents import Document
    from psycopg2.extras import RealDictCursor
    from db import get_conn

    emb = _get_exp_embeddings()
    vec = emb.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
    fetch_k = k * 3 if rerank else k
    # Match SP's default threshold (0.4). Kept as a parameter pass-through so
    # the behaviour is visible to the reader without reading the SP.
    threshold = 0.4
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # SET LOCAL so the bump applies only within this transaction.
                # Integer literal is safe; no user input flows through this SET.
                cur.execute("SET LOCAL hnsw.ef_search = 200")
                cur.execute(
                    "SELECT * FROM match_documents(%s::vector, %s, %s, %s)",
                    (embedding_str, threshold, fetch_k, layer),
                )
                rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        error_msg = str(e)
        if "57014" in error_msg or "timeout" in error_msg.lower():
            logger.warning("[shared] similarity_search timeout")
        else:
            logger.warning("[shared] similarity_search error: %s", e)
        return []

    docs = [
        Document(page_content=row.get("content", ""), metadata=row.get("metadata", {}))
        for row in rows
        if row.get("content")
    ]

    if rerank and len(docs) > 2:
        try:
            ranked = emb.rerank(query, [d.page_content for d in docs], top_k=k)
            docs = [docs[idx] for idx, _score in ranked]
        except Exception as e:
            logger.warning("[shared] rerank failed, using original order: %s", e)
            docs = docs[:k]
    else:
        docs = docs[:k]

    return docs


# ── Corpus Ingestion (modern_analysis layer) ─────────────────────────
# Chunking/embedding pipeline used by both hub-curation auto-ingest and the
# manual literature-drop CLI. Keeps insert shape compatible with the existing
# match_documents() SP and the historical chunk size (~760 char avg, 1000 max).

_CORPUS_CHUNK_SIZE = 900
_CORPUS_CHUNK_OVERLAP = 120


def _chunk_text(text: str, size: int = _CORPUS_CHUNK_SIZE, overlap: int = _CORPUS_CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker tuned to match the existing corpus distribution."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break on a newline or whitespace near the boundary for cleaner cuts.
        if end < len(text):
            window = text[max(start, end - 120):end]
            cut_rel = max(window.rfind("\n"), window.rfind(". "), window.rfind(" "))
            if cut_rel > 0:
                end = (end - 120 if end - 120 > start else start) + cut_rel + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def ingest_to_corpus(
    content: str,
    source: str,
    *,
    layer: str = "modern_analysis",
    author: str | None = None,
    year: int | None = None,
    extra_metadata: dict | None = None,
    skip_if_source_url_exists: bool = True,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Chunk → embed → insert into lenin_corpus. Returns the number of chunks inserted.

    By default, skips ingest (returns 0) when extra_metadata['source_url'] already
    exists anywhere in the corpus — prevents duplicate embedding when two curations
    or drops point at the same article. Pass `skip_if_source_url_exists=False` to
    force-insert (e.g. reingest flows that delete the prior rows themselves).
    """
    from db import get_conn

    source_url = (extra_metadata or {}).get("source_url")
    if skip_if_source_url_exists and source_url:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM lenin_corpus WHERE metadata->>'source_url' = %s LIMIT 1",
                    (source_url,),
                )
                if cur.fetchone():
                    logger.info(
                        "[shared] ingest_to_corpus: skipping %r — source_url already in corpus",
                        source_url[:80],
                    )
                    return 0

    size = int(chunk_size or _CORPUS_CHUNK_SIZE)
    overlap = int(chunk_overlap if chunk_overlap is not None else _CORPUS_CHUNK_OVERLAP)
    chunks = _chunk_text(content, size=size, overlap=overlap)
    if not chunks:
        return 0

    emb = _get_exp_embeddings()
    vectors = emb.embed_documents(chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        )

    base_meta = {"layer": layer, "source": source}
    if author:
        base_meta["author"] = author
    if year:
        base_meta["year"] = int(year)
    if extra_metadata:
        base_meta.update(extra_metadata)

    rows = []
    chunk_count = len(chunks)
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        emb_str = "[" + ",".join(str(v) for v in vec) + "]"
        chunk_meta = dict(base_meta)
        chunk_meta.update({
            "chunk_index": idx,
            "chunk_count": chunk_count,
            "chunk_size": size,
            "chunk_overlap": overlap,
        })
        rows.append((chunk, json.dumps(chunk_meta, ensure_ascii=False), emb_str))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO lenin_corpus(content, metadata, embedding) VALUES (%s, %s::jsonb, %s::vector)",
                rows,
            )
        conn.commit()

    return len(chunks)


def save_self_produced_analysis(
    *,
    title: str,
    content: str,
    category: str = "insight",
    source_context: str = "",
) -> dict:
    """Persist the agent's own reusable analysis in a dedicated vector layer."""
    try:
        title = (title or "").strip()
        content = (content or "").strip()
        if not title:
            return {"ok": False, "error": "title is required"}
        if not content:
            return {"ok": False, "error": "content is required"}

        now = datetime.now(timezone.utc)
        source = f"self_analysis:{now.strftime('%Y%m%dT%H%M%SZ')}:{title[:80]}"
        chunks = ingest_to_corpus(
            content,
            source=source,
            layer="self_produced_analysis",
            author="Cyber-Lenin Orchestrator",
            year=now.year,
            extra_metadata={
                "title": title,
                "category": category or "insight",
                "source_context": source_context or "",
                "created_at": now.isoformat(),
                "source_type": "self_produced_analysis",
            },
            skip_if_source_url_exists=False,
        )
        logger.info("[shared] saved self_produced_analysis: %r (%d chunks)", title, chunks)
        return {"ok": True, "chunks": chunks, "source": source, "layer": "self_produced_analysis"}
    except Exception as e:
        logger.warning("[shared] save_self_produced_analysis error: %s", e)
        return {"ok": False, "error": str(e)}


def public_self_analysis_source(kind: str, slug: str) -> str:
    """Stable lenin_corpus metadata.source for public Cyber-Lenin outputs."""
    clean_kind = (kind or "public").strip().lower().replace(":", "_")
    clean_slug = (slug or "").strip().lower()
    return f"public_self_analysis:{clean_kind}:{clean_slug}"


def index_public_self_analysis(
    *,
    kind: str,
    slug: str,
    title: str,
    content: str,
    public_url: str,
    summary: str | None = None,
    content_sha256: str | None = None,
    extra_metadata: dict | None = None,
    chunk_size: int = 3200,
    chunk_overlap: int = 240,
) -> dict:
    """Index a public Cyber-Lenin output into self_produced_analysis.

    The source is stable per public artifact, so publish/edit paths can delete
    and reinsert the same document without creating duplicate chunks.
    """
    try:
        slug = (slug or "").strip()
        title = (title or "").strip()
        content = (content or "").strip()
        if not slug:
            return {"ok": False, "error": "slug is required"}
        if not title:
            return {"ok": False, "error": "title is required"}
        if not content:
            return {"ok": False, "error": "content is required"}

        source = public_self_analysis_source(kind, slug)
        deleted = delete_corpus_source(source, layer="self_produced_analysis")
        now = datetime.now(timezone.utc)
        metadata = {
            "title": title,
            "slug": slug,
            "kind": kind,
            "public_url": public_url,
        }
        if content_sha256:
            metadata["content_sha256"] = content_sha256
        if extra_metadata:
            metadata.update(extra_metadata)

        indexed_content = f"# {title}\n\n"
        if summary:
            indexed_content += f"Summary: {summary.strip()}\n\n"
        indexed_content += content
        chunks = ingest_to_corpus(
            indexed_content,
            source=source,
            layer="self_produced_analysis",
            author="Cyber-Lenin",
            year=now.year,
            extra_metadata=metadata,
            skip_if_source_url_exists=False,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(
            "[shared] indexed public self analysis: %s/%s (%d chunks, %d deleted)",
            kind,
            slug,
            chunks,
            deleted,
        )
        return {
            "ok": True,
            "source": source,
            "layer": "self_produced_analysis",
            "chunks": chunks,
            "deleted": deleted,
        }
    except Exception as e:
        logger.warning("[shared] index_public_self_analysis error: %s", e)
        return {"ok": False, "error": str(e)}


def delete_public_self_analysis_index(kind: str, slug: str) -> dict:
    """Remove one public artifact from the self_produced_analysis vector layer."""
    try:
        source = public_self_analysis_source(kind, slug)
        deleted = delete_corpus_source(source, layer="self_produced_analysis")
        return {"ok": True, "source": source, "deleted": deleted}
    except Exception as e:
        logger.warning("[shared] delete_public_self_analysis_index error: %s", e)
        return {"ok": False, "error": str(e)}


def fetch_corpus_source_context(
    source: str,
    *,
    center_index: int = 0,
    window: int = 1,
    max_chars: int = 9000,
) -> str:
    """Fetch neighboring chunks from one corpus source for parent-context expansion."""
    from db import query as db_query

    source = (source or "").strip()
    if not source:
        return ""
    center_index = max(0, int(center_index or 0))
    window = max(0, int(window or 0))
    max_chars = max(1000, int(max_chars or 9000))
    lo = max(0, center_index - window)
    hi = center_index + window
    try:
        rows = db_query(
            """
            SELECT content, metadata
              FROM lenin_corpus
             WHERE metadata->>'source' = %s
               AND COALESCE((metadata->>'chunk_index')::int, 0) BETWEEN %s AND %s
             ORDER BY COALESCE((metadata->>'chunk_index')::int, 0)
            """,
            (source, lo, hi),
        )
    except Exception as e:
        logger.warning("[shared] fetch_corpus_source_context error: %s", e)
        return ""

    parts: list[str] = []
    total = 0
    for row in rows:
        text = str(row.get("content") or "").strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n... (context truncated)"
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


def delete_corpus_source(source: str, layer: str | None = None) -> int:
    """Delete all chunks matching a given metadata.source (optionally filtered by layer).

    Use before re-ingesting the same document to avoid duplicate chunks."""
    from db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            if layer:
                cur.execute(
                    "DELETE FROM lenin_corpus WHERE metadata->>'source' = %s AND metadata->>'layer' = %s",
                    (source, layer),
                )
            else:
                cur.execute(
                    "DELETE FROM lenin_corpus WHERE metadata->>'source' = %s",
                    (source,),
                )
            deleted = cur.rowcount
        conn.commit()
    return deleted


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
