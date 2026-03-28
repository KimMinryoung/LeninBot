"""telegram_bot.py — Telegram bot interface (aiogram 3.x).

Features:
- General messages → Claude Sonnet 4.6 with tool-use (vector_search, knowledge_graph_search, web_search, file system, execute_python)
- /chat <message> → CLAW pipeline (LangGraph agent: intent→retrieve→KG→strategize→generate)
- /task <content> → Save to PostgreSQL queue, background worker processes, push on completion
- /status → Show last 5 tasks
- /clear → Reset chat history

Tools are lazy-loaded from chatbot.py to share the BGE-M3 embedding model and other heavy resources.
Security: ALLOWED_USER_IDS whitelist, unauthorized users silently ignored.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from shared import KST, CORE_IDENTITY
from skills_loader import build_skills_prompt
from db import query as _query, execute as _execute, query_one as _query_one, get_conn as _get_conn
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, BufferedInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from aiogram.filters import Command
import anthropic
import base64

# Extracted modules
from telegram_tools import TOOLS, TOOL_HANDLERS
from claude_loop import sanitize_messages, estimate_tokens, chat_with_tools
from telegram_tasks import (
    process_task, broadcast, system_monitor,
    task_worker, schedule_worker, check_deploy_meta,
    recover_processing_tasks_on_startup,
    checkpoint_task_on_shutdown,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Suppress TelegramConflictError spam during deploy (old/new instance overlap)
class _ConflictFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "TelegramConflictError" not in record.getMessage()

logging.getLogger("aiogram.dispatcher").addFilter(_ConflictFilter())
logging.getLogger("aiogram.event").addFilter(_ConflictFilter())

# Throttle Neo4j DNS/connection retry spam (100s of warnings per second when AuraDB is down)
class _ThrottleFilter(logging.Filter):
    def __init__(self, interval: float = 60.0):
        super().__init__()
        self._last: dict[str, float] = {}
        self._interval = interval

    def filter(self, record: logging.LogRecord) -> bool:
        import time
        # Group by first 80 chars of message to dedup similar warnings
        key = record.getMessage()[:80]
        now = time.monotonic()
        last = self._last.get(key, 0.0)
        if now - last < self._interval:
            return False
        self._last[key] = now
        return True

logging.getLogger("neo4j").addFilter(_ThrottleFilter(60.0))

# ── Config ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}

def _ensure_table():
    """Create telegram_tasks and telegram_chat_history tables if not exists."""
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_tasks (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            content     TEXT NOT NULL,
            status      VARCHAR(20) DEFAULT 'pending',
            result      TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_chat_history (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            role        VARCHAR(10) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    # Index for fast user_id lookups
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_user_id
        ON telegram_chat_history (user_id, id DESC)
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_schedules (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            content     TEXT NOT NULL,
            cron_expr   VARCHAR(100) NOT NULL,
            enabled     BOOLEAN DEFAULT TRUE,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            last_run_at TIMESTAMPTZ
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_error_log (
            id          SERIAL PRIMARY KEY,
            level       VARCHAR(10) NOT NULL DEFAULT 'error',
            source      VARCHAR(100) NOT NULL,
            message     TEXT NOT NULL,
            detail      TEXT,
            task_id     INTEGER,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_error_log_created
        ON telegram_error_log (created_at DESC)
    """)
    # Task chaining columns (additive — safe for existing rows)
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS parent_task_id INTEGER REFERENCES telegram_tasks(id)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS scratchpad TEXT DEFAULT ''")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS depth INTEGER DEFAULT 0")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS mission_id INTEGER")
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_parent
        ON telegram_tasks(parent_task_id) WHERE parent_task_id IS NOT NULL
    """)
    # Mission context tables
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_missions (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            title       TEXT NOT NULL,
            status      VARCHAR(20) DEFAULT 'active',
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            closed_at   TIMESTAMPTZ
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_mission_events (
            id          SERIAL PRIMARY KEY,
            mission_id  INTEGER NOT NULL REFERENCES telegram_missions(id),
            source      TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_mission_events_timeline
        ON telegram_mission_events(mission_id, created_at)
    """)


# ── Error/Warning Logger ────────────────────────────────────────────
def _log_event(
    level: str,        # "error" | "warning"
    source: str,       # e.g. "chat", "task", "tool", "final_response"
    message: str,
    detail: str | None = None,
    task_id: int | None = None,
) -> None:
    """Persist an error or warning event to telegram_error_log."""
    try:
        _execute(
            "INSERT INTO telegram_error_log (level, source, message, detail, task_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (level[:10], source[:100], message[:2000], detail[:4000] if detail else None, task_id),
        )
    except Exception as _le:
        logger.warning("_log_event DB write failed: %s", _le)


# ── LLM clients ─────────────────────────────────────────────────────
_claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# OpenAI client (lazy — only created if key exists)
_openai_client = None
if OPENAI_API_KEY:
    from openai import AsyncOpenAI
    _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
_CLAUDE_MAX_TOKENS = 4096
_CLAUDE_MAX_TOKENS_TASK = 16384  # Tasks need longer output for full reports

# ── Runtime Config (mutable at runtime via /config) ──────────────────
_CONFIG_DEFAULTS = {
    "chat_budget": 0.30,       # USD per chat turn
    "task_budget": 1.00,       # USD per background task
    "chat_model": "high",      # "high" | "medium" | "low"
    "task_model": "high",      # "high" | "medium" | "low"
    "max_rounds_chat": 50,
    "max_rounds_task": 50,
    "provider": "claude",      # "claude" | "openai"
}

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config() -> dict:
    """Load config from disk, falling back to defaults for missing keys."""
    config = dict(_CONFIG_DEFAULTS)
    try:
        with open(_CONFIG_PATH, "r") as f:
            saved = json.loads(f.read())
        for key, val in saved.items():
            if key in _CONFIG_DEFAULTS:
                # Ensure type matches default
                default_val = _CONFIG_DEFAULTS[key]
                if isinstance(default_val, float):
                    config[key] = float(val)
                elif isinstance(default_val, int):
                    config[key] = int(val)
                else:
                    config[key] = str(val)
        logger.info("Config loaded from %s: %s", _CONFIG_PATH, config)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Config load failed, using defaults: %s", e)
    return config


def _save_config():
    """Persist current config to disk."""
    try:
        with open(_CONFIG_PATH, "w") as f:
            f.write(json.dumps(_config, indent=2))
    except Exception as e:
        logger.warning("Config save failed: %s", e)


_config = _load_config()

# Display metadata for config panel
_CONFIG_META = {
    "chat_budget":      {"label": "대화 예산",     "unit": "$",  "options": [0.10, 0.20, 0.30, 0.50, 1.00]},
    "task_budget":      {"label": "태스크 예산",   "unit": "$",  "options": [0.50, 1.00, 2.00, 3.00, 5.00]},
    "chat_model":       {"label": "대화 모델",     "unit": "",   "options": ["high", "medium", "low"]},
    "task_model":       {"label": "태스크 모델",   "unit": "",   "options": ["high", "medium", "low"]},
    "max_rounds_chat":  {"label": "대화 라운드",   "unit": "회", "options": [15, 30, 50, 80]},
    "max_rounds_task":  {"label": "태스크 라운드", "unit": "회", "options": [15, 30, 50, 80]},
    "provider":         {"label": "LLM 제공자",   "unit": "",   "options": ["claude", "openai"]},
}

_MODEL_ALIAS_MAP = {
    "haiku":  ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    "sonnet": ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    "opus":   ("claude-opus-4-6", "claude-opus-4-6"),
}

_OPENAI_MODEL_MAP = {
    "gpt54":     "gpt-5.4",
    "gpt54mini": "gpt-5.4-mini",
    "gpt54nano": "gpt-5.4-nano",
}

# Tier → provider-specific model mapping
_TIER_MAP = {
    "claude": {"high": "opus",   "medium": "sonnet", "low": "haiku"},
    "openai": {"high": "gpt54",  "medium": "gpt54mini", "low": "gpt54nano"},
}

def _tier_to_display(tier: str) -> str:
    """Return a display label showing tier + actual model for current provider."""
    provider = _config.get("provider", "claude")
    tier_map = _TIER_MAP.get(provider, _TIER_MAP["claude"])
    alias = tier_map.get(tier, tier)
    if provider == "openai":
        model_name = _OPENAI_MODEL_MAP.get(alias, alias)
    else:
        model_name = alias
    return f"{tier} ({model_name})"


async def _resolve_model(alias: str, fallback: str) -> str:
    """Resolve a Claude model alias to its actual ID via the Models API (non-blocking)."""
    try:
        resolved = await asyncio.to_thread(
            lambda: anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).models.retrieve(model_id=alias).id
        )
        logger.info("Resolved model %s => %s", alias, resolved)
        return resolved
    except Exception as e:
        logger.warning("Model resolve failed for %s, using fallback %s: %s", alias, fallback, e)
        return fallback


# Lazy model resolution cache — maps alias → resolved ID
_resolved_models: dict[str, str] = {}


async def _get_model_by_alias(alias: str) -> str:
    """Resolve a model short name (haiku/sonnet/opus) to its full ID, with caching."""
    if alias in _resolved_models:
        return _resolved_models[alias]
    model_alias, fallback = _MODEL_ALIAS_MAP.get(alias, ("claude-sonnet-4-6", "claude-sonnet-4-6"))
    resolved = await _resolve_model(model_alias, fallback)
    _resolved_models[alias] = resolved
    return resolved


def _resolve_openai_model(alias: str) -> str:
    """Resolve OpenAI model alias (gpt54/gpt54mini/gpt54nano) to actual model ID."""
    return _OPENAI_MODEL_MAP.get(alias, alias)


def _resolve_tier(tier: str) -> str:
    """Resolve a tier name (high/medium/low) to provider-specific model alias."""
    provider = _config.get("provider", "claude")
    tier_map = _TIER_MAP.get(provider, _TIER_MAP["claude"])
    return tier_map.get(tier, tier)  # passthrough if not a tier name


async def _get_model() -> str:
    """Get the current chat model based on runtime config."""
    alias = _resolve_tier(_config["chat_model"])
    if _config.get("provider") == "openai" or alias in _OPENAI_MODEL_MAP:
        return _resolve_openai_model(alias)
    return await _get_model_by_alias(alias)


async def _get_model_task() -> str:
    """Get the current task model based on runtime config."""
    alias = _resolve_tier(_config["task_model"])
    if _config.get("provider") == "openai" or alias in _OPENAI_MODEL_MAP:
        return _resolve_openai_model(alias)
    return await _get_model_by_alias(alias)


async def _get_model_light() -> str:
    """Get the light model (Haiku) — used for compression, reflection, etc."""
    return await _get_model_by_alias("haiku")


async def _get_model_moon() -> str:
    """Get the Moon (local LLM) model name — async wrapper for await compatibility."""
    from llm_client import MOON_MODEL
    return MOON_MODEL


def _extract_text(response) -> str:
    """Safely extract text from Claude API response, handling empty content."""
    if response.content:
        return response.content[0].text
    return ""


# ── Local LLM (OpenAI-compatible: llama-server, Ollama, etc.) ─────
_LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080")
_LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-9b")
_local_llm_available: bool | None = None  # None = not checked yet


async def _local_llm_generate(prompt: str, max_tokens: int = 2048) -> str | None:
    """Call local LLM via OpenAI-compatible /v1/chat/completions.

    Works with llama-server, TabbyAPI, Ollama (/v1 endpoint), vLLM, etc.
    Handles thinking models (Qwen3.5) — extracts content, ignores reasoning_content.
    Falls back gracefully — if server is down, callers should use Haiku.
    """
    global _local_llm_available
    import httpx

    if _local_llm_available is False:
        return None

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{_LOCAL_LLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": _LOCAL_LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            result = (msg.get("content") or "").strip()
            if not result:
                # Fallback: reasoning field (Ollama) or reasoning_content (llama-server)
                result = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            if _local_llm_available is None:
                _local_llm_available = True
                logger.info("Local LLM available: %s @ %s", _LOCAL_LLM_MODEL, _LOCAL_LLM_BASE_URL)
            return result or None
    except Exception as e:
        if _local_llm_available is not False:
            logger.info("Local LLM not available (%s), falling back to Haiku", e)
            _local_llm_available = False
        return None


def _current_datetime_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


# ── System Alerts (injected into system prompt) ─────────────────────
import time as _time

_MAX_ALERTS = 5
_ALERT_TTL = 24 * 60 * 60  # 24 hours

# Each alert: (monotonic_timestamp, formatted_string)
_system_alerts: list[tuple[float, str]] = []


def _prune_alerts():
    """Remove expired alerts and trim to max count."""
    now = _time.monotonic()
    _system_alerts[:] = [(t, m) for t, m in _system_alerts if now - t < _ALERT_TTL]
    while len(_system_alerts) > _MAX_ALERTS:
        _system_alerts.pop(0)


def _add_system_alert(msg: str):
    """Add a system alert visible to the bot in its system prompt."""
    _system_alerts.append((_time.monotonic(), f"[{datetime.now(KST).strftime('%H:%M')}] {msg}"))
    _prune_alerts()


def _clear_system_alert(keyword: str):
    """Remove alerts containing keyword (e.g. when issue resolves)."""
    _system_alerts[:] = [(t, m) for t, m in _system_alerts if keyword not in m]


def _format_system_alerts() -> str:
    _prune_alerts()
    if not _system_alerts:
        return ""
    items = "\n".join(f"- {m}" for _, m in _system_alerts)
    return f"\n<system-alerts>\n{items}\n</system-alerts>"


def _get_finance_context() -> str:
    """Get finance data summary for prompt injection. Never fails."""
    try:
        from finance_data import finance_summary
        summary = finance_summary()
        if summary:
            return f"\n<market-data>\n{summary}\n</market-data>"
    except Exception as e:
        logger.debug("Finance data unavailable: %s", e)
    return ""


_SYSTEM_PROMPT_TEMPLATE = CORE_IDENTITY + """
Operating via Telegram. Use tools proactively when data would improve the answer — don't rely on memory alone.

<tool-strategy>
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page, then analyze with context from other tools
- Self-reflection → read_diary; cross-interface memory → read_chat_logs
- Past lessons/mistakes → recall_experience (semantic search over accumulated daily insights)
- Store important facts → write_kg
- Your own source code → read_file (e.g. read_file("telegram_bot.py"), read_file("shared.py"))
- Server file management → list_directory, read_file, write_file
- Data processing / automation → execute_python
- Real-time market prices → get_finance_data
</tool-strategy>

<delegation>
You have specialized agents. Use the `delegate` tool to dispatch tasks:
- programmer: 코드 작성/수정/디버깅/파일 편집 전문 (예산 $1.50)
- general: 범용 리서치/분석 (예산 $1.00)

When to delegate vs handle directly:
- 간단한 질문, 일상 대화, 짧은 조회 → 직접 처리
- 멀티스텝 코딩, 파일 수정 → delegate(agent="programmer")
- 심층 리서치, 다중 소스 분석, 장문 보고서 → delegate(agent="general")
- 대화에서 도구를 10회 넘게 호출해야 할 것 같으면 즉시 delegate로 전환.
- 사용자에게 "계속할까요?"라고 묻지 말고, 스스로 판단해서 위임하라.

Context passing — 에이전트는 현재 대화를 직접 볼 수 없다. 반드시 `context` 필드에:
1. 사용자의 원래 요청 (원문 또는 핵심 요약)
2. 지금까지 대화에서 발견한 사항 (도구 결과, 분석, 결정)
3. 왜 이 에이전트에게 위임하는지 (이유와 기대 결과)
를 포함하라. context가 부실하면 에이전트가 맹목적으로 작업하게 된다.
</delegation>

<mission-management>
- 활성 미션이 있으면 시스템 프롬프트에 타임라인이 주입된다. 이를 활용해 맥락을 유지하라.
- 과제가 **완전히 완수**되었다고 판단하면 `mission(action="close")`를 호출해 미션을 종료하라.
- 아직 미완료이면 미션을 열어두어라. 다음 태스크나 대화에서 이어갈 수 있다.
</mission-management>

<response-rules>
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
</response-rules>

<context>
<current-time>{current_datetime}</current-time>
{system_alerts}
{skills_section}
</context>
"""


# Task system prompts are now defined per-agent in agents/ package.
# See agents/general.py, agents/programmer.py, etc.

# ── Chat History ─────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10  # 10 pairs = 20 messages
_HISTORY_TOKEN_LIMIT = 40_000  # compress if history exceeds this
_RECENT_TURNS_KEEP = 4  # keep last N turns uncompressed (4 turns = 8 msgs)


# Per-user clear marker: messages with id <= this value are ignored
_clear_after_id: dict[int, int] = {}



def _normalize_history_content(content) -> str:
    """Convert mixed/legacy message content into plain text.

    Claude history can contain structured content blocks in older rows.
    We strip tool blocks and keep only user-visible text so the next
    API call never includes dangling tool_use IDs.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        s = content.strip()
        # Legacy rows may store structured content as JSON string.
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = json.loads(s)
                return _normalize_history_content(parsed)
            except Exception:
                return content
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = _normalize_history_content(block)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        btype = content.get("type")
        if btype in ("tool_use", "server_tool_use"):
            return ""
        if btype == "text":
            return str(content.get("text", ""))
        if btype in ("tool_result", "web_search_tool_result"):
            return _normalize_history_content(content.get("content", ""))
        if "text" in content:
            return str(content.get("text", ""))
        if "content" in content:
            return _normalize_history_content(content.get("content"))
        return ""

    return str(content)


def _load_chat_history(user_id: int) -> list[dict]:
    """Load recent chat history from DB for a user (after last /clear)."""
    limit = MAX_HISTORY_TURNS * 2
    min_id = _clear_after_id.get(user_id, 0)
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, id FROM telegram_chat_history"
                "  WHERE user_id = %s AND id > %s ORDER BY id DESC LIMIT %s"
                ") sub ORDER BY id ASC",
                (user_id, min_id, limit),
            )
            rows = cur.fetchall()

    normalized: list[dict] = []
    for r in rows:
        role = r["role"] if r["role"] in ("user", "assistant") else "user"
        text = _normalize_history_content(r["content"])
        if not text:
            # Keep role alternation stable with a minimal placeholder.
            text = "(empty)"
        normalized.append({"role": role, "content": text})
    return normalized


def _save_chat_message(user_id: int, role: str, content: str):
    """Append a single message to DB chat history."""
    _execute(
        "INSERT INTO telegram_chat_history (user_id, role, content) VALUES (%s, %s, %s)",
        (user_id, role, content),
    )


def _clear_chat_history(user_id: int):
    """Mark current position as clear point — history before this is ignored.

    Persists the marker to DB so it survives bot restarts.
    """
    row = _query_one(
        "SELECT MAX(id) AS max_id FROM telegram_chat_history WHERE user_id = %s",
        (user_id,),
    )
    max_id = (row["max_id"] or 0) if row else 0
    _clear_after_id[user_id] = max_id
    # Persist to DB (upsert)
    _execute(
        "INSERT INTO chat_clear_markers (user_id, clear_after_id) VALUES (%s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET clear_after_id = EXCLUDED.clear_after_id",
        (user_id, max_id),
    )
    # Also delete stored chunk summaries
    _execute("DELETE FROM chat_history_summaries WHERE user_id = %s", (user_id,))


# ── Chunked History Summaries ─────────────────────────────────────────
_SUMMARY_CHUNK_SIZE = 10  # messages per summary chunk
_MAX_SUMMARY_CHUNKS = 3   # max chunks to include in context
_summary_table_ready = False


def _ensure_summary_table():
    global _summary_table_ready
    if _summary_table_ready:
        return
    _execute(
        "CREATE TABLE IF NOT EXISTS chat_clear_markers ("
        "  user_id BIGINT PRIMARY KEY,"
        "  clear_after_id BIGINT NOT NULL DEFAULT 0"
        ")"
    )
    # Load persisted clear markers into memory
    rows = _query("SELECT user_id, clear_after_id FROM chat_clear_markers")
    for r in rows:
        current = _clear_after_id.get(r["user_id"], 0)
        _clear_after_id[r["user_id"]] = max(current, r["clear_after_id"])
    _execute(
        "CREATE TABLE IF NOT EXISTS chat_history_summaries ("
        "  id SERIAL PRIMARY KEY,"
        "  user_id BIGINT NOT NULL,"
        "  chunk_start_id BIGINT NOT NULL,"
        "  chunk_end_id BIGINT NOT NULL,"
        "  summary TEXT NOT NULL,"
        "  msg_count INTEGER DEFAULT 0,"
        "  created_at TIMESTAMPTZ DEFAULT NOW()"
        ")"
    )
    _summary_table_ready = True


def _load_context_with_summaries(user_id: int) -> list[dict]:
    """Load chat context: chunk summaries + recent raw messages."""
    _ensure_summary_table()
    min_id = _clear_after_id.get(user_id, 0)

    # Last N chunk summaries (DESC then reverse for chronological order)
    summaries = _query(
        "SELECT id, chunk_start_id, chunk_end_id, summary FROM chat_history_summaries "
        "WHERE user_id = %s AND chunk_start_id > %s "
        "ORDER BY chunk_start_id DESC LIMIT %s",
        (user_id, min_id, _MAX_SUMMARY_CHUNKS),
    )
    summaries.reverse()

    # Validate: check that the newest summary still references existing chat rows.
    # If chat_history was purged but summaries remain, drop orphaned summaries.
    if summaries:
        check_id = summaries[-1]["chunk_end_id"]
        probe = _query_one(
            "SELECT id FROM telegram_chat_history WHERE id = %s AND user_id = %s",
            (check_id, user_id),
        )
        if not probe:
            orphan_ids = [s["id"] for s in summaries]
            logger.warning(
                "Orphaned summaries detected (chat row #%d missing) — purging %d summaries for user %d",
                check_id, len(orphan_ids), user_id,
            )
            _execute(
                "DELETE FROM chat_history_summaries WHERE user_id = %s",
                (user_id,),
            )
            summaries = []

    # Raw messages start after last chunk (or after clear marker)
    raw_after = summaries[-1]["chunk_end_id"] if summaries else min_id

    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, role, content FROM ("
                "  SELECT id, role, content FROM telegram_chat_history"
                "  WHERE user_id = %s AND id > %s ORDER BY id DESC LIMIT 20"
                ") sub ORDER BY id ASC",
                (user_id, raw_after),
            )
            raw_rows = cur.fetchall()

    # Build context: summaries as pairs + raw messages as text
    context: list[dict] = []
    for s in summaries:
        context.append({
            "role": "user",
            "content": f"[대화 요약 #{s['chunk_start_id']}~#{s['chunk_end_id']}]\n{s['summary']}",
        })
        context.append({"role": "assistant", "content": "이전 대화 내용을 파악했습니다."})

    for r in raw_rows:
        role = r["role"] if r["role"] in ("user", "assistant") else "user"
        text = _normalize_history_content(r["content"])
        context.append({"role": role, "content": text or "(empty)"})

    return context


async def _maybe_summarize_chunk(user_id: int):
    """Create a summary chunk if enough unsummarized messages have accumulated."""
    try:
        await asyncio.to_thread(_ensure_summary_table)
        min_id = _clear_after_id.get(user_id, 0)

        last = await asyncio.to_thread(
            _query_one,
            "SELECT chunk_end_id FROM chat_history_summaries "
            "WHERE user_id = %s AND chunk_start_id > %s "
            "ORDER BY chunk_end_id DESC LIMIT 1",
            (user_id, min_id),
        )
        raw_after = last["chunk_end_id"] if last else min_id

        # Bootstrap: no summaries yet → only consider recent messages, not from the dawn of time
        if not last:
            latest = await asyncio.to_thread(
                _query_one,
                "SELECT MAX(id) AS max_id FROM telegram_chat_history WHERE user_id = %s AND id > %s",
                (user_id, min_id),
            )
            if latest and latest["max_id"]:
                raw_after = max(min_id, latest["max_id"] - _SUMMARY_CHUNK_SIZE * (_MAX_SUMMARY_CHUNKS + 1))

        rows = await asyncio.to_thread(
            _query,
            "SELECT id, role, content FROM telegram_chat_history "
            "WHERE user_id = %s AND id > %s ORDER BY id ASC LIMIT %s",
            (user_id, raw_after, _SUMMARY_CHUNK_SIZE + 5),
        )

        if len(rows) < _SUMMARY_CHUNK_SIZE:
            return

        chunk = rows[:_SUMMARY_CHUNK_SIZE]
        chunk_start_id = chunk[0]["id"]
        chunk_end_id = chunk[-1]["id"]

        conversation_text = "\n".join(
            f"[{r['role']}] {_normalize_history_content(r['content'])[:500]}"
            for r in chunk
        )
        summary_prompt = (
            "아래 대화를 핵심 정보만 남기고 간결하게 요약해라. "
            "사용자가 어떤 주제를 물었고, 어떤 결론/답변이 나왔는지 위주로. "
            "고유명사, 수치, 날짜는 보존. 300자 이내.\n\n"
            + conversation_text
        )

        summary = await _local_llm_generate(summary_prompt)
        if not summary:
            resp = await _claude.messages.create(
                model=await _get_model_light(),
                max_tokens=512,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary = _extract_text(resp)

        await asyncio.to_thread(
            _execute,
            "INSERT INTO chat_history_summaries "
            "(user_id, chunk_start_id, chunk_end_id, summary, msg_count) "
            "VALUES (%s, %s, %s, %s, %s)",
            (user_id, chunk_start_id, chunk_end_id, summary, len(chunk)),
        )
        logger.info(
            "Chunk summary created: user=%d msgs=#%d~#%d (%d msgs)",
            user_id, chunk_start_id, chunk_end_id, len(chunk),
        )
    except Exception as e:
        logger.warning("Chunk summarization failed: %s", e)


async def _compress_history(messages: list[dict]) -> list[dict]:
    """Compress chat history if it exceeds the token limit.

    Summarizes older messages into a single context message using Haiku,
    keeping the most recent turns intact for conversational continuity.
    """
    total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    if total_tokens <= _HISTORY_TOKEN_LIMIT:
        return messages

    # Split into old (to summarize) and recent (to keep)
    keep_count = _RECENT_TURNS_KEEP * 2  # user+assistant pairs
    if len(messages) <= keep_count:
        return messages  # not enough messages to split

    old_msgs = messages[:-keep_count]
    recent_msgs = messages[-keep_count:]

    logger.info(
        "Compressing history: %d msgs (%d tokens) → summarize %d old, keep %d recent",
        len(messages), total_tokens, len(old_msgs), len(recent_msgs),
    )

    # Build summary request
    conversation_text = "\n".join(
        f"[{m['role']}] {m['content'][:1000]}" for m in old_msgs
    )
    summary_prompt = (
        "아래 대화를 핵심 정보만 남기고 간결하게 요약해라. "
        "사용자가 어떤 주제를 물었고, 어떤 결론/답변이 나왔는지 위주로. "
        "고유명사, 수치, 날짜는 보존. 300자 이내.\n\n"
        f"{conversation_text}"
    )

    try:
        # Try local LLM first (free), fall back to Haiku (paid)
        summary = await _local_llm_generate(summary_prompt)
        if not summary:
            resp = await _claude.messages.create(
                model=await _get_model_light(),
                max_tokens=512,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary = _extract_text(resp)
    except Exception as e:
        logger.warning("History compression failed: %s — using truncation fallback", e)
        # Fallback: just drop old messages
        return recent_msgs

    # Inject summary as a system-like context message
    compressed = [
        {"role": "user", "content": f"[이전 대화 요약]\n{summary}"},
        {"role": "assistant", "content": "네, 이전 대화 내용을 파악했습니다. 이어서 진행하겠습니다."},
    ] + recent_msgs

    new_tokens = sum(estimate_tokens(m["content"]) for m in compressed)
    logger.info("History compressed: %d tokens → %d tokens", total_tokens, new_tokens)
    return compressed

# ── CLAW pipeline (lazy-loaded) ──────────────────────────────────────
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from chatbot import graph
        _graph = graph
    return _graph


# ── Helpers ──────────────────────────────────────────────────────────
def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into chunks respecting Telegram's 4096 char limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos <= 0:
            split_pos = text.rfind(" ", 0, max_len)
        if split_pos <= 0:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


# ── Progress Callback (live tool progress via Telegram) ──────────────

_bot_instance: Bot | None = None  # set in bot_main()


def _make_progress_callback(chat_id: int):
    """Create an on_progress callback that sends tool execution progress via Telegram.

    Collects events per round, sends one message per round to avoid flood.
    """
    _buf: list[str] = []
    _current_round = [0]

    async def _flush():
        if not _buf or not _bot_instance:
            return
        text = "\n".join(_buf)
        _buf.clear()
        try:
            for chunk in _split_message(f"```\n{text}\n```"):
                await _bot_instance.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
        except Exception as e:
            logger.debug("Progress message send failed: %s", e)

    async def _on_progress(event: str, detail: str):
        # Extract round number from detail prefix "[N] ..."
        round_num = 0
        if detail.startswith("["):
            try:
                round_num = int(detail[1:detail.index("]")])
            except (ValueError, IndexError):
                pass

        # New round started → flush previous round's buffer
        if round_num > _current_round[0] and _current_round[0] > 0:
            await _flush()
        if round_num > 0:
            _current_round[0] = round_num

        if event == "thinking":
            _buf.append(f"💭 {detail}")
        elif event == "tool_call":
            _buf.append(detail)
        elif event == "tool_result":
            _buf.append(detail)
        elif event == "budget":
            _buf.append(f"💰 {detail}")

    # Expose flush for final cleanup
    _on_progress.flush = _flush
    return _on_progress


def _is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS


# ── Thin wrapper: _chat_with_tools (injects module-level dependencies) ──

async def _chat_with_tools(
    messages: list[dict],
    max_rounds: int | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    budget_usd: float | None = None,
    extra_tools: list | None = None,
    extra_handlers: dict | None = None,
    on_progress=None,
    budget_tracker: dict | None = None,
) -> str:
    """Call LLM with tools — dispatches to Claude or OpenAI based on provider config."""
    # Resolve runtime defaults strictly by None (not truthiness).
    resolved_max_rounds = _config["max_rounds_chat"] if max_rounds is None else max_rounds
    resolved_max_tokens = _CLAUDE_MAX_TOKENS if max_tokens is None else max_tokens
    resolved_budget = _config["chat_budget"] if budget_usd is None else budget_usd

    # Defensive coercion to keep downstream budget checks deterministic.
    try:
        resolved_budget = float(resolved_budget)
    except (TypeError, ValueError):
        logger.warning("Invalid budget_usd=%r; falling back to chat_budget=%s", budget_usd, _config["chat_budget"])
        resolved_budget = float(_config["chat_budget"])

    if resolved_budget <= 0:
        logger.warning("Non-positive budget_usd=%s; clamping to 0.01", resolved_budget)
        resolved_budget = 0.01

    sys_prompt = system_prompt or _SYSTEM_PROMPT_TEMPLATE.format(
        current_datetime=_current_datetime_str(),
        system_alerts=_format_system_alerts(),
        skills_section=build_skills_prompt(),
    )
    # Deduplicate tools by name — extra_tools override TOOLS with the same name
    seen_names: set[str] = set()
    merged_tools: list[dict] = []
    for t in (extra_tools or []):
        name = t.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            merged_tools.append(t)
    for t in TOOLS:
        name = t.get("name", "")
        if name not in seen_names:
            seen_names.add(name)
            merged_tools.append(t)
    merged_handlers = {**TOOL_HANDLERS, **(extra_handlers or {})}

    # ── Provider dispatch: Claude vs OpenAI ──
    if _config.get("provider") == "openai" and _openai_client:
        from openai_tool_loop import chat_with_tools as openai_chat
        return await openai_chat(
            messages,
            client=_openai_client,
            model=model or await _get_model(),
            tools=merged_tools,
            tool_handlers=merged_handlers,
            system_prompt=sys_prompt,
            max_rounds=resolved_max_rounds,
            max_tokens=resolved_max_tokens,
            log_event=_log_event,
            budget_usd=resolved_budget,
            on_progress=on_progress,
            budget_tracker=budget_tracker,
        )

    return await chat_with_tools(
        messages,
        client=_claude,
        model=model or await _get_model(),
        tools=merged_tools,
        tool_handlers=merged_handlers,
        system_prompt=sys_prompt,
        max_rounds=resolved_max_rounds,
        max_tokens=resolved_max_tokens,
        log_event=_log_event,
        budget_usd=resolved_budget,
        on_progress=on_progress,
        budget_tracker=budget_tracker,
    )


# ── Router & Handlers ───────────────────────────────────────────────
_pending_approvals: dict = {}  # 자가수정 승인 대기 (approval_id → entry)
router = Router()


_HELP_TEXT = """\
*레닌봇 커맨드 목록*

*대화*
/chat <메시지> — CLAW 파이프라인 질의 (RAG+KG+전략)
  일반 메시지 — Claude 직접 대화 (도구 사용 가능)
/clear — 대화 히스토리 초기화

*태스크*
/task <내용> — 백그라운드 태스크 등록 (Sonnet, $1 예산)
/status — 시스템 대시보드 (태스크·에러·KG)
/status\\_auto — 자율 생성 태스크 확인
/report <id> — 태스크 리포트 파일 재전송

*스케줄*
/schedule <cron> | <내용> — 정기 태스크 등록
  예: `/schedule 0 9 * * * | 오늘의 뉴스 브리핑`
/schedules — 등록된 스케줄 목록
/unschedule <id> — 스케줄 삭제

*시스템*
/kg — 지식그래프 현황 조회
/errors \\[n] \\[error|warning] — 에러/경고 로그
/config — 설정 패널 (모델, 예산, 라운드 수)
/fallback — 모델 토글 (sonnet ↔ haiku, API 과부하 시)
/provider — LLM 제공자 전환 (Claude ↔ OpenAI)
/restart \\[telegram|api|all] — 서비스 재시작만 (기본: telegram)
/deploy \\[telegram|api|all] — 서버 배포 (git pull + restart, 기본: all)
/modify <파일> | <이유> | <내용> — 서버 파일 수정

/help — 이 도움말 표시
"""


@router.message(Command("start"))
async def cmd_start(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    await message.answer(
        "레닌봇 텔레그램 인터페이스입니다.\n\n" + _HELP_TEXT,
        parse_mode="Markdown",
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    await message.answer(_HELP_TEXT, parse_mode="Markdown")


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    await asyncio.to_thread(_clear_chat_history, message.from_user.id)
    # Close active mission on history clear
    try:
        from telegram_mission import get_active_mission, close_mission
        mission = await asyncio.to_thread(get_active_mission, message.from_user.id)
        if mission:
            await asyncio.to_thread(close_mission, mission["id"])
    except Exception:
        pass
    await message.answer("대화 히스토리가 초기화되었습니다.")


@router.message(Command("mission"))
async def cmd_mission(message: Message):
    """View, create, or close the active mission.
    Usage:
      /mission               — 현재 미션 상태 조회
      /mission create <제목>  — 새 미션 생성
      /mission close         — 활성 미션 종료
    """
    if not _is_allowed(message.from_user.id):
        return
    uid = message.from_user.id
    raw_arg = (message.text or "").removeprefix("/mission").strip()
    arg_lower = raw_arg.lower()

    try:
        from telegram_mission import get_active_mission, get_mission_events, close_mission, create_mission
        mission = await asyncio.to_thread(get_active_mission, uid)

        # --- CREATE ---
        if arg_lower.startswith("create"):
            title = raw_arg[len("create"):].strip()
            if not title:
                await message.answer("❌ 미션 제목을 입력하세요.\n예: `/mission create 3월 금값 분석`")
                return
            if mission:
                await message.answer(
                    f"⚠️ 이미 활성 미션이 있습니다: *#{mission['id']}* {mission['title']}\n"
                    f"먼저 `/mission close`로 종료하세요."
                )
                return
            new_mission = await asyncio.to_thread(create_mission, uid, title)
            await message.answer(
                f"✅ 미션 생성됨\n"
                f"🎯 *#{new_mission['id']}*: {new_mission['title']}"
            )
            return

        # --- CLOSE ---
        if arg_lower == "close":
            if not mission:
                await message.answer("활성 미션이 없습니다.")
                return
            await asyncio.to_thread(close_mission, mission["id"])
            await message.answer(f"✅ 미션 #{mission['id']} 종료: {mission['title']}")
            return

        # --- STATUS (default) ---
        if not mission:
            await message.answer(
                "활성 미션이 없습니다.\n"
                "새 미션을 만들려면: `/mission create <제목>`"
            )
            return
        events = await asyncio.to_thread(get_mission_events, mission["id"], 10)
        lines = [f"🎯 *미션 #{mission['id']}*: {mission['title']}", f"생성: {mission['created_at']}"]
        if events:
            lines.append(f"\n타임라인 ({len(events)}건):")
            for e in events:
                lines.append(f"  `[{e['source']}]` {e['event_type']}: {str(e['content'] or '')[:100]}")
        await message.answer("\n".join(lines))
    except Exception as e:
        await message.answer(f"미션 오류: {e}")


@router.message(Command("errors"))
async def cmd_errors(message: Message):
    """Show recent error/warning log entries."""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/errors").strip()
    # Parse optional limit and level filter
    # Usage: /errors [n] [error|warning|all]
    limit = 20
    level_filter = None
    for token in arg.split():
        if token.isdigit():
            limit = min(int(token), 50)
        elif token.lower() in ("error", "warning", "warn"):
            level_filter = "error" if token.lower() == "error" else "warning"
    try:
        if level_filter:
            rows = await asyncio.to_thread(
                _query,
                "SELECT id, level, source, message, detail, task_id, created_at "
                "FROM telegram_error_log WHERE level = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (level_filter, limit),
            )
        else:
            rows = await asyncio.to_thread(
                _query,
                "SELECT id, level, source, message, detail, task_id, created_at "
                "FROM telegram_error_log ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
    except Exception as e:
        await message.answer(f"에러 로그 조회 실패: {e}")
        return
    if not rows:
        await message.answer("✅ 기록된 에러/경고 없음.")
        return
    level_icons = {"error": "🔴", "warning": "🟡"}
    lines = [f"🗒️ *에러/경고 로그* (최근 {len(rows)}건)\n"]
    for r in rows:
        icon = level_icons.get(r["level"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M:%S")
        task_info = f" [태스크#{r['task_id']}]" if r["task_id"] else ""
        lines.append(
            f"{icon} `{ts}` [{r['source']}]{task_info}\n"
            f"   {r['message'][:120]}"
        )
    for chunk in _split_message("\n\n".join(lines)):
        await message.answer(chunk, parse_mode="Markdown")


@router.message(Command("chat"))
async def cmd_chat(message: Message):
    """Route message through the CLAW pipeline (LangGraph agent)."""
    if not _is_allowed(message.from_user.id):
        return
    content = (message.text or "").removeprefix("/chat").strip()
    if not content:
        await message.answer("사용법: /chat <메시지>")
        return

    user_id = message.from_user.id
    await message.answer("CLAW 파이프라인 처리 중...")

    try:
        from langchain_core.messages import HumanMessage

        g = _get_graph()
        thread_id = f"tg_{user_id}"
        inputs = {"messages": [HumanMessage(content=content)]}
        config = {"configurable": {"thread_id": thread_id}}

        answer = None
        logs: list[str] = []
        async for output in g.astream(inputs, config=config, stream_mode="updates"):
            for node_name, node_content in output.items():
                if node_name == "log_conversation":
                    continue
                if "logs" in node_content:
                    logs.extend(node_content["logs"])
                if node_name == "generate":
                    last_msg = node_content["messages"][-1]
                    answer = last_msg.content

        if answer:
            for chunk in _split_message(answer):
                await message.answer(chunk)
        else:
            await message.answer("파이프라인에서 답변을 생성하지 못했습니다.")

        if logs:
            log_summary = "\n".join(logs[-10:])  # last 10 log lines
            for chunk in _split_message(f"[처리 로그]\n{log_summary}"):
                await message.answer(chunk)

    except Exception as e:
        logger.error("CLAW pipeline error: %s", e)
        await message.answer(f"CLAW 파이프라인 오류: {e}")


@router.message(Command("task"))
async def cmd_task(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    content = (message.text or "").removeprefix("/task").strip()
    if not content:
        await message.answer("사용법: /task <내용>")
        return
    try:
        uid = message.from_user.id

        # Resolve mission: use active or auto-create
        mission_id = None
        try:
            from telegram_mission import get_active_mission, create_mission
            mission = await asyncio.to_thread(get_active_mission, uid)
            if mission:
                mission_id = mission["id"]
        except Exception as e:
            logger.warning("Mission lookup failed: %s", e)

        rows = await asyncio.to_thread(
            _query,
            "INSERT INTO telegram_tasks (user_id, content, mission_id) VALUES (%s, %s, %s) RETURNING id",
            (uid, content, mission_id),
        )
        task_id = rows[0]["id"] if rows else None
        msg = f"태스크가 큐에 추가되었습니다:\n{content}"

        # Auto-create mission if none existed
        if task_id and mission_id is None:
            try:
                from telegram_mission import create_mission
                mission = await asyncio.to_thread(create_mission, uid, content[:80], task_id)
                mission_id = mission["id"]
                msg += f"\n\n🎯 미션 #{mission_id} 자동 생성"
            except Exception as e:
                logger.warning("Mission auto-create failed: %s", e)

        await message.answer(msg)
    except Exception as e:
        logger.error("Task insert error: %s", e)
        await message.answer(f"태스크 등록 실패: {e}")



@router.message(Command("stats"))
async def cmd_stats(message: Message):
    """시스템 리소스 현황 — CPU/메모리/디스크/네트워크 (psutil 실시간 + JSON 추이)."""
    if not _is_allowed(message.from_user.id):
        return

    import psutil
    from datetime import timezone, timedelta
    from scripts.metrics_snapshot import parse_cpu_json, parse_memory_json, _sparkline

    KST_tz = timezone(timedelta(hours=9))
    now_kst = datetime.now(timezone.utc).astimezone(KST_tz)
    ts_str = now_kst.strftime("%Y-%m-%d %H:%M KST")

    await message.answer("⏳ 메트릭 수집 중...")

    def _collect():
        cpu = psutil.cpu_percent(interval=1)
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        import time
        net_before = psutil.net_io_counters()
        time.sleep(0.5)
        net_after = psutil.net_io_counters()
        rx_kbs = round((net_after.bytes_recv - net_before.bytes_recv) / 1024 / 0.5, 1)
        tx_kbs = round((net_after.bytes_sent - net_before.bytes_sent) / 1024 / 0.5, 1)
        return {
            "cpu": cpu,
            "mem_pct": vm.percent,
            "mem_used": round(vm.used / 1024**3, 2),
            "mem_total": round(vm.total / 1024**3, 2),
            "mem_avail": round(vm.available / 1024**3, 2),
            "disk_pct": disk.percent,
            "disk_used": round(disk.used / 1024**3, 1),
            "disk_total": round(disk.total / 1024**3, 1),
            "disk_free": round(disk.free / 1024**3, 1),
            "rx_kbs": rx_kbs,
            "tx_kbs": tx_kbs,
        }

    def _make_bar(pct: float, width: int = 20) -> str:
        filled = int(round(pct / 100 * width))
        filled = max(0, min(filled, width))
        return "█" * filled + "░" * (width - filled)

    def _alert(pct: float) -> str:
        if pct >= 90:
            return " 🔴"
        if pct >= 75:
            return " 🟡"
        return " 🟢"

    try:
        m = await asyncio.to_thread(_collect)

        def _get_sparks():
            try:
                cpu_rows = parse_cpu_json(12)
                mem_rows = parse_memory_json(12)
                cpu_spark = _sparkline([v for _, v in cpu_rows]) if cpu_rows else "—"
                mem_spark = _sparkline([v for _, v in mem_rows]) if mem_rows else "—"
                return cpu_spark, mem_spark
            except Exception:
                return "—", "—"

        cpu_spark, mem_spark = await asyncio.to_thread(_get_sparks)

        out = [
            f"📊 *시스템 메트릭* — {ts_str}",
            "",
            f"🖥 *CPU*{_alert(m['cpu'])}",
            f"  사용률: `{m['cpu']:.1f}%`",
            f"  `[{_make_bar(m['cpu'])}]`",
            f"  추이(12h): `{cpu_spark}`",
            "",
            f"🧠 *메모리*{_alert(m['mem_pct'])}",
            f"  사용: `{m['mem_used']} / {m['mem_total']} GiB ({m['mem_pct']:.1f}%)`",
            f"  여유: `{m['mem_avail']} GiB`",
            f"  `[{_make_bar(m['mem_pct'])}]`",
            f"  추이(12h): `{mem_spark}`",
            "",
            f"💾 *디스크 (/)*{_alert(m['disk_pct'])}",
            f"  사용: `{m['disk_used']} / {m['disk_total']} GiB ({m['disk_pct']:.1f}%)`",
            f"  여유: `{m['disk_free']} GiB`",
            f"  `[{_make_bar(m['disk_pct'])}]`",
            "",
            f"🌐 *네트워크*",
            f"  ↓ `{m['rx_kbs']} kB/s`  ↑ `{m['tx_kbs']} kB/s`",
        ]
        await message.answer("\n".join(out), parse_mode="Markdown")
    except Exception as e:
        logger.error("cmd_stats error: %s", e)
        await message.answer(f"⚠️ 메트릭 수집 실패: {e}")

@router.message(Command("status"))
async def cmd_status(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    uid = message.from_user.id

    # Gather all dashboard data in parallel
    tasks_f = asyncio.to_thread(
        _query,
        "SELECT id, content, status, created_at FROM telegram_tasks "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",
        (uid,),
    )
    errors_f = asyncio.to_thread(
        _query,
        "SELECT level, count(*) AS cnt FROM telegram_error_log "
        "WHERE created_at > NOW() - INTERVAL '24 hours' "
        "GROUP BY level ORDER BY level",
        None,
    )
    task_stats_f = asyncio.to_thread(
        _query,
        "SELECT status, count(*) AS cnt FROM telegram_tasks "
        "GROUP BY status",
        None,
    )

    try:
        tasks, errors, task_stats = await asyncio.gather(tasks_f, errors_f, task_stats_f)
    except Exception as e:
        logger.error("Status dashboard query error: %s", e)
        await message.answer(f"대시보드 조회 실패: {e}")
        return

    # -- Build dashboard --
    lines = ["*시스템 대시보드*\n"]

    # 1. Task summary
    stat_map = {r["status"]: r["cnt"] for r in task_stats}
    total_tasks = sum(stat_map.values())
    lines.append(
        f"*태스크* ({total_tasks}건): "
        f"✅{stat_map.get('done', 0)} "
        f"⏳{stat_map.get('pending', 0)} "
        f"🔄{stat_map.get('processing', 0)} "
        f"❌{stat_map.get('failed', 0)} "
        f"🔀{stat_map.get('handed_off', 0)}"
    )

    # 2. Error counts (24h)
    err_map = {r["level"]: r["cnt"] for r in errors}
    err_total = sum(err_map.values())
    if err_total:
        lines.append(
            f"*에러 (24h)*: 🔴error {err_map.get('error', 0)} "
            f"🟡warning {err_map.get('warning', 0)}"
        )
    else:
        lines.append("*에러 (24h)*: 없음")

    # 3. KG stats (quick, non-blocking)
    try:
        from shared import fetch_kg_stats
        kg = await asyncio.to_thread(fetch_kg_stats)
        if "error" not in kg:
            entity_total = sum(v for v in kg.get("entity_types", {}).values())
            lines.append(
                f"*KG*: 엔티티 {entity_total} | "
                f"관계 {kg.get('edge_count', 0)} | "
                f"에피소드 {kg.get('episode_count', 0)}"
            )
        else:
            lines.append(f"*KG*: ⚠️ {kg['error'][:60]}")
    except Exception as e:
        lines.append(f"*KG*: ⚠️ 조회 실패")

    # 4. Recent tasks
    if tasks:
        lines.append("\n*최근 태스크:*")
        status_icons = {"pending": "⏳", "processing": "🔄", "done": "✅", "failed": "❌", "handed_off": "🔀"}
        for r in tasks:
            icon = status_icons.get(r["status"], "❓")
            ts = r["created_at"].strftime("%m/%d %H:%M")
            preview = r["content"][:45]
            lines.append(f"{icon} `[{r['id']}]` {preview}\n   {r['status']} | {ts}")
    else:
        lines.append("\n태스크 없음")

    await message.answer("\n".join(lines), parse_mode="Markdown")


@router.message(Command("kg"))
async def cmd_kg(message: Message):
    """Directly show KG stats — no LLM involved."""
    if not _is_allowed(message.from_user.id):
        return
    from shared import fetch_kg_stats
    await message.answer("KG 조회 중...")
    try:
        stats = await asyncio.to_thread(fetch_kg_stats)
    except Exception as e:
        await message.answer(f"KG 조회 실패: {e}")
        return
    if "error" in stats:
        await message.answer(f"⚠️ KG 오류: {stats['error']}")
        return

    lines = ["📊 *지식그래프 현황* (Neo4j Local)\n"]
    lines.append(f"엔티티: {sum(v for v in stats.get('entity_types', {}).values())}개")
    for label, cnt in stats.get("entity_types", {}).items():
        lines.append(f"  {label}: {cnt}")
    lines.append(f"관계(엣지): {stats.get('edge_count', 0)}개")
    lines.append(f"에피소드: {stats.get('episode_count', 0)}건")
    episodes = stats.get("recent_episodes", [])
    if episodes:
        lines.append("\n*최근 에피소드:*")
        for ep in episodes:
            lines.append(f"  • {ep.get('name', '?')} [{ep.get('group_id', '')}]")
    await message.answer("\n".join(lines))


@router.message(Command("report"))
async def cmd_report(message: Message):
    """Directly fetch a task report from DB and send as file — no LLM involved."""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/report").strip()
    if not arg:
        await message.answer("사용법: /report <task_id>")
        return
    try:
        task_id = int(arg)
    except ValueError:
        await message.answer("task_id는 숫자여야 합니다.")
        return
    try:
        row = await asyncio.to_thread(
            _query_one,
            "SELECT id, content, status, result FROM telegram_tasks WHERE id = %s",
            (task_id,),
        )
    except Exception as e:
        await message.answer(f"조회 실패: {e}")
        return
    if not row:
        await message.answer(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")
        return
    if row["status"] != "done" or not row.get("result"):
        await message.answer(f"태스크 #{task_id} 상태: {row['status']} — 완료된 리포트가 없습니다.")
        return
    report = row["result"]
    doc = BufferedInputFile(report.encode("utf-8"), filename=f"report_task_{task_id}.md")
    await message.answer_document(doc, caption=f"태스크 #{task_id} 리포트 (DB 원문, {len(report)}자)")


@router.message(Command("status_auto"))
async def cmd_status_auto(message: Message):
    """Show recent self-generated (autonomous) tasks."""
    if not _is_allowed(message.from_user.id):
        return
    try:
        rows = await asyncio.to_thread(
            _query,
            "SELECT id, content, status, created_at FROM telegram_tasks "
            "WHERE user_id = 0 ORDER BY created_at DESC LIMIT 10",
        )
    except Exception as e:
        logger.error("Auto-task status query error: %s", e)
        await message.answer(f"조회 실패: {e}")
        return
    if not rows:
        await message.answer("자율 생성된 태스크가 없습니다.")
        return
    status_icons = {"pending": "⏳", "processing": "🔄", "done": "✅", "failed": "❌", "handed_off": "🔀"}
    lines = ["🤖 *자율 생성 태스크* (최근 10건)\n"]
    for r in rows:
        icon = status_icons.get(r["status"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M")
        preview = r["content"][:60]
        lines.append(f"{icon} [{r['id']}] {preview}\n   상태: {r['status']} | {ts}")
    await message.answer("\n\n".join(lines))


@router.message(Command("schedule"))
async def cmd_schedule(message: Message):
    """Add a cron schedule: /schedule <cron_expr> | <task content>"""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/schedule").strip()
    if not arg or "|" not in arg:
        await message.answer(
            "사용법: /schedule <cron식> | <태스크 내용>\n\n"
            "예시:\n"
            "  /schedule 0 9 * * * | 오늘의 국제 뉴스 브리핑\n"
            "  /schedule 0 8 * * 1 | 주간 지정학 정세 분석\n"
            "  /schedule 0 */6 * * * | 6시간마다 KG 상태 점검\n\n"
            "cron 형식: 분 시 일 월 요일 (KST 기준)"
        )
        return
    parts = arg.split("|", 1)
    cron_expr = parts[0].strip()
    content = parts[1].strip()
    if not content:
        await message.answer("태스크 내용이 비어있습니다.")
        return
    # Validate cron expression
    try:
        from croniter import croniter
        croniter(cron_expr)
    except (ValueError, KeyError) as e:
        await message.answer(f"잘못된 cron 표현식: {cron_expr}\n오류: {e}")
        return
    try:
        # Set last_run_at = NOW() so the first fire waits for the next cron window
        await asyncio.to_thread(
            _execute,
            "INSERT INTO telegram_schedules (user_id, content, cron_expr, last_run_at) "
            "VALUES (%s, %s, %s, NOW())",
            (message.from_user.id, content, cron_expr),
        )
        await message.answer(
            f"✅ 스케줄 등록 완료\n"
            f"  cron: `{cron_expr}` (KST)\n"
            f"  내용: {content[:100]}"
        )
    except Exception as e:
        await message.answer(f"스케줄 등록 실패: {e}")


@router.message(Command("schedules"))
async def cmd_schedules(message: Message):
    """List all schedules for the user."""
    if not _is_allowed(message.from_user.id):
        return
    try:
        rows = await asyncio.to_thread(
            _query,
            "SELECT id, content, cron_expr, enabled, last_run_at "
            "FROM telegram_schedules WHERE user_id = %s ORDER BY id",
            (message.from_user.id,),
        )
    except Exception as e:
        await message.answer(f"조회 실패: {e}")
        return
    if not rows:
        await message.answer("등록된 스케줄이 없습니다.")
        return
    lines = ["📅 *등록된 스케줄*\n"]
    for r in rows:
        status = "✅" if r["enabled"] else "⏸️"
        last = r["last_run_at"].strftime("%m/%d %H:%M") if r["last_run_at"] else "미실행"
        preview = r["content"][:60]
        lines.append(
            f"{status} [{r['id']}] `{r['cron_expr']}`\n"
            f"   {preview}\n"
            f"   마지막 실행: {last}"
        )
    await message.answer("\n\n".join(lines))


@router.message(Command("unschedule"))
async def cmd_unschedule(message: Message):
    """Delete a schedule: /unschedule <id>"""
    if not _is_allowed(message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/unschedule").strip()
    if not arg:
        await message.answer("사용법: /unschedule <schedule_id>")
        return
    try:
        sched_id = int(arg)
    except ValueError:
        await message.answer("schedule_id는 숫자여야 합니다.")
        return
    try:
        row = await asyncio.to_thread(
            _query_one,
            "DELETE FROM telegram_schedules WHERE id = %s AND user_id = %s RETURNING id",
            (sched_id, message.from_user.id),
        )
    except Exception as e:
        await message.answer(f"삭제 실패: {e}")
        return
    if row:
        await message.answer(f"🗑️ 스케줄 [{sched_id}] 삭제 완료")
    else:
        await message.answer(f"스케줄 [{sched_id}]을(를) 찾을 수 없습니다.")


@router.message(Command("restart"))
async def cmd_restart(message: Message):
    """Restart service(s) without git pull. Pure systemctl restart."""
    if not _is_allowed(message.from_user.id):
        return

    args = (message.text or "").split(maxsplit=1)
    target = args[1].strip().lower() if len(args) > 1 else "telegram"
    if target not in ("telegram", "api", "all"):
        await message.answer(f"❌ 알 수 없는 대상: `{target}`\n사용법: `/restart [telegram|api|all]`", parse_mode="Markdown")
        return

    services = {
        "telegram": ["leninbot-telegram"],
        "api": ["leninbot-api"],
        "all": ["leninbot-api", "leninbot-telegram"],  # API first, telegram last
    }[target]

    status_msg = await message.answer(f"🔄 서비스 재시작 중... ({target})")
    results = []
    for svc in services:
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "-n", "systemctl", "restart", svc,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                results.append(f"✅ {svc}")
            else:
                results.append(f"❌ {svc}: {stdout.decode(errors='replace').strip()}")
        except asyncio.TimeoutError:
            results.append(f"⏱ {svc}: timeout")
        except (asyncio.CancelledError, ConnectionError, OSError):
            return  # telegram being restarted — expected

    try:
        await status_msg.edit_text(f"서비스 재시작 완료:\n" + "\n".join(results))
    except Exception:
        pass  # bot was restarted


@router.message(Command("deploy"))
async def cmd_deploy(message: Message):
    """Run deploy.sh — git pull + restart services. Output sent back via Telegram."""
    if not _is_allowed(message.from_user.id):
        return
    deploy_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy.sh")
    if not os.path.isfile(deploy_script):
        await message.answer("deploy.sh를 찾을 수 없습니다.")
        return

    # Parse service target: /deploy [telegram|api|all] (default: all)
    args = (message.text or "").split(maxsplit=1)
    target = args[1].strip().lower() if len(args) > 1 else "all"
    if target not in ("telegram", "api", "all"):
        await message.answer(f"❌ 알 수 없는 대상: `{target}`\n사용법: `/deploy [telegram|api|all]`", parse_mode="Markdown")
        return

    status_msg = await message.answer(f"🚀 Deploy 시작... (대상: {target})")
    try:
        # Run deploy.sh detached (setsid) so it survives bot restart
        log_path = "/tmp/leninbot-deploy.log"
        proc = await asyncio.create_subprocess_exec(
            "setsid", "bash", deploy_script, f"--{target}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        # Read output until process exits or bot gets killed by restart
        output_lines: list[str] = []
        try:
            async for line in proc.stdout:
                output_lines.append(line.decode(errors="replace").rstrip())
            await proc.wait()
        except (asyncio.CancelledError, ConnectionError, OSError):
            return  # bot is being restarted by deploy.sh — expected, curl handles notification

        result = "\n".join(output_lines[-30:])  # last 30 lines
        if proc.returncode == 0:
            _add_system_alert(f"Deploy 완료 (대상: {target})")
            await status_msg.edit_text(f"✅ Deploy 완료\n```\n{result}\n```", parse_mode="Markdown")
        else:
            _add_system_alert(f"Deploy 실패 (대상: {target}, exit {proc.returncode})")
            await status_msg.edit_text(f"❌ Deploy 실패 (exit {proc.returncode})\n```\n{result}\n```", parse_mode="Markdown")
    except Exception as e:
        # ServerDisconnectedError / CancelledError = bot killed by deploy restart — expected
        err_name = type(e).__name__
        err_str = str(e)
        if ("Disconnect" in err_name or "Disconnect" in err_str
                or isinstance(e, (asyncio.CancelledError, ConnectionError, OSError))):
            return  # deploy.sh curl handles notification
        _add_system_alert(f"Deploy 오류 (대상: {target}): {type(e).__name__}")
        try:
            await status_msg.edit_text(f"❌ Deploy 오류: {e}")
        except Exception:
            pass


@router.message(Command("fallback"))
async def cmd_fallback(message: Message):
    """Toggle chat+task model between high and low tier (for API overload)."""
    if not _is_allowed(message.from_user.id):
        return
    if _config["chat_model"] in ("high", "medium"):
        _config["chat_model"] = "low"
        _config["task_model"] = "low"
        _resolved_models.clear()
        _add_system_alert(f"⚠️ Fallback 모드 활성화: → {_tier_to_display('low')}")
        await message.answer(f"⚡ Fallback ON — 모든 모델을 {_tier_to_display('low')}로 전환")
    else:
        _config["chat_model"] = "high"
        _config["task_model"] = "high"
        _resolved_models.clear()
        _add_system_alert(f"Fallback 해제: → {_tier_to_display('high')}")
        await message.answer(f"✅ Fallback OFF — 모든 모델을 {_tier_to_display('high')}로 복귀")
    _save_config()


@router.message(Command("provider"))
async def cmd_provider(message: Message):
    """Toggle LLM provider between Claude and OpenAI."""
    if not _is_allowed(message.from_user.id):
        return
    if not _openai_client:
        await message.answer("❌ OPENAI_API_KEY가 설정되지 않아 전환할 수 없습니다.")
        return
    current = _config.get("provider", "claude")
    if current == "claude":
        _config["provider"] = "openai"
        _add_system_alert("🔄 Provider 전환: Claude → OpenAI")
        await message.answer(f"🔄 OpenAI로 전환\n대화: {_tier_to_display(_config['chat_model'])}\n태스크: {_tier_to_display(_config['task_model'])}")
    else:
        _config["provider"] = "claude"
        _resolved_models.clear()
        _add_system_alert("🔄 Provider 전환: OpenAI → Claude")
        await message.answer(f"🔄 Claude로 전환\n대화: {_tier_to_display(_config['chat_model'])}\n태스크: {_tier_to_display(_config['task_model'])}")
    _save_config()


@router.message(F.photo)
async def handle_photo(message: Message):
    """사용자가 이미지를 보내면 Claude Vision으로 분석"""
    if not _is_allowed(message.from_user.id):
        return
    await message.chat.do("typing")

    user_id = message.from_user.id
    import time as _time

    # 가장 큰 해상도 이미지 선택
    photo = message.photo[-1]
    logger.info("photo received: user_id=%s file_id=%s size=%dx%d",
                user_id, photo.file_id, photo.width, photo.height)
    file = await message.bot.get_file(photo.file_id)

    # 이미지 다운로드 (bytes)
    file_bytes = await message.bot.download_file(file.file_path)
    image_data = base64.b64encode(file_bytes.read()).decode("utf-8")

    # Detect media type from file extension (Telegram supports JPEG, PNG, WebP)
    _ext = (file.file_path or "").rsplit(".", 1)[-1].lower() if file.file_path else ""
    _media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    media_type = _media_type_map.get(_ext, "image/jpeg")

    # caption이 있으면 프롬프트로 사용
    caption = message.caption or "이 이미지를 분석해줘."

    # 채팅 히스토리 저장 — user 메시지
    user_history_text = f"[이미지] {caption}" if message.caption else "[이미지]"
    await asyncio.to_thread(_save_chat_message, user_id, "user", user_history_text)

    # 직전 1턴(user+assistant)을 맥락으로 포함
    recent = await asyncio.to_thread(_load_chat_history, user_id)
    # recent 마지막은 방금 저장한 [이미지] → 그 앞 2개가 직전 턴
    context_msgs = recent[-3:-1] if len(recent) >= 3 else []
    # Claude API는 user로 시작해야 함 — assistant로 시작하면 컨텍스트 제거
    if context_msgs and context_msgs[0]["role"] != "user":
        context_msgs = []

    # Vision API 호출 — provider에 따라 분기
    t_start = _time.monotonic()
    try:
        model_id = await _get_model()
        if _config.get("provider") == "openai" and _openai_client:
            # OpenAI Vision: image_url with base64 data URI
            messages = context_msgs + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": caption,
                        },
                    ],
                }
            ]
            response = await _openai_client.chat.completions.create(
                model=model_id,
                max_completion_tokens=1024,
                messages=messages,
            )
            reply_text = response.choices[0].message.content or ""
            usage = response.usage
            in_tok = getattr(usage, "prompt_tokens", "?") if usage else "?"
            out_tok = getattr(usage, "completion_tokens", "?") if usage else "?"
        else:
            # Claude Vision: base64 image source
            messages = context_msgs + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": caption,
                        },
                    ],
                }
            ]
            response = await _claude.messages.create(
                model=model_id,
                max_tokens=1024,
                messages=messages,
            )
            reply_text = _extract_text(response)
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "input_tokens", "?") if usage else "?"
            out_tok = getattr(usage, "output_tokens", "?") if usage else "?"
        elapsed = _time.monotonic() - t_start
        logger.info("photo vision done: user_id=%s elapsed=%.2fs in_tokens=%s out_tokens=%s",
                    user_id, elapsed, in_tok, out_tok)

        # 채팅 히스토리 저장 — assistant 응답
        await asyncio.to_thread(_save_chat_message, user_id, "assistant", reply_text)

        await message.reply(reply_text)
    except Exception as e:
        logger.error("handle_photo error: %s", e)
        await message.reply(f"❌ 이미지 분석 중 오류: {e}")


@router.message(F.text, ~Command("config"), ~Command("modify"))
async def handle_message(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    user_id = message.from_user.id
    user_text = message.text

    # Save user message to DB, load context (chunk summaries + raw messages)
    await asyncio.to_thread(_save_chat_message, user_id, "user", user_text)
    history = await asyncio.to_thread(_load_context_with_summaries, user_id)
    history = sanitize_messages(history)

    # Auto-recall: fetch relevant past experiences for context injection
    experience_context = await _fetch_relevant_experiences(user_text)

    # Mission context: inject active mission timeline
    from telegram_mission import build_mission_context
    mission_context = await asyncio.to_thread(build_mission_context, user_id)

    try:
        system_override = None
        extra_context = (experience_context or "") + mission_context
        if extra_context:
            system_override = _SYSTEM_PROMPT_TEMPLATE.format(
                current_datetime=_current_datetime_str(),
                system_alerts=_format_system_alerts(),
                skills_section=build_skills_prompt(),
            ) + extra_context
        # Bind mission tool handler to this user
        from telegram_tools import build_mission_handler
        mission_handler = build_mission_handler(user_id)
        progress_cb = _make_progress_callback(message.chat.id)
        bt = {}
        reply = await _chat_with_tools(
            history, system_prompt=system_override, on_progress=progress_cb, budget_tracker=bt,
            extra_handlers={"mission": mission_handler},
        )
        if hasattr(progress_cb, "flush"):
            await progress_cb.flush()
    except Exception as e:
        bt = {}  # no budget info on exception
        err_str = str(e)
        is_tool_pair_error = "tool_use" in err_str and "tool_result" in err_str

        if is_tool_pair_error:
            # Auto-recovery: clear context and retry with current message only
            logger.warning("Tool pair 400 error — clearing context and retrying: %s", e)
            _log_event("warning", "chat", f"Tool pair error auto-recovery: {e}")
            _clear_chat_history(user_id)
            try:
                fresh_msgs = [{"role": "user", "content": user_text}]
                progress_cb = _make_progress_callback(message.chat.id)
                reply = await _chat_with_tools(fresh_msgs, on_progress=progress_cb)
                if hasattr(progress_cb, "flush"):
                    await progress_cb.flush()
            except Exception as e2:
                logger.error("Retry after clear also failed: %s", e2)
                reply = f"오류가 발생했습니다 (자동 복구 실패): {e2}"
        else:
            logger.error("Claude API error: %s", e)
            _log_event("error", "chat", f"Claude API error: {e}", detail=user_text[:500])
            reply = f"오류가 발생했습니다: {e}"

    # Mission: log interrupted tool work to active mission
    if bt.get("was_interrupted") and bt.get("tool_work_details"):
        try:
            from telegram_mission import get_active_mission, add_mission_event
            mission = await asyncio.to_thread(get_active_mission, user_id)
            if mission:
                details = bt["tool_work_details"]
                event_content = (
                    f"Chat interrupted (rounds={bt.get('rounds_used', '?')}, "
                    f"cost=${bt.get('total_cost', 0):.2f})\n"
                    + "\n".join(details[:20])
                )[:2000]
                await asyncio.to_thread(
                    add_mission_event, mission["id"], "chat", "tool_result", event_content
                )
        except Exception as e:
            logger.debug("Mission event for interrupted chat failed: %s", e)

    # Auto-escalation: extract [CONTINUE_TASK: ...] marker and create background task
    continuation_task = None
    if "[CONTINUE_TASK:" in reply:
        import re
        match = re.search(r"\[CONTINUE_TASK:\s*(.+?)\]", reply, re.DOTALL)
        if match:
            continuation_task = match.group(1).strip()
            # Remove the marker from the reply shown to user
            reply = reply[:match.start()].rstrip()

    # Save assistant reply to DB, then try to create chunk summary in background
    await asyncio.to_thread(_save_chat_message, user_id, "assistant", reply)
    asyncio.create_task(_maybe_summarize_chunk(user_id))

    for chunk in _split_message(reply):
        await message.answer(chunk)

    # Create background task for unfinished work
    if continuation_task:
        task_content = f"[자동 승격] 대화 중 미완료 작업 이어서 수행:\n{continuation_task}\n\n원래 질문: {user_text[:500]}"
        # Inherit active mission
        cont_mission_id = None
        try:
            from telegram_mission import get_active_mission
            m = await asyncio.to_thread(get_active_mission, user_id)
            if m:
                cont_mission_id = m["id"]
        except Exception:
            pass
        task_row = await asyncio.to_thread(
            _query_one,
            "INSERT INTO telegram_tasks (user_id, content, status, mission_id, agent_type) VALUES (%s, %s, 'pending', %s, 'general') RETURNING id",
            (user_id, task_content, cont_mission_id),
        )
        task_id = task_row["id"] if task_row else "?"
        await message.answer(f"🔄 미완료 작업을 백그라운드 태스크 `[{task_id}]`로 자동 생성했습니다. 완료되면 알려드리겠습니다.")

    # Auto-reflection: every 5 exchanges, reflect on recent conversations
    _reflection_counter[user_id] = _reflection_counter.get(user_id, 0) + 1
    if _reflection_counter[user_id] >= 5:
        _reflection_counter[user_id] = 0
        asyncio.create_task(_reflect_on_recent(user_id))


# ── Auto-Recall & Reflection (experiential learning) ─────────────────
_reflection_counter: dict[int, int] = {}


async def _fetch_relevant_experiences(user_text: str) -> str:
    """Search experiential_memory for insights relevant to the user's message."""
    try:
        from shared import search_experiential_memory
        results = await asyncio.to_thread(search_experiential_memory, user_text, 3)
        if not results:
            return ""
        lines = []
        for r in results:
            cat = r.get("category", "?")
            lines.append(f"- [{cat}] {r['content']}")
        body = "\n".join(lines)
        return f"\n<past-experiences>\n{body}\n위 경험을 참고하되, 현재 대화 맥락에 맞게 판단해라.\n</past-experiences>"
    except Exception as e:
        logger.debug("Experience recall failed (non-critical): %s", e)
        return ""

_REFLECTION_PROMPT = """\
아래 대화에서 배울 점을 추출해라. 다음 카테고리별로 1개씩만 (해당 없으면 생략):

- **lesson**: 새로 배운 사실이나 지식
- **mistake**: 잘못된 답변, 도구 오용, 사용자 수정이 있었던 부분
- **pattern**: 반복적인 사용자 요구나 질문 패턴
- **insight**: 분석/논의에서 도출된 깊은 통찰
- **observation**: 기술적 발견이나 시스템 동작에 대한 관찰

각 항목을 한 줄로, 앞에 카테고리를 붙여 작성. 예:
lesson: 시리아 내전에서 러시아의 군사 개입은 2015년부터이며...
mistake: 사용자가 물어본 것은 경제 제재인데 군사적 측면만 답변했음
pattern: 사용자는 자주 한국 정치와 국제 정세의 연관성을 묻는다

배울 게 없으면 "NONE"이라고만 답해.

대화:
"""


async def _reflect_on_recent(user_id: int):
    """Background task: reflect on recent conversations and save insights."""
    try:
        history = await asyncio.to_thread(_load_chat_history, user_id)
        if len(history) < 4:
            return  # too little to reflect on

        # Build conversation text for reflection
        conv_text = "\n".join(
            f"[{m['role']}] {m['content'][:500]}" for m in history
        )
        prompt = _REFLECTION_PROMPT + conv_text

        # Try local LLM first (free), fall back to Haiku (paid)
        result = await _local_llm_generate(prompt)
        if not result:
            resp = await _claude.messages.create(
                model=await _get_model_light(),
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            result = _extract_text(resp).strip()

        if result.upper() == "NONE":
            logger.info("Reflection: nothing to learn from recent conversation")
            return

        # Parse and save each insight
        from shared import save_experiential_memory
        valid_categories = {"lesson", "mistake", "pattern", "insight", "observation"}
        saved = 0
        for line in result.split("\n"):
            line = line.strip().lstrip("- ")
            if ":" not in line:
                continue
            cat, content = line.split(":", 1)
            cat = cat.strip().lower()
            content = content.strip()
            if cat in valid_categories and len(content) > 10:
                success = await asyncio.to_thread(
                    save_experiential_memory, content, cat, "auto_reflection"
                )
                if success:
                    saved += 1

        if saved:
            logger.info("Reflection: saved %d experience(s) from user %d conversation", saved, user_id)
    except Exception as e:
        logger.warning("Reflection failed: %s", e)


# ═══════════════════════════════════════════════════════════════
#  자가수정 핸들러 — Telegram 전용 (chatbot.py에는 없음)
# ═══════════════════════════════════════════════════════════════

@router.message(Command("modify"))
async def cmd_modify(message: Message):
    """자가수정 명령어 — 허가된 Telegram 사용자만"""
    if not _is_allowed(message.from_user.id):
        return
    import os as _os, time as _time, uuid as _uuid
    content = (message.text or "").removeprefix("/modify").strip()
    parts = content.split("|", 2)
    if len(parts) != 3:
        await message.answer(
            "사용법:\n`/modify <파일경로> | <수정이유> | <새 내용 전체>`",
            parse_mode="Markdown"
        )
        return

    filepath, reason, new_content = [p.strip() for p in parts]

    # 경로 보안: leninbot 디렉토리 밖 거부
    base = "/home/grass/leninbot"
    abs_path = _os.path.realpath(_os.path.join(base, filepath))
    if not (abs_path == base or abs_path.startswith(base + "/")):
        await message.answer("❌ 허용된 디렉토리 밖의 파일은 수정할 수 없어.")
        return
    if not _os.path.isfile(abs_path):
        await message.answer(f"❌ 파일을 찾을 수 없어: `{filepath}`", parse_mode="Markdown")
        return

    # diff 생성
    import difflib as _dl
    try:
        with open(abs_path, "r", encoding="utf-8") as _f:
            old_content = _f.read()
        diff_lines = list(_dl.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm=""
        ))
        diff_text = "".join(diff_lines)
        insertions = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        deletions  = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    except Exception as e:
        await message.answer(f"❌ diff 생성 실패: {e}")
        return

    if not diff_lines:
        await message.answer("ℹ️ 변경사항 없음. 현재 파일과 동일해.")
        return

    # 승인 대기 등록 (5분 유효)
    approval_id = str(_uuid.uuid4())[:8]
    _pending_approvals[approval_id] = {
        "filepath": abs_path,
        "new_content": new_content,
        "reason": reason,
        "expire": _time.time() + 300,
    }

    diff_preview = diff_text[:3000] + ("\n…(생략)…" if len(diff_text) > 3000 else "")
    summary = (
        f"📝 *자가수정 요청*\n"
        f"파일: `{filepath}`\n"
        f"이유: {reason}\n"
        f"변경: +{insertions} / -{deletions} 라인\n\n"
        f"```\n{diff_preview}\n```"
    )
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ 승인", callback_data=f"selfmod_approve:{approval_id}"),
        InlineKeyboardButton(text="❌ 거부", callback_data=f"selfmod_reject:{approval_id}"),
    ]])
    await message.answer(summary, parse_mode="Markdown", reply_markup=kb)


@router.callback_query(F.data.startswith("selfmod_approve:"))
async def cb_modify_approve(callback: CallbackQuery):
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    import time as _time
    approval_id = callback.data.split(":", 1)[1]
    entry = _pending_approvals.pop(approval_id, None)
    if entry is None:
        await callback.message.edit_text("⚠️ 승인 정보를 찾을 수 없어. 만료됐거나 이미 처리됨.")
        return
    if _time.time() > entry["expire"]:
        await callback.message.edit_text("⏰ 승인 시간 초과 (5분). 다시 `/modify`를 실행해.")
        return

    await callback.message.edit_text("⚙️ 패치 적용 중…")
    await callback.answer()

    sys.path.insert(0, "/home/grass/leninbot")
    from self_modification_core import self_modify_with_safety
    try:
        result = await asyncio.to_thread(
            self_modify_with_safety,
            filepath=entry["filepath"],
            new_content=entry["new_content"],
            reason=entry["reason"],
            request_approval=False,
            skip_tests=False,
        )
    except Exception as e:
        await callback.message.edit_text(
            f"❌ 패치 적용 중 예외 발생:\n`{e}`", parse_mode="Markdown"
        )
        return

    if result.status == "success":
        commit_info = f"\n커밋: `{result.commit_hash}`" if result.commit_hash else ""
        await callback.message.edit_text(
            f"✅ *패치 완료*\n"
            f"파일: `{result.filepath}`\n"
            f"변경: {result.changes_count} 라인{commit_info}\n"
            f"⚠️ 재시작 후 적용됩니다.",
            parse_mode="Markdown"
        )
    else:
        await callback.message.edit_text(
            f"❌ *패치 실패* ({result.status})\n`{result.error}`",
            parse_mode="Markdown"
        )


@router.callback_query(F.data.startswith("selfmod_reject:"))
async def cb_modify_reject(callback: CallbackQuery):
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    approval_id = callback.data.split(":", 1)[1]
    _pending_approvals.pop(approval_id, None)
    await callback.message.edit_text("❌ 수정 거부됨. 원본 파일 유지.")
    await callback.answer()


# ── Config Panel ────────────────────────────────────────────────────

def _config_display_value(key: str, val) -> str:
    """Format a config value for display, with model tier → actual model mapping."""
    meta = _CONFIG_META[key]
    unit = meta["unit"]
    if key in ("chat_model", "task_model"):
        return _tier_to_display(val)
    return f"{unit}{val}" if unit == "$" else f"{val}{unit}"


def _config_summary() -> str:
    """Build a text summary of current config values."""
    lines = ["*현재 설정*\n"]
    for key, meta in _CONFIG_META.items():
        val = _config[key]
        display = _config_display_value(key, val)
        lines.append(f"  {meta['label']}: `{display}`")
    return "\n".join(lines)


def _config_main_keyboard() -> InlineKeyboardMarkup:
    """Build the main config panel keyboard — one button per setting."""
    rows = []
    for key, meta in _CONFIG_META.items():
        val = _config[key]
        display = _config_display_value(key, val)
        rows.append([InlineKeyboardButton(
            text=f"{meta['label']}: {display}",
            callback_data=f"cfg_select:{key}",
        )])
    rows.append([InlineKeyboardButton(text="닫기", callback_data="cfg_close")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _config_options_keyboard(key: str) -> InlineKeyboardMarkup:
    """Build option selection keyboard for a specific config key."""
    meta = _CONFIG_META[key]
    current = _config[key]
    buttons = []
    for opt in meta["options"]:
        display = _config_display_value(key, opt)
        marker = " ✓" if opt == current else ""
        buttons.append(InlineKeyboardButton(
            text=f"{display}{marker}",
            callback_data=f"cfg_set:{key}:{opt}",
        ))
    # Arrange in rows of 3
    rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
    rows.append([InlineKeyboardButton(text="← 뒤로", callback_data="cfg_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(Command("config"))
async def cmd_config(message: Message):
    """Open the config panel."""
    if not _is_allowed(message.from_user.id):
        return
    await message.answer(
        _config_summary(),
        parse_mode="Markdown",
        reply_markup=_config_main_keyboard(),
    )


@router.callback_query(F.data.startswith("cfg_select:"))
async def cb_config_select(callback: CallbackQuery):
    """User tapped a config key — show options."""
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    key = callback.data.split(":", 1)[1]
    if key not in _CONFIG_META:
        await callback.answer("알 수 없는 설정", show_alert=True)
        return
    meta = _CONFIG_META[key]
    await callback.message.edit_text(
        f"*{meta['label']}* 변경\n현재: `{_config[key]}`",
        parse_mode="Markdown",
        reply_markup=_config_options_keyboard(key),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("cfg_set:"))
async def cb_config_set(callback: CallbackQuery):
    """User selected a new value for a config key."""
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    parts = callback.data.split(":", 2)
    if len(parts) != 3:
        await callback.answer("잘못된 데이터", show_alert=True)
        return
    key, raw_val = parts[1], parts[2]
    if key not in _CONFIG_META:
        await callback.answer("알 수 없는 설정", show_alert=True)
        return

    # Convert value to the right type
    current = _config[key]
    if isinstance(current, float):
        new_val = float(raw_val)
    elif isinstance(current, int):
        new_val = int(raw_val)
    else:
        new_val = raw_val

    old_val = _config[key]
    _config[key] = new_val

    # If model changed, clear resolved cache so it re-resolves on next use
    if key in ("chat_model", "task_model") and new_val != old_val:
        _resolved_models.pop(new_val, None)

    logger.info("Config changed: %s = %s → %s", key, old_val, new_val)
    _save_config()
    await callback.answer(f"{_CONFIG_META[key]['label']}: {new_val}")

    # Return to main config panel
    await callback.message.edit_text(
        _config_summary(),
        parse_mode="Markdown",
        reply_markup=_config_main_keyboard(),
    )


@router.callback_query(F.data == "cfg_back")
async def cb_config_back(callback: CallbackQuery):
    """Return to main config panel."""
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    await callback.message.edit_text(
        _config_summary(),
        parse_mode="Markdown",
        reply_markup=_config_main_keyboard(),
    )
    await callback.answer()


@router.callback_query(F.data == "cfg_close")
async def cb_config_close(callback: CallbackQuery):
    """Close the config panel."""
    if not _is_allowed(callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    await callback.message.edit_text("설정 패널을 닫았습니다.")
    await callback.answer()


# ── Entry Point ──────────────────────────────────────────────────────

async def bot_main():
    """Start the Telegram bot. Callable from api.py lifespan or standalone."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set, skipping bot")
        return
    if not ALLOWED_USER_IDS:
        logger.warning("ALLOWED_USER_IDS not set, skipping bot")
        return

    # Ensure task table exists
    await asyncio.to_thread(_ensure_table)
    recovery = await recover_processing_tasks_on_startup(stale_minutes=60, max_resume_attempts=2)
    handed_off = int(recovery.get("handed_off", recovery.get("resumed", 0)))
    closed_stale = int(recovery.get("closed_stale", 0))
    closed_repeated = int(recovery.get("closed_repeated", 0))
    if handed_off or closed_stale or closed_repeated:
        _add_system_alert(
            f"재시작 복구: handoff {handed_off}건 / 오래된 작업 종료 {closed_stale}건 / 반복 중단 작업 종료 {closed_repeated}건"
        )

    global _bot_instance
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    _bot_instance = bot
    dp = Dispatcher()
    dp.include_router(router)

    # Register commands for Telegram "/" autocomplete menu
    from aiogram.types import BotCommand
    await bot.set_my_commands([
        BotCommand(command="help", description="커맨드 목록"),
        BotCommand(command="chat", description="CLAW 파이프라인 질의"),
        BotCommand(command="task", description="백그라운드 태스크 등록"),
        BotCommand(command="status", description="시스템 대시보드"),
        BotCommand(command="stats", description="시스템 리소스 현황"),
        BotCommand(command="status_auto", description="자율 생성 태스크 확인"),
        BotCommand(command="report", description="태스크 리포트 재전송"),
        BotCommand(command="schedule", description="정기 태스크 등록"),
        BotCommand(command="schedules", description="등록된 스케줄 목록"),
        BotCommand(command="unschedule", description="스케줄 삭제"),
        BotCommand(command="kg", description="지식그래프 현황"),
        BotCommand(command="config", description="설정 패널"),
        BotCommand(command="fallback", description="모델 토글 (sonnet↔haiku)"),
        BotCommand(command="errors", description="에러/경고 로그"),
        BotCommand(command="restart", description="서비스 재시작"),
        BotCommand(command="deploy", description="서버 배포 (git pull)"),
        BotCommand(command="modify", description="서버 파일 수정"),
        BotCommand(command="mission", description="미션 상태 / close"),
        BotCommand(command="clear", description="대화 히스토리 초기화"),
    ])

    # Detect fresh deploy — inject context so the bot knows it was just updated
    await check_deploy_meta(bot, add_alert_fn=_add_system_alert)

    # Build process_task closure with module-level dependencies
    async def _process_task_wrapper(b: Bot, task: dict):
        from self_tools import build_task_context_tools
        from telegram_tools import TOOLS as BASE_TOOLS, TOOL_HANDLERS as BASE_HANDLERS
        from telegram_tools import MISSION_TOOL, build_mission_handler

        # ── Agent-aware task execution ──────────────────────────────
        agent_type = task.get("agent_type") or "general"
        try:
            from agents import get_agent
            spec = get_agent(agent_type)
        except (ValueError, ImportError):
            # Fallback: unknown agent_type treated as general
            from agents import get_agent
            spec = get_agent("general")

        # Filter base tools to agent's allowed set
        agent_tools, agent_handlers = spec.filter_tools(BASE_TOOLS, BASE_HANDLERS)

        # Add task-context tools (save_finding, request_continuation)
        ctx_tools, ctx_handlers = build_task_context_tools(
            task["id"], task["user_id"], task.get("depth", 0),
            mission_id=task.get("mission_id"),
        )
        agent_tools.extend(ctx_tools)
        agent_handlers.update(ctx_handlers)

        # Add mission tool
        agent_tools.append(MISSION_TOOL)
        agent_handlers["mission"] = build_mission_handler(task["user_id"])

        # Render agent-specific system prompt
        system_prompt = spec.render_prompt(
            current_datetime=_current_datetime_str(),
            system_alerts=_format_system_alerts(),
            finance_data=_get_finance_context(),
        )

        # Send progress to the task's user (or all users if self-generated)
        target_chat_id = task["user_id"] if task["user_id"] != 0 else next(iter(ALLOWED_USER_IDS), 0)
        progress_cb = _make_progress_callback(target_chat_id) if target_chat_id else None

        # ── Provider dispatch: Claude vs MOON PC (OpenAI-compatible) ──
        if spec.provider == "moon":
            from openai_tool_loop import chat_with_tools as moon_chat_with_tools
            from llm_client import MOON_BASE, MOON_MODEL, _health_ok

            if not _health_ok(MOON_BASE):
                logger.warning("MOON PC unavailable for agent %s; falling back to Claude", spec.name)
                chosen_chat_fn = _chat_with_tools
                chosen_model_fn = _get_model_task
                chosen_max_tokens = _CLAUDE_MAX_TOKENS_TASK
            else:
                async def _moon_chat_with_tools(
                    messages, max_rounds=None, system_prompt=None, model=None,
                    max_tokens=None, budget_usd=None, extra_tools=None,
                    extra_handlers=None, on_progress=None, budget_tracker=None,
                ):
                    merged_tools = list(extra_tools or []) + agent_tools
                    merged_handlers = {**agent_handlers, **(extra_handlers or {})}
                    return await moon_chat_with_tools(
                        messages,
                        base_url=MOON_BASE,
                        model=model or MOON_MODEL,
                        tools=merged_tools,
                        tool_handlers=merged_handlers,
                        system_prompt=system_prompt or system_prompt,
                        max_rounds=max_rounds or spec.max_rounds,
                        max_tokens=max_tokens or 4096,
                        log_event=_log_event,
                        budget_usd=budget_usd or 0.0,
                        budget_tracker=budget_tracker,
                        on_progress=on_progress,
                    )
                chosen_chat_fn = _moon_chat_with_tools
                chosen_model_fn = _get_model_moon
                chosen_max_tokens = 4096
        else:
            chosen_chat_fn = _chat_with_tools
            chosen_model_fn = _get_model_task
            chosen_max_tokens = _CLAUDE_MAX_TOKENS_TASK

        def _on_task_complete(task_id: int, status: str, summary: str):
            icon = "✅" if status == "done" else "❌"
            _add_system_alert(f"{icon} 태스크 #{task_id} {status}: {summary[:200]}")

        await process_task(
            b, task,
            chat_with_tools_fn=chosen_chat_fn,
            get_model_fn=chosen_model_fn,
            task_system_prompt=system_prompt,
            max_tokens_task=chosen_max_tokens,
            allowed_user_ids=ALLOWED_USER_IDS,
            log_event_fn=_log_event,
            extra_tools=agent_tools,
            extra_handlers=agent_handlers,
            budget_usd=spec.budget_usd,
            on_progress=progress_cb,
            on_complete=_on_task_complete,
        )
        # Flush remaining progress buffer
        if progress_cb and hasattr(progress_cb, "flush"):
            await progress_cb.flush()

    # Start background workers (keep handles for graceful cancellation)
    _runtime_state: dict[str, int | None] = {"current_task_id": None}
    _bg_tasks = [
        asyncio.create_task(
            task_worker(bot, process_task_fn=_process_task_wrapper, runtime_state=_runtime_state),
            name="task_worker",
        ),
        asyncio.create_task(
            system_monitor(
                bot,
                allowed_user_ids=ALLOWED_USER_IDS,
                add_alert_fn=_add_system_alert,
                clear_alert_fn=_clear_system_alert,
            ),
            name="system_monitor",
        ),
        asyncio.create_task(
            schedule_worker(bot, allowed_user_ids=ALLOWED_USER_IDS),
            name="schedule_worker",
        ),
    ]

    # Graceful shutdown: notify + stop polling cleanly when SIGTERM received (Render deploy)
    import signal

    def _handle_sigterm(*_):
        logger.info("SIGTERM received — stopping polling gracefully")
        # Schedule shutdown notification before stopping
        async def _shutdown_notify_and_checkpoint():
            task_id = _runtime_state.get("current_task_id")
            if task_id:
                ok = await checkpoint_task_on_shutdown(int(task_id))
                if ok:
                    logger.info("Shutdown checkpoint saved for in-flight task #%s", task_id)
            await broadcast(bot, "🔄 *서버 재시작 중* — 새 버전 배포가 시작됩니다.", ALLOWED_USER_IDS)
        try:
            asyncio.get_event_loop().create_task(_shutdown_notify_and_checkpoint())
        except Exception:
            pass
        asyncio.get_event_loop().call_soon_threadsafe(dp.stop_polling)

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (ValueError, OSError):
        pass  # signal only works in main thread; skip if called from a thread

    logger.info("Bot starting (allowed users: %s)", ALLOWED_USER_IDS)

    # Notify when polling is actually ready to receive messages
    async def _notify_ready():
        """Wait for polling to start, then send ready notification."""
        await asyncio.sleep(2)  # brief wait for polling loop to initialize
        from shared import get_kg_service
        kg = await asyncio.to_thread(get_kg_service)
        kg_status = "connected" if kg else "unavailable"
        _add_system_alert(f"Deploy 완료 — KG: {kg_status}")
        await broadcast(bot, (
            f"🟢 *Deploy 완료* — 메시지 수신 준비 완료.\n"
            f"  KG (Neo4j): {kg_status}"
        ), ALLOWED_USER_IDS)

    asyncio.create_task(_notify_ready(), name="startup_notify")

    # drop_pending_updates: new instance takes over quickly, avoids processing stale updates
    await dp.start_polling(bot, drop_pending_updates=True)
    # After polling stops — graceful shutdown sequence
    # 1. Cancel background tasks
    for t in _bg_tasks:
        t.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)
    logger.info("Background tasks cancelled")

    # 2. Release Telegram session
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.session.close()
    except Exception:
        pass



if __name__ == "__main__":
    asyncio.run(bot_main())
