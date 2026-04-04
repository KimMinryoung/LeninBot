"""telegram_bot.py — Telegram bot core: chat history, system prompt, LLM dispatch, bot_main.

Command handlers → telegram_commands.py
LLM config/model resolution → bot_config.py
Tool definitions → telegram_tools.py
Background tasks → telegram_tasks.py
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from shared import KST, CORE_IDENTITY
from skills_loader import build_skills_prompt
from db import query as _query, execute as _execute, query_one as _query_one, get_conn as _get_conn
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router

# Extracted modules
from bot_config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    _claude, _openai_client,
    _CLAUDE_MAX_TOKENS, _CLAUDE_MAX_TOKENS_TASK,
    _config, _save_config, _CONFIG_DEFAULTS, _CONFIG_META,
    _resolved_models, _tier_to_display,
    _get_model, _get_model_task, _get_model_light, _get_model_moon,
    get_current_model_selection,
    _extract_text,
)
from telegram_tools import TOOLS, TOOL_HANDLERS
from claude_loop import chat_with_tools, dedupe_tools_by_name
from telegram_tasks import (
    process_task, system_monitor,
    task_worker, schedule_worker, check_deploy_meta,
    recover_processing_tasks_on_startup,
    checkpoint_task_on_shutdown, persist_task_restart_state,
    _delegate_to_browser_worker, check_browser_worker_alive,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

_runtime_state: dict = {"active_task_ids": set()}

# Per-coroutine task context — allows concurrent tasks to know their own task_id
import contextvars
current_task_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_task_ctx", default=None)

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
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}
# Single-owner enforcement: all outbound messages go to this user only.
OWNER_USER_ID: int = next(iter(ALLOWED_USER_IDS)) if len(ALLOWED_USER_IDS) == 1 else 0
EMAIL_BRIDGE_ENABLED = os.getenv("EMAIL_BRIDGE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
EMAIL_POLL_INTERVAL_SECONDS = max(30, int(os.getenv("EMAIL_POLL_INTERVAL_SECONDS", "120")))
EMAIL_APPROVAL_BASE_URL = os.getenv("EMAIL_APPROVAL_BASE_URL", "").rstrip("/")
EMAIL_DEFAULT_APPROVER_USER_ID = int(os.getenv("EMAIL_DEFAULT_APPROVER_USER_ID", "0") or "0")
EMAIL_LOG_DIR = Path(os.getenv("EMAIL_LOG_DIR", str(Path(__file__).resolve().parent / "logs" / "email_bridge")))

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
        CREATE TABLE IF NOT EXISTS telegram_system_events (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            event_type  VARCHAR(50) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
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
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS agent_type VARCHAR(50)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS mission_id INTEGER")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS tool_log TEXT DEFAULT ''")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS metadata JSONB")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS verification_status VARCHAR(20) DEFAULT 'pending'")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_initiated BOOLEAN DEFAULT FALSE")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_target_service VARCHAR(20)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_completed BOOLEAN DEFAULT FALSE")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS post_restart_phase VARCHAR(50)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_attempt_count INTEGER DEFAULT 0")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_requested_at TIMESTAMPTZ")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS resumed_after_restart BOOLEAN DEFAULT FALSE")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_reentry_block_reason TEXT")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS verification_details TEXT")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS verification_attempts INTEGER DEFAULT 0")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS last_verification_at TIMESTAMPTZ")
    # Task group columns for parallel delegation + synthesis
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS plan_id INTEGER")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS plan_role VARCHAR(20)")
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_parent
        ON telegram_tasks(parent_task_id) WHERE parent_task_id IS NOT NULL
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_plan
        ON telegram_tasks(plan_id) WHERE plan_id IS NOT NULL
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_agent_user
        ON telegram_tasks(user_id, agent_type, status) WHERE status = 'done'
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
    _execute("""
        CREATE TABLE IF NOT EXISTS email_threads (
            id SERIAL PRIMARY KEY,
            provider VARCHAR(50) NOT NULL DEFAULT 'imap_smtp',
            external_thread_id VARCHAR(255),
            subject TEXT,
            participants JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(provider, external_thread_id)
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS email_messages (
            id SERIAL PRIMARY KEY,
            thread_id INTEGER REFERENCES email_threads(id) ON DELETE SET NULL,
            provider VARCHAR(50) NOT NULL DEFAULT 'imap_smtp',
            direction VARCHAR(20) NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'received',
            mailbox VARCHAR(50),
            external_message_id VARCHAR(255),
            in_reply_to VARCHAR(255),
            sender_email TEXT,
            sender_name TEXT,
            recipient_emails JSONB NOT NULL DEFAULT '[]'::jsonb,
            cc_emails JSONB NOT NULL DEFAULT '[]'::jsonb,
            bcc_emails JSONB NOT NULL DEFAULT '[]'::jsonb,
            subject TEXT,
            text_body TEXT,
            html_body TEXT,
            raw_headers JSONB NOT NULL DEFAULT '{}'::jsonb,
            attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            received_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            approved_by BIGINT,
            approved_at TIMESTAMPTZ,
            approval_note TEXT,
            sent_at TIMESTAMPTZ,
            draft_saved_at TIMESTAMPTZ,
            audit_log JSONB NOT NULL DEFAULT '[]'::jsonb,
            UNIQUE(provider, external_message_id)
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_email_messages_status_created
        ON email_messages(status, created_at DESC)
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_email_messages_thread_created
        ON email_messages(thread_id, created_at DESC)
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_email_messages_provider_imap_uid
        ON email_messages(provider, ((metadata->>'imap_uid')))
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS email_bridge_events (
            id SERIAL PRIMARY KEY,
            message_id INTEGER REFERENCES email_messages(id) ON DELETE CASCADE,
            event_type VARCHAR(50) NOT NULL,
            detail TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS email_bridge_state (
            provider VARCHAR(50) PRIMARY KEY,
            state JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS file_registry (
            id SERIAL PRIMARY KEY,
            local_path TEXT NOT NULL,
            public_url TEXT,
            filename TEXT NOT NULL,
            content_type VARCHAR(100),
            description TEXT,
            category VARCHAR(50) DEFAULT 'general',
            file_size BIGINT,
            created_by_task_id INTEGER,
            created_by_agent VARCHAR(50),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_file_registry_category
        ON file_registry(category, created_at DESC)
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


def _append_email_audit_entry(message_id: int, event: str, actor: str, metadata: dict | None = None) -> None:
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    _execute(
        """
        UPDATE email_messages
        SET audit_log = COALESCE(audit_log, '[]'::jsonb) || jsonb_build_array(
            jsonb_build_object(
                'at', NOW(),
                'event', %s,
                'actor', %s,
                'metadata', %s::jsonb
            )
        ),
        updated_at = NOW()
        WHERE id = %s
        """,
        (event[:50], actor[:100], metadata_json, message_id),
    )
    _execute(
        "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
        (message_id, event[:50], None, metadata_json),
    )


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


def _format_current_model_context(kind: str = "chat") -> str:
    """Format runtime-selected model info for prompt/context injection."""
    selection = get_current_model_selection(kind)
    return (
        f"<current-model provider=\"{selection['provider']}\" tier=\"{selection['tier']}\" "
        f"alias=\"{selection['alias']}\">{selection['model_id']}</current-model>"
    )


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


_env_context_cache: str | None = None
_env_context_ts: float = 0


def _build_env_context() -> str:
    """Build runtime environment info block for agents. Cached for 1 hour."""
    import time
    global _env_context_cache, _env_context_ts
    if _env_context_cache and (time.time() - _env_context_ts) < 3600:
        return _env_context_cache

    import subprocess, platform
    lines = ["<runtime-environment>"]

    # OS
    lines.append(f"OS: {platform.platform()}")

    # Python / venv
    venv = "/home/grass/leninbot/venv"
    lines.append(f"Python venv: {venv}/bin/python")
    lines.append("Always use this venv for package installation. No global installs (--break-system-packages).")

    # Key packages
    try:
        r = subprocess.run(
            [f"{venv}/bin/pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=10,
        )
        pkg_names = {"playwright", "browser-use", "anthropic", "aiogram", "langchain-core", "graphiti-core"}
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].lower() in pkg_names:
                lines.append(f"  {parts[0]}=={parts[1]}")
    except Exception:
        pass

    # Playwright browsers
    import glob
    chromium = glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
    if chromium:
        chromium.sort(reverse=True)
        lines.append(f"Playwright Chromium: {chromium[0]}")
    lines.append(f"Xvfb: {'available' if os.path.exists('/usr/bin/xvfb-run') else 'not found'}")

    # System permissions
    lines.append("sudo privileges: `sudo apt update/install` available (NOPASSWD).")
    lines.append("System packages: use `sudo apt install -y <pkg>`. No global pip installs.")

    # Services
    lines.append("Services (systemd): leninbot-telegram, leninbot-api, leninbot-embedding, leninbot-neo4j")
    lines.append("After modifying service code, restart via restart_service tool. Do not restart directly via subprocess.")

    lines.append("</runtime-environment>")
    _env_context_cache = "\n".join(lines)
    _env_context_ts = time.time()
    return _env_context_cache


_SYSTEM_PROMPT_TEMPLATE = CORE_IDENTITY + """
Operating via Telegram. Use tools proactively when data would improve the answer — don't rely on memory alone.

<tool-strategy>
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page, then analyze with context from other tools
- Self-reflection → read_self(source="diary"); cross-interface memory → read_self(source="chat_logs")
- Past lessons/mistakes → recall_experience (semantic search over accumulated daily insights)
- Store important facts → write_kg
- Real-time market prices → get_finance_data
</tool-strategy>

<context-isolation>
**You are the orchestrator. You have no access to programming tools (read_file, write_file, patch_file, list_directory, execute_python).**
If you need to read/modify/execute code, you must delegate via `delegate(agent="programmer")`.
Your role is to understand the user's intent, dispatch tasks to the appropriate agents, and synthesize results.
The `<current_state>` block contains structured completed/in-progress/pending tasks. Use it to avoid duplicate work and determine next steps. Detailed tool execution logs are only accessible to each agent itself.
</context-isolation>

<delegation>
You have specialized agents. Use the `delegate` tool to dispatch tasks:
- programmer: code writing/editing/debugging/file management ($1.50)
- analyst: default agent for information analysis/research. Web search + collection + KG cross-validation + pattern extraction + knowledge storage ($1.00)
- scout: routine patrols, large-scale platform crawling ($1.00)
- browser: AI browser automation — login, form input, multi-page navigation, dynamic site data extraction ($1.50)
- visualizer: image generation, visual concepts ($1.00)

When to delegate vs handle directly:
- Simple questions, casual conversation, quick lookups → handle directly
- **"analyze/investigate/look into"** → delegate(agent="analyst")
- **Code reading/editing/execution/file management** → delegate(agent="programmer")
- **Moltbook patrols, large-scale crawling** → delegate(agent="scout")
- **Website login, form submission, complex browser operations** → delegate(agent="browser")
- **Image generation** → delegate(agent="visualizer")
- **Multiple agents need to work simultaneously** → multi_delegate (parallel execution + automatic result synthesis)
- If a conversation looks like it will need 10+ tool calls, switch to delegate immediately.
- Do not ask the user "should I continue?" — judge and delegate on your own.

Parallel delegation with `multi_delegate`:
- Compound requests (e.g., "investigate X and fix Y's code") should be handled in parallel via multi_delegate.
- After all subtasks complete, a synthesis task automatically consolidates results.
- Specifying synthesis_instructions with consolidation criteria yields better results.

Context passing — agents automatically receive recent conversation and their own execution history, but specify the current conversation's key context in the `context` field:
1. The user's original request (verbatim or key summary)
2. Findings from the conversation so far (tool results, analysis, decisions)
3. Why you are delegating to this agent (reason and expected outcome)
</delegation>

<mission-management>
- Missions are auto-created when delegate is called. The user does not need to create them explicitly.
- Check the task status in `<current_state>` to determine whether the mission is complete.
- **Mission close condition**: Only call `mission(action="close")` when the user's original goal has been **fully achieved**.
- **Do NOT close when**:
  - Task results contain incomplete reasons such as "budget exhausted", "limit reached", "stopped due to error"
  - `<not_started>` or `<in_progress>` tasks remain
  - The user may issue follow-up work
- If incomplete tasks remain, delegate follow-up work or keep the mission open.
</mission-management>

<temporal-awareness>
Conversation history includes timestamps ([YYYY-MM-DD HH:MM]) on user messages.
Infer elapsed time from the timestamps. Large gaps may indicate context switches or changed circumstances.
</temporal-awareness>

<response-rules>
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
- Do not use markdown formatting (**, *, #, ```, - etc.) in Telegram messages. Write in plain text only, as a human would. Markdown is allowed only when writing files (.md).
</response-rules>

<context>
<current-time>{current_datetime}</current-time>
{current_model}
{system_alerts}
{skills_section}
</context>
"""


# ── Chat History ─────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10  # 10 pairs = 20 messages


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


def _save_system_event(user_id: int, event_type: str, content: str):
    """Save a system event to the dedicated events table (not chat history)."""
    _execute(
        "INSERT INTO telegram_system_events (user_id, event_type, content) VALUES (%s, %s, %s)",
        (user_id, event_type, content),
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


_RAW_MSG_LIMIT = 30  # recent raw messages to include


def _load_context_with_summaries(user_id: int) -> list[dict]:
    """Load chat context: chunk summaries + recent raw messages.

    Summaries are injected as a single context preamble (not fake conversation
    pairs) so the model sees a clear timeline:
      [context preamble with summaries]  →  [recent raw messages in order]
    Raw messages always include the most recent turns regardless of summary
    coverage, preventing context loss when summaries are stale or missing.
    """
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

    # Always load the most recent raw messages regardless of summary coverage.
    # This ensures recent context is never lost even when summaries are stale.
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, role, content, created_at FROM ("
                "  SELECT id, role, content, created_at FROM telegram_chat_history"
                "  WHERE user_id = %s AND id > %s ORDER BY id DESC LIMIT %s"
                ") sub ORDER BY id ASC",
                (user_id, min_id, _RAW_MSG_LIMIT),
            )
            raw_rows = cur.fetchall()

    # Determine which summaries are still useful (cover messages not fully
    # included in the raw window). Use <= to avoid gaps: a summary whose
    # chunk_end_id equals raw_min_id may overlap by one message, but that's
    # better than losing the older messages in that chunk entirely.
    raw_min_id = raw_rows[0]["id"] if raw_rows else None
    useful_summaries = []
    if raw_min_id is not None:
        useful_summaries = [s for s in summaries if s["chunk_end_id"] <= raw_min_id]

    # Build context: summary preamble + raw messages with timestamps
    context: list[dict] = []

    # Inject summaries as a single context block (not fake conversation pairs)
    if useful_summaries:
        summary_lines = []
        for s in useful_summaries:
            summary_lines.append(
                f"• (msgs #{s['chunk_start_id']}~#{s['chunk_end_id']}): {s['summary']}"
            )
        preamble = (
            "[Prior conversation summary — below is a summary of conversations before the recent messages]\n"
            + "\n".join(summary_lines)
        )
        context.append({"role": "user", "content": preamble})
        context.append({"role": "assistant", "content": "Acknowledged. I have reviewed the prior conversation context. Proceeding."})

    # Load system events that fall within the raw message time window.
    # These are interleaved chronologically with raw messages as inline
    # annotations — not as fake conversation pairs.
    system_events = []
    if raw_rows:
        first_ts = raw_rows[0].get("created_at")
        if first_ts:
            try:
                with _get_conn() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(
                            "SELECT event_type, content, created_at FROM telegram_system_events "
                            "WHERE user_id = %s AND created_at >= %s ORDER BY created_at ASC",
                            (user_id, first_ts),
                        )
                        system_events = cur.fetchall()
            except Exception:
                pass  # table might not exist yet

    # Build a queue of system events sorted by timestamp for interleaving
    event_idx = 0

    # Append raw messages with exact timestamps on user messages.
    # System events are interleaved chronologically.
    for r in raw_rows:
        role = r["role"] if r["role"] in ("user", "assistant") else "user"
        text = _normalize_history_content(r["content"])
        ts = r.get("created_at")

        if ts and hasattr(ts, "strftime") and role == "user":
            # Interleave system events that occurred before this message
            event_notes = []
            while event_idx < len(system_events):
                evt = system_events[event_idx]
                evt_ts = evt.get("created_at")
                if evt_ts and evt_ts <= ts:
                    evt_kst = evt_ts.astimezone(KST) if evt_ts.tzinfo else evt_ts
                    evt_time = evt_kst.strftime("%H:%M")
                    event_notes.append(f"[{evt_time} SYSTEM/{evt.get('event_type', '?')}] {evt.get('content', '')}")
                    event_idx += 1
                else:
                    break
            if event_notes:
                context.append({"role": "assistant", "content": "\n".join(event_notes)})

            ts_kst = ts.astimezone(KST) if ts.tzinfo else ts
            time_str = ts_kst.strftime("%Y-%m-%d %H:%M")
            text = f"[{time_str}] {text}" if text else f"[{time_str}]"

        context.append({"role": role, "content": text or "(empty)"})

    # Append any remaining system events after the last message
    trailing_events = []
    while event_idx < len(system_events):
        evt = system_events[event_idx]
        evt_ts = evt.get("created_at")
        if evt_ts and hasattr(evt_ts, "strftime"):
            evt_kst = evt_ts.astimezone(KST) if evt_ts.tzinfo else evt_ts
            evt_time = evt_kst.strftime("%H:%M")
        else:
            evt_time = "?"
        trailing_events.append(f"[{evt_time} SYSTEM/{evt.get('event_type', '?')}] {evt.get('content', '')}")
        event_idx += 1
    if trailing_events:
        context.append({"role": "assistant", "content": "\n".join(trailing_events)})

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
            "Summarize the conversation below concisely, keeping only key information. "
            "1) What topic/request did the user raise? "
            "2) What conclusions/answers/results were reached? "
            "3) Note any items still in progress or unresolved. "
            "Preserve proper nouns, numbers, dates, and specific decisions. 500 characters max.\n\n"
            + conversation_text
        )

        summary = await _local_llm_generate(summary_prompt)
        if not summary:
            resp = await _claude.messages.create(
                model=await _get_model_light(),
                max_tokens=768,
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
    task_id: int | None = None,
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
        current_model=_format_current_model_context("chat"),
        system_alerts=_format_system_alerts(),
        skills_section=build_skills_prompt(),
    )
    # Orchestrator context isolation: programming tools are reserved for programmer agent.
    _ORCHESTRATOR_BLOCKED_TOOLS = {"read_file", "write_file", "patch_file", "list_directory", "execute_python"}
    is_orchestrator = extra_tools is None

    if is_orchestrator:
        # Orchestrator: use all TOOLS except blocked ones
        seen_names: set[str] = set()
        merged_tools: list[dict] = []
        for t in TOOLS:
            name = t.get("name", "")
            if name not in seen_names:
                if name in _ORCHESTRATOR_BLOCKED_TOOLS:
                    continue
                seen_names.add(name)
                merged_tools.append(t)
    else:
        # Task/agent: use ONLY extra_tools (already filtered by agent spec).
        # Do NOT merge full TOOLS — that would bypass agent tool restrictions.
        merged_tools = list(extra_tools or [])
    # Orchestrator: full handler set. Task/agent: only the handlers for allowed tools.
    if is_orchestrator:
        merged_handlers = {**TOOL_HANDLERS, **(extra_handlers or {})}
    else:
        merged_handlers = dict(extra_handlers or {})

    # Inject run_agent handler (needs _chat_with_tools closure — can't be registered at import time)
    if is_orchestrator and "run_agent" not in merged_handlers:
        from self_tools import build_run_agent_handler
        merged_handlers["run_agent"] = build_run_agent_handler(_chat_with_tools)

    # ── Provider dispatch: Claude vs OpenAI vs Local ──
    if _config.get("provider") == "local":
        from openai_tool_loop import chat_with_tools as openai_chat
        from llm_client import _resolve_backend, LOCAL_SEMAPHORE
        backend = _resolve_backend()
        async with LOCAL_SEMAPHORE:
            return await openai_chat(
                messages,
                client=None,
                base_url=backend["base"],
                model=model or backend["model"],
                tools=merged_tools,
                tool_handlers=merged_handlers,
                system_prompt=sys_prompt,
                max_rounds=resolved_max_rounds,
                max_tokens=resolved_max_tokens,
                log_event=_log_event,
                budget_usd=resolved_budget,
                on_progress=on_progress,
                budget_tracker=budget_tracker,
                task_id=task_id,
            )

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
            task_id=task_id,
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
        task_id=task_id,
    )


# ── Router & Handlers ───────────────────────────────────────────────
router = Router()

# Register command handlers from extracted module
from telegram_commands import register_handlers
register_handlers(router, ctx={
    "is_allowed": _is_allowed,
    "split_message": _split_message,
    "save_chat_message": _save_chat_message,
    "save_system_event": _save_system_event,
    "load_chat_history": _load_chat_history,
    "load_context_with_summaries": _load_context_with_summaries,
    "clear_chat_history": _clear_chat_history,
    "log_event": _log_event,
    "config": _config,
    "save_config": _save_config,
    "CONFIG_META": _CONFIG_META,
    "resolved_models": _resolved_models,
    "tier_to_display": _tier_to_display,
    "chat_with_tools": _chat_with_tools,
    "get_model": _get_model,
    "make_progress_callback": _make_progress_callback,
    "SYSTEM_PROMPT_TEMPLATE": _SYSTEM_PROMPT_TEMPLATE,
    "current_datetime_str": _current_datetime_str,
    "format_current_model_context": _format_current_model_context,
    "format_system_alerts": _format_system_alerts,
    "add_system_alert": _add_system_alert,
    "clear_system_alert": _clear_system_alert,
    "claude_client": _claude,
    "openai_client": _openai_client,
    "extract_text": _extract_text,
    "local_llm_generate": _local_llm_generate,
    "get_model_light": _get_model_light,
    "maybe_summarize_chunk": _maybe_summarize_chunk,
    "build_skills_prompt": build_skills_prompt,
    "ALLOWED_USER_IDS": ALLOWED_USER_IDS,
    "CLAUDE_MAX_TOKENS": _CLAUDE_MAX_TOKENS,
    "email_approval_base_url": EMAIL_APPROVAL_BASE_URL,
})


# ── Entry Point ──────────────────────────────────────────────────────

async def _email_bridge_poll_loop(bot: Bot):
    from email_bridge import build_inbound_summary_notification, get_email_message, run_polling_cycle
    await asyncio.sleep(15)
    while True:
        try:
            result = await asyncio.to_thread(run_polling_cycle, 10)
            processed = result.get("processed") or []
            for item in processed:
                if item.get("processing_status") != "stored":
                    continue
                stored_message_id = item.get("stored_message_id")
                if not stored_message_id:
                    continue
                row = await asyncio.to_thread(get_email_message, stored_message_id)
                if not row:
                    continue
                target_chat_id = result.get("operations_chat_id") or EMAIL_DEFAULT_APPROVER_USER_ID
                if target_chat_id:
                    text = build_inbound_summary_notification(item, row)
                    text += "\n\n승인 후 내부 입력 전달: /email_deliver <inbound_id> [메모]"
                    await bot.send_message(int(target_chat_id), text, parse_mode="Markdown")
        except Exception as e:
            logger.warning("email bridge poll loop error: %s", e)
            _log_event("warning", "email_bridge", f"poll loop error: {e}")
        await asyncio.sleep(EMAIL_POLL_INTERVAL_SECONDS)


async def bot_main():
    """Start the Telegram bot. Callable from api.py lifespan or standalone."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set, skipping bot")
        return
    if not ALLOWED_USER_IDS:
        logger.warning("ALLOWED_USER_IDS not set, skipping bot")
        return
    if len(ALLOWED_USER_IDS) > 1:
        logger.error("Security: ALLOWED_USER_IDS must contain exactly one user ID, got %d. Aborting.", len(ALLOWED_USER_IDS))
        return

    # Ensure task table exists
    await asyncio.to_thread(_ensure_table)
    recovery = await recover_processing_tasks_on_startup(stale_minutes=60, max_resume_attempts=2)
    handed_off = int(recovery.get("handed_off", recovery.get("resumed", 0)))
    closed_stale = int(recovery.get("closed_stale", 0))
    closed_repeated = int(recovery.get("closed_repeated", 0))
    if handed_off or closed_stale or closed_repeated:
        _add_system_alert(
            f"Restart recovery: handoff {handed_off} / stale closed {closed_stale} / repeated-failure closed {closed_repeated}"
        )

    global _bot_instance
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    _bot_instance = bot
    dp = Dispatcher()
    dp.include_router(router)

    email_bridge_task = None
    if EMAIL_BRIDGE_ENABLED:
        email_bridge_task = asyncio.create_task(_email_bridge_poll_loop(bot), name="email-bridge-poll")

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
        BotCommand(command="email", description="이메일 현황"),
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
        BotCommand(command="agents", description="에이전트 현황 / 워커 상태"),
        BotCommand(command="clear", description="대화 히스토리 초기화"),
    ])

    # Detect fresh deploy — inject context so the bot knows it was just updated
    await check_deploy_meta(bot, add_alert_fn=_add_system_alert)

    # ── Orchestrator callback: interpret task results for the user ──
    async def _orchestrator_report_task(b: Bot, task: dict, result: dict, chat_id: int):
        """Trigger an orchestrator turn to interpret task results, communicate to user, and redelegate if needed."""
        task_id = task["id"]
        agent_type = task.get("agent_type") or "analyst"
        status = result.get("status", "unknown")
        was_interrupted = result.get("was_interrupted", False)

        try:
            # Load tool_log from DB — contains the actual work done (tool calls + results)
            tool_log = ""
            try:
                row = await asyncio.to_thread(
                    _query_one,
                    "SELECT tool_log FROM telegram_tasks WHERE id = %s", (task_id,),
                )
                tool_log = (row or {}).get("tool_log", "") or ""
            except Exception:
                pass

            # Build context for the orchestrator
            if status == "done":
                report = result.get("report", "")
                interrupted_note = ""
                if was_interrupted:
                    interrupted_note = (
                        "\n\n⚠️ This agent was interrupted due to budget/turn limit. "
                        "Check the agent's response for any incomplete work."
                    )

                # If report is thin but tool_log has substance, include tool_log
                tool_log_section = ""
                if tool_log and (len(report) < 200 or was_interrupted):
                    tool_log_section = f"\n\nAgent work log (tool call history):\n{tool_log[:5000]}"

                prompt = (
                    f"[TASK REPORT] Task #{task_id} [{agent_type}] completed{' (interrupted)' if was_interrupted else ''}\n\n"
                    f"Original request:\n{task.get('content', '')[:1000]}\n\n"
                    f"Execution result:\n{report[:3000]}"
                    f"{tool_log_section}"
                    f"{interrupted_note}\n\n"
                    f"## Your role\n"
                    f"1. Relay the results to the user concisely, covering only key points. Do not use markdown formatting.\n"
                    f"2. Re-delegation judgment: Only delegate follow-up work when ALL of these conditions are met:\n"
                    f"   - The agent could not finish due to budget/turn limits\n"
                    f"   - Additional work can yield meaningful improvement\n"
                    f"   - The cause is NOT external factors (permission denied, blocked, CAPTCHA, API error, etc.)\n"
                    f"   If re-delegation is unnecessary, just relay the results."
                )
            else:
                error = result.get("error", "unknown error")
                prompt = (
                    f"[TASK REPORT] Task #{task_id} [{agent_type}] failed\n\n"
                    f"Original request:\n{task.get('content', '')[:500]}\n\n"
                    f"Error: {error}\n\n"
                    f"Inform the user of the failure and its cause concisely. "
                    f"Do not re-delegate if the issue would not be resolved by retrying."
                )

            # Load recent chat history for context
            history = await asyncio.to_thread(_load_context_with_summaries, chat_id)
            from telegram_commands import sanitize_messages
            history = sanitize_messages(history)
            history.append({"role": "user", "content": prompt})

            # Run orchestrator — budget enough for response + optional redelegate call
            reply = await _chat_with_tools(
                history,
                budget_usd=0.15,
                max_rounds=5,
            )

            # Save orchestrator reply to chat history (system event to separate table)
            await asyncio.to_thread(_save_system_event, chat_id, "task_report", f"task #{task_id} [{agent_type}] {status}")
            await asyncio.to_thread(_save_chat_message, chat_id, "assistant", reply)

            for chunk in _split_message(reply):
                await b.send_message(chat_id=chat_id, text=chunk)

        except Exception as e:
            logger.warning("Orchestrator callback failed for task #%d: %s", task_id, e)
            # Fallback: send simple summary directly
            try:
                if status == "done":
                    fallback = f"Task #{task_id} [{agent_type}] completed: {result.get('summary', '')[:500]}"
                else:
                    fallback = f"Task #{task_id} [{agent_type}] failed: {result.get('error', '')[:300]}"
                await b.send_message(chat_id=chat_id, text=fallback)
            except Exception:
                pass

    # Build process_task closure with module-level dependencies
    async def _process_task_wrapper(b: Bot, task: dict):
        # Set per-coroutine context so tools can identify the running task
        current_task_ctx.set({"task_id": task["id"], "agent_type": task.get("agent_type")})

        from self_tools import build_task_context_tools
        from telegram_tools import TOOLS as BASE_TOOLS, TOOL_HANDLERS as BASE_HANDLERS
        from telegram_tools import build_mission_handler

        # ── Agent-aware task execution ──────────────────────────────
        agent_type = task.get("agent_type") or "analyst"

        # ── Browser task delegation to external worker process ──
        if agent_type == "browser":
            worker_result = await _delegate_to_browser_worker(task)
            if worker_result is not None:
                # Worker handled it — trigger orchestrator callback from main process
                task_id = task["id"]
                user_id = task["user_id"]
                status = worker_result.get("status", "done")
                summary = worker_result.get("result_summary", "")

                icon = "✅" if status == "done" else "❌"
                _add_system_alert(f"{icon} Task #{task_id} {status} (browser worker): {summary[:200]}")

                # Read full result from DB for orchestrator callback
                row = _query_one("SELECT result FROM telegram_tasks WHERE id = %s", (task_id,))
                full_report = (row or {}).get("result", summary)
                orch_result = {
                    "status": status,
                    "task_id": task_id,
                    "summary": summary,
                    "report": full_report,
                    "is_subtask": False,
                    "was_interrupted": False,
                }
                if worker_result.get("error"):
                    orch_result["error"] = worker_result["error"]
                target_uid = user_id if user_id != 0 else OWNER_USER_ID
                if target_uid:
                    await _orchestrator_report_task(b, task, orch_result, target_uid)
                return  # Done — worker handled everything
            # else: worker unreachable, fall through to in-process execution
            logger.info("Browser worker unavailable; executing task #%d in-process", task["id"])

        try:
            from agents import get_agent
            spec = get_agent(agent_type)
        except (ValueError, ImportError):
            from agents import get_agent
            spec = get_agent("analyst")

        # Filter base tools to agent's allowed set
        agent_tools, agent_handlers = spec.filter_tools(BASE_TOOLS, BASE_HANDLERS)

        # Add task-context tools (save_finding)
        ctx_tools, ctx_handlers = build_task_context_tools(
            task["id"], task["user_id"], task.get("depth", 0),
            mission_id=task.get("mission_id"),
        )
        agent_tools.extend(ctx_tools)
        agent_handlers.update(ctx_handlers)

        # Bind mission handler without re-adding schema.
        # MISSION_TOOL is already in BASE_TOOLS via telegram_tools.TOOLS append.
        # Re-appending here duplicates the tool name and breaks API validation.
        if "mission" in {t.get("name") for t in agent_tools}:
            agent_handlers["mission"] = build_mission_handler(task["user_id"])

        # Final safety net against future registry composition mistakes.
        agent_tools = dedupe_tools_by_name(agent_tools)

        # Render agent-specific system prompt
        system_prompt = spec.render_prompt(
            current_datetime=_current_datetime_str(),
            system_alerts=_format_system_alerts(),
            finance_data=_get_finance_context(),
        )

        # Inject runtime environment info for programmer (needs venv, packages, services)
        if agent_type == "programmer":
            system_prompt += "\n" + _build_env_context()

        # Send progress to the task's user (or all users if self-generated)
        target_chat_id = task["user_id"] if task["user_id"] != 0 else OWNER_USER_ID
        progress_cb = _make_progress_callback(target_chat_id) if target_chat_id else None

        # ── Provider dispatch: Claude vs local LLM (OpenAI-compatible) ──
        if spec.provider == "moon":
            from openai_tool_loop import chat_with_tools as moon_chat_with_tools
            from llm_client import MOON_BASE, MOON_MODEL, LOCAL_BASE, LOCAL_MODEL, _health_ok

            # Try MOON PC first, then local llama-server, then Claude
            if _health_ok(MOON_BASE):
                llm_base, llm_model = MOON_BASE, MOON_MODEL
                logger.info("Agent %s: using MOON PC (%s)", spec.name, MOON_BASE)
            elif _health_ok(LOCAL_BASE):
                llm_base, llm_model = LOCAL_BASE, LOCAL_MODEL
                logger.info("Agent %s: MOON unavailable, using local LLM (%s)", spec.name, LOCAL_BASE)
            else:
                llm_base = None
                logger.warning("MOON PC and local LLM both unavailable for agent %s; falling back to Claude", spec.name)

            if llm_base is None:
                chosen_chat_fn = _chat_with_tools
                chosen_model_fn = _get_model_task
                chosen_max_tokens = _CLAUDE_MAX_TOKENS_TASK
            else:
                _llm_base, _llm_model = llm_base, llm_model  # capture for closure

                async def _moon_chat_with_tools(
                    messages, max_rounds=None, system_prompt=None, model=None,
                    max_tokens=None, budget_usd=None, extra_tools=None,
                    extra_handlers=None, on_progress=None, budget_tracker=None,
                    task_id=None,
                ):
                    # extra_tools already contains the agent's filtered tools (passed from process_task)
                    merged_tools = list(extra_tools or [])
                    merged_handlers = dict(extra_handlers or {})
                    return await moon_chat_with_tools(
                        messages,
                        base_url=_llm_base,
                        model=model or _llm_model,
                        tools=merged_tools,
                        tool_handlers=merged_handlers,
                        system_prompt=system_prompt,
                        max_rounds=max_rounds or spec.max_rounds,
                        max_tokens=max_tokens or 8192,
                        log_event=_log_event,
                        budget_usd=budget_usd or 0.0,
                        budget_tracker=budget_tracker,
                        on_progress=on_progress,
                        task_id=task_id,
                    )
                chosen_chat_fn = _moon_chat_with_tools
                chosen_model_fn = _get_model_moon
                chosen_max_tokens = 8192
        else:
            chosen_chat_fn = _chat_with_tools
            chosen_model_fn = _get_model_task
            chosen_max_tokens = _CLAUDE_MAX_TOKENS_TASK

        def _on_task_complete(task_id: int, status: str, summary: str, **_kw):
            icon = "✅" if status == "done" else "❌"
            _add_system_alert(f"{icon} Task #{task_id} {status}: {summary[:200]}")

        result = await process_task(
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

        # ── Orchestrator callback: report result to user via orchestrator ──
        result = result or {}
        is_subtask = result.get("is_subtask", False)
        if not is_subtask and result.get("status") in ("done", "failed"):
            target_uid = task["user_id"] if task["user_id"] != 0 else OWNER_USER_ID
            if target_uid:
                await _orchestrator_report_task(b, task, result, target_uid)

    # Start background workers (keep handles for graceful cancellation)
    _bg_tasks = [
        asyncio.create_task(
            task_worker(bot, process_task_fn=_process_task_wrapper, runtime_state=_runtime_state, max_concurrency=_config.get("task_concurrency", 2)),
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
    if email_bridge_task is not None:
        _bg_tasks.insert(0, email_bridge_task)

    # Graceful shutdown: notify + stop polling cleanly when SIGTERM received (Render deploy)
    import signal

    def _handle_sigterm(*_):
        logger.info("SIGTERM received — stopping polling gracefully")
        # Schedule shutdown notification before stopping
        async def _shutdown_notify_and_checkpoint():
            # Merge in-memory and Redis active task sets for comprehensive checkpoint
            active_ids = set(_runtime_state.get("active_task_ids", set()))
            try:
                from redis_state import get_active_task_ids
                active_ids |= get_active_task_ids()
            except Exception:
                pass
            for task_id in active_ids:
                try:
                    ok = await checkpoint_task_on_shutdown(int(task_id))
                    if ok:
                        logger.info("Shutdown checkpoint saved for in-flight task #%s", task_id)
                except Exception as e:
                    logger.warning("Shutdown checkpoint failed for task #%s: %s", task_id, e)
            # Save restart marker to chat history so the bot retains awareness after restart
            restart_ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
            if OWNER_USER_ID:
                try:
                    await asyncio.to_thread(
                        _save_system_event, OWNER_USER_ID, "restart",
                        f"SIGTERM received, service restart initiated ({restart_ts})"
                    )
                except Exception:
                    pass
            if OWNER_USER_ID:
                try:
                    await bot.send_message(chat_id=OWNER_USER_ID, text="🔄 *서버 재시작 중* — 새 버전 배포가 시작됩니다.")
                except Exception:
                    pass
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
        _add_system_alert("Deploy complete — Telegram service running")
        # Save startup marker to chat history so the bot knows it just restarted
        startup_ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
        recovery_summary = ""
        if handed_off or closed_stale or closed_repeated:
            recovery_summary = f" Task recovery: handoff {handed_off}, expired {closed_stale}, repeated-failure {closed_repeated}."
        if OWNER_USER_ID:
            try:
                _save_system_event(
                    OWNER_USER_ID, "startup",
                    f"Telegram service restart complete ({startup_ts}).{recovery_summary}"
                )
            except Exception as e:
                logger.warning("Failed to save startup marker for owner: %s", e)
            try:
                await bot.send_message(chat_id=OWNER_USER_ID, text="🟢 Telegram 서비스 재시작 완료 — 메시지 수신 준비 완료.")
            except Exception:
                pass

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
