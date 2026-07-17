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
import re
import warnings
from datetime import datetime
from pathlib import Path
from identity.prompts import CORE_IDENTITY, EXTERNAL_SOURCE_RULE
from shared import KST
from agents.base import CHAT_AUDIENCE_BLOCK, load_political_line_body
from skills_loader import build_skills_prompt
from db import query as _query, execute as _execute, query_one as _query_one, get_conn as _get_conn
from psycopg2.extras import RealDictCursor

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import CallbackQuery, ChatMemberUpdated, Message
from aiogram.utils.backoff import BackoffConfig

from secrets_loader import get_secret

# Extracted modules
from bot_config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    _claude, _openai_client, _deepseek_client, _deepseek_anthropic_client, _kimi_client,
    _CLAUDE_MAX_TOKENS, _CLAUDE_MAX_TOKENS_TASK,
    _config, _save_config, _CONFIG_DEFAULTS, _CONFIG_META,
    _resolved_models, _tier_to_display,
    _get_model, _get_model_task, _get_model_light, _get_model_moon,
    _get_task_provider,
    get_current_model_selection, get_task_verification_mode,
    _extract_text,
)
from runtime_profile import resolve_runtime_profile
from telegram.schema import hydrate_summary_state
from runtime_tools.allowlists import select_orchestrator_tools
from runtime_tools.registry import TOOLS, TOOL_HANDLERS
from claude_loop import chat_with_tools, dedupe_tools_by_name
from telegram.bot_api10 import TelegramBotApi10Client, TelegramBotApiError
from telegram.tasks import (
    process_task, system_monitor,
    task_worker, schedule_worker, check_deploy_meta,
    recover_processing_tasks_on_startup,
    checkpoint_task_on_shutdown, persist_task_restart_state,
    _delegate_to_browser_worker, check_browser_worker_alive,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

_runtime_state: dict = {"active_task_ids": set()}
DEEPSEEK_CONTEXT_LIMIT = int(os.getenv("DEEPSEEK_CONTEXT_LIMIT", "1000000"))

# Per-coroutine task context — allows concurrent tasks to know their own task_id
import contextvars
current_task_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_task_ctx", default=None)

# Suppress TelegramConflictError spam during deploy (old/new instance overlap)
class _ConflictFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "TelegramConflictError" not in record.getMessage()

logging.getLogger("aiogram.dispatcher").addFilter(_ConflictFilter())
logging.getLogger("aiogram.event").addFilter(_ConflictFilter())
warnings.filterwarnings(
    "ignore",
    message=r"Detected unknown update type\.",
    category=RuntimeWarning,
    module=r"aiogram\.dispatcher\.event\.handler",
)

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
TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", "") or ""
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}
# Single-owner enforcement: all outbound messages go to this user only.
OWNER_USER_ID: int = next(iter(ALLOWED_USER_IDS)) if len(ALLOWED_USER_IDS) == 1 else 0
EMAIL_BRIDGE_ENABLED = os.getenv("EMAIL_BRIDGE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
EMAIL_POLLING_ENABLED = os.getenv("EMAIL_POLLING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
EMAIL_POLL_INTERVAL_SECONDS = max(30, int(os.getenv("EMAIL_POLL_INTERVAL_SECONDS", "120")))
EMAIL_APPROVAL_BASE_URL = os.getenv("EMAIL_APPROVAL_BASE_URL", "").rstrip("/")
EMAIL_DEFAULT_APPROVER_USER_ID = int(os.getenv("EMAIL_DEFAULT_APPROVER_USER_ID", "0") or "0")
EMAIL_LOG_DIR = Path(os.getenv("EMAIL_LOG_DIR", str(Path(__file__).resolve().parent.parent / "logs" / "email_bridge")))
PUBLIC_ACCESS_NOTICE = (
    "이 계정은 Cyber-Lenin 운영 인터페이스입니다.\n\n"
    "공개 글은 https://cyber-lenin.com 에서 볼 수 있고, "
    "텔레그램 채널은 https://t.me/cyber_lenin_kr 입니다."
)
PUBLIC_ACCESS_NOTICE_COOLDOWN_SECONDS = max(
    0,
    int(os.getenv("PUBLIC_ACCESS_NOTICE_COOLDOWN_SECONDS", "86400") or "0"),
)
CHAT_PERSIST_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_CHAT_PERSIST_TIMEOUT_SECONDS", "15"))
TELEGRAM_GUEST_MODE_POLICY = os.getenv("TELEGRAM_GUEST_MODE_POLICY", "owner_only").strip().lower() or "owner_only"
TELEGRAM_POLLING_TIMEOUT_SECONDS = max(10, int(os.getenv("TELEGRAM_POLLING_TIMEOUT_SECONDS", "30") or "30"))
TELEGRAM_SESSION_TIMEOUT_SECONDS = max(
    TELEGRAM_POLLING_TIMEOUT_SECONDS + 15,
    int(os.getenv("TELEGRAM_SESSION_TIMEOUT_SECONDS", "75") or "75"),
)
TELEGRAM_POLLING_CONCURRENCY_LIMIT = max(1, int(os.getenv("TELEGRAM_POLLING_CONCURRENCY_LIMIT", "8") or "8"))
TELEGRAM_BACKOFF_MIN_SECONDS = max(0.1, float(os.getenv("TELEGRAM_BACKOFF_MIN_SECONDS", "1.0") or "1.0"))
TELEGRAM_BACKOFF_MAX_SECONDS = max(
    TELEGRAM_BACKOFF_MIN_SECONDS,
    float(os.getenv("TELEGRAM_BACKOFF_MAX_SECONDS", "30.0") or "30.0"),
)
TELEGRAM_BACKOFF_FACTOR = max(1.0, float(os.getenv("TELEGRAM_BACKOFF_FACTOR", "1.7") or "1.7"))
TELEGRAM_BACKOFF_JITTER = max(0.0, float(os.getenv("TELEGRAM_BACKOFF_JITTER", "0.2") or "0.2"))
TELEGRAM_CONNECTIVITY_WATCHDOG_SECONDS = max(
    0,
    int(os.getenv("TELEGRAM_CONNECTIVITY_WATCHDOG_SECONDS", "60") or "60"),
)
TELEGRAM_CONNECTIVITY_PROBE_TIMEOUT_SECONDS = max(
    3,
    int(os.getenv("TELEGRAM_CONNECTIVITY_PROBE_TIMEOUT_SECONDS", "10") or "10"),
)
TELEGRAM_CONNECTIVITY_NOTIFY_AFTER_FAILURES = max(
    1,
    int(os.getenv("TELEGRAM_CONNECTIVITY_NOTIFY_AFTER_FAILURES", "3") or "3"),
)
DEFAULT_ALLOWED_UPDATES = [
    "message",
    "edited_message",
    "callback_query",
    "my_chat_member",
    "chat_member",
    "message_reaction",
    "message_reaction_count",
    "guest_message",
]
_public_access_notice_last: dict[int, float] = {}
_GROUP_CHAT_TYPES = {"group", "supergroup"}


def _telegram_user_id(event) -> int | None:
    user = getattr(event, "from_user", None)
    uid = getattr(user, "id", None)
    return int(uid) if uid is not None else None


def _telegram_username(event) -> str:
    user = getattr(event, "from_user", None)
    return str(getattr(user, "username", "") or "")


def _telegram_chat(event):
    chat = getattr(event, "chat", None)
    if chat is not None:
        return chat
    message = getattr(event, "message", None)
    return getattr(message, "chat", None)


async def _maybe_reply_public_access_notice(message: Message, user_id: int | None) -> None:
    if user_id is None:
        return
    chat_type = str(getattr(message.chat, "type", "") or "")
    if chat_type != "private":
        return

    import time

    now = time.monotonic()
    last = _public_access_notice_last.get(user_id, 0.0)
    if PUBLIC_ACCESS_NOTICE_COOLDOWN_SECONDS and now - last < PUBLIC_ACCESS_NOTICE_COOLDOWN_SECONDS:
        return
    _public_access_notice_last[user_id] = now
    await message.answer(PUBLIC_ACCESS_NOTICE, disable_web_page_preview=True)


async def _maybe_leave_unauthorized_group(event, data: dict, user_id: int | None) -> bool:
    chat = _telegram_chat(event)
    chat_type = str(getattr(chat, "type", "") or "")
    if chat_type not in _GROUP_CHAT_TYPES:
        return False

    bot = data.get("bot") or getattr(event, "bot", None)
    if bot is None:
        logger.warning(
            "cannot leave unauthorized Telegram group without bot instance chat_id=%s user_id=%s",
            getattr(chat, "id", None),
            user_id,
        )
        return True

    try:
        await bot.leave_chat(getattr(chat, "id"))
        logger.info(
            "left unauthorized Telegram group chat_id=%s title=%s user_id=%s username=%s",
            getattr(chat, "id", None),
            getattr(chat, "title", ""),
            user_id,
            _telegram_username(event),
        )
    except Exception as e:
        logger.warning(
            "failed to leave unauthorized Telegram group chat_id=%s user_id=%s: %s",
            getattr(chat, "id", None),
            user_id,
            e,
        )
    return True


class OwnerOnlyMiddleware(BaseMiddleware):
    """Stop non-owner Telegram events before command, LLM, or tool handlers run."""

    async def __call__(self, handler, event, data):
        user_id = _telegram_user_id(event)
        if user_id is not None and _is_allowed(user_id):
            return await handler(event, data)

        if isinstance(event, Message):
            if await _maybe_leave_unauthorized_group(event, data, user_id):
                return None
            await _maybe_reply_public_access_notice(event, user_id)
            logger.info(
                "blocked unauthorized Telegram message user_id=%s username=%s chat_id=%s chat_type=%s",
                user_id,
                _telegram_username(event),
                getattr(event.chat, "id", None),
                getattr(event.chat, "type", None),
            )
            return None

        if isinstance(event, ChatMemberUpdated):
            if await _maybe_leave_unauthorized_group(event, data, user_id):
                return None
            logger.info(
                "blocked unauthorized Telegram chat member update user_id=%s username=%s chat_id=%s chat_type=%s",
                user_id,
                _telegram_username(event),
                getattr(event.chat, "id", None),
                getattr(event.chat, "type", None),
            )
            return None

        if isinstance(event, CallbackQuery):
            if await _maybe_leave_unauthorized_group(event, data, user_id):
                return None
            logger.info(
                "blocked unauthorized Telegram callback user_id=%s username=%s",
                user_id,
                _telegram_username(event),
            )
            try:
                await event.answer()
            except Exception:
                pass
            return None

        logger.info("blocked unauthorized Telegram event type=%s user_id=%s", type(event).__name__, user_id)
        return None


async def _ignore_chat_member_update(event: ChatMemberUpdated):
    return None


def _allowed_updates() -> list[str]:
    raw = os.getenv("TELEGRAM_ALLOWED_UPDATES", "").strip()
    if not raw:
        return list(DEFAULT_ALLOWED_UPDATES)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or list(DEFAULT_ALLOWED_UPDATES)


def _has_guest_message(update) -> bool:
    return bool(getattr(update, "guest_message", None))


def _guest_message_dict(update) -> dict:
    value = getattr(update, "guest_message", None)
    return value if isinstance(value, dict) else {}


def _guest_message_user_id(guest_message: dict) -> int | None:
    user = guest_message.get("from") or guest_message.get("from_user") or guest_message.get("guest_bot_caller_user")
    if isinstance(user, dict) and user.get("id") is not None:
        try:
            return int(user["id"])
        except (TypeError, ValueError):
            return None
    return None


async def _handle_guest_update(update, bot: Bot):
    guest_message = _guest_message_dict(update)
    guest_query_id = str(guest_message.get("guest_query_id") or "").strip()
    caller_user_id = _guest_message_user_id(guest_message)
    chat = guest_message.get("chat") if isinstance(guest_message.get("chat"), dict) else {}
    chat_id = chat.get("id") if isinstance(chat, dict) else None

    logger.info(
        "blocked Telegram guest_message update_id=%s caller_user_id=%s chat_id=%s policy=%s",
        getattr(update, "update_id", None),
        caller_user_id,
        chat_id,
        TELEGRAM_GUEST_MODE_POLICY,
    )
    if OWNER_USER_ID:
        try:
            await asyncio.to_thread(
                _save_system_event,
                OWNER_USER_ID,
                "guest_message_blocked",
                f"Blocked guest_message update={getattr(update, 'update_id', None)} "
                f"caller_user_id={caller_user_id} chat_id={chat_id}",
            )
        except Exception as e:
            logger.warning("failed to persist guest_message block event: %s", e)

    if not guest_query_id:
        return None

    notice = (
        "Cyber-Lenin은 비공개 운영 봇입니다. 공개 글은 https://cyber-lenin.com 에서 볼 수 있고, "
        "텔레그램 채널은 https://t.me/cyber_lenin_kr 입니다."
    )
    try:
        client = TelegramBotApi10Client(token=TELEGRAM_BOT_TOKEN)
        await client.answer_guest_query_text(guest_query_id, notice)
    except TelegramBotApiError as e:
        logger.warning("answerGuestQuery failed for guest_message update_id=%s: %s", getattr(update, "update_id", None), e)
    except Exception as e:
        logger.warning("unexpected answerGuestQuery failure update_id=%s: %s", getattr(update, "update_id", None), e)
    return None


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


def _format_current_model_context(kind: str = "chat", provider: str = "claude") -> str:
    """Format runtime-selected model info for prompt/context injection.

    Leads with the human-readable product name ("Claude Opus 4.8", "GPT-5.5 Pro")
    so self-identification works cleanly; the raw API id and tier stay available
    as secondary metadata. `provider` controls BOTH the surface form (XML for
    Claude, Markdown elsewhere) AND which tier map the model is resolved from —
    so an agent pinned to Claude while config.provider="openai" still surfaces
    the real Claude model it's running on, not the chat-side GPT.
    """
    sel = get_current_model_selection(kind, provider_override=provider)
    name = sel["display_name"]
    model_id = sel["model_id"]
    tier = sel["tier"]
    if provider == "claude":
        return (
            f"<current-model tier=\"{tier}\" id=\"{model_id}\">{name}</current-model>"
        )
    return f"- **Current Model**: {name} (id: `{model_id}`, tier: {tier})"


def _build_runtime_prelude(provider: str = "claude", kind: str = "chat") -> str:
    """Render the volatile runtime header (time + active model).

    Goes at the top of extra context so the system prompt itself stays
    byte-identical across turns (prompt-cache friendly). Returned without any
    leading/trailing whitespace — separator insertion is the caller's job
    (`_join_context_blocks`).
    """
    current_time = _current_datetime_str()
    current_model = _format_current_model_context(kind, provider)
    if provider == "claude":
        return (
            f"<runtime>\n<current-time>{current_time}</current-time>\n"
            f"{current_model}\n</runtime>"
        )
    return (
        f"### Runtime\n"
        f"- **Current Time**: {current_time}\n"
        f"{current_model}"
    )


def _join_context_blocks(*blocks: str) -> str:
    """Concatenate non-empty context blocks with a blank-line separator.

    Every block (XML tag group or Markdown section) gets bounded by an actual
    blank line in the output, which both CommonMark/GFM parsers and LLM
    attention treat as a real section break. Empty or whitespace-only blocks
    are skipped, so callers can unconditionally pass optional context slots.
    """
    cleaned = [b.strip() for b in blocks if b and b.strip()]
    return "\n\n".join(cleaned)


def _merge_runtime_context_into_last_user(
    messages: list[dict], runtime_context: str
) -> list[dict]:
    """Fold per-turn runtime context into the trailing user message content.

    Placing volatile context (time, mission, alerts, …) immediately before the
    current user query — rather than at the start of the message array — keeps
    the history prefix byte-stable across turns so prompt caching (Claude
    ephemeral / OpenAI automatic) keeps hitting. Returns a new list; the caller's
    list and its inner dicts are left untouched.
    """
    if not runtime_context or not runtime_context.strip():
        return list(messages)

    ctx = runtime_context.strip()
    result = list(messages)
    if result and result[-1].get("role") == "user":
        last = dict(result[-1])
        existing = last.get("content", "")
        if isinstance(existing, str):
            last["content"] = f"{ctx}\n\n{existing}".strip() if existing else ctx
        elif isinstance(existing, list):
            last["content"] = [{"type": "text", "text": ctx}] + list(existing)
        else:
            last["content"] = f"{ctx}\n\n{existing}".strip()
        result[-1] = last
    else:
        result.append({"role": "user", "content": ctx})
    return result


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


def _format_system_alerts(provider: str = "claude") -> str:
    """Format recent system alerts as a standalone context block.

    Returned without any surrounding whitespace — `_join_context_blocks` is
    responsible for separator insertion. `provider` selects the structure:
    Claude → `<system-alerts>` XML, others → `### System Alerts` Markdown.
    """
    _prune_alerts()
    if not _system_alerts:
        return ""
    items = "\n".join(f"- {m}" for _, m in _system_alerts)
    if provider == "claude":
        return f"<system-alerts>\n{items}\n</system-alerts>"
    return f"### System Alerts\n{items}"


async def _telegram_connectivity_watchdog(bot: Bot) -> None:
    """Probe Telegram periodically so network stalls become visible."""
    if TELEGRAM_CONNECTIVITY_WATCHDOG_SECONDS <= 0:
        return

    notified_down = False
    failure_count = 0
    while True:
        await asyncio.sleep(TELEGRAM_CONNECTIVITY_WATCHDOG_SECONDS)
        try:
            await asyncio.wait_for(bot.get_me(), timeout=TELEGRAM_CONNECTIVITY_PROBE_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            failure_count += 1
            if failure_count < TELEGRAM_CONNECTIVITY_NOTIFY_AFTER_FAILURES:
                logger.info(
                    "Telegram connectivity probe failed (%d/%d): %s: %s",
                    failure_count,
                    TELEGRAM_CONNECTIVITY_NOTIFY_AFTER_FAILURES,
                    type(e).__name__,
                    e,
                )
                continue

            if not notified_down:
                _add_system_alert(f"Telegram connectivity degraded: {type(e).__name__}: {e}")
                logger.warning(
                    "Telegram connectivity probe failed after %d consecutive probes: %s: %s",
                    failure_count,
                    type(e).__name__,
                    e,
                )
                if OWNER_USER_ID:
                    try:
                        await asyncio.to_thread(
                            _save_system_event,
                            OWNER_USER_ID,
                            "telegram_connectivity_down",
                            f"Telegram connectivity probe failed: {type(e).__name__}: {e}",
                        )
                    except Exception:
                        pass
            elif failure_count % 5 == 0:
                logger.warning("Telegram connectivity still degraded after %d probes: %s", failure_count, e)
            notified_down = True
            continue

        if notified_down:
            _clear_system_alert("Telegram connectivity degraded")
            logger.info("Telegram connectivity restored after %d failed probes", failure_count)
            if OWNER_USER_ID:
                try:
                    await asyncio.to_thread(
                        _save_system_event,
                        OWNER_USER_ID,
                        "telegram_connectivity_restored",
                        f"Telegram connectivity restored after {failure_count} failed probes",
                    )
                    await bot.send_message(
                        chat_id=OWNER_USER_ID,
                        text=f"🟢 Telegram 연결 복구 — 실패 probe {failure_count}회 후 정상화.",
                    )
                except Exception:
                    pass
        elif failure_count:
            logger.info("Telegram connectivity probe recovered after %d transient failures", failure_count)
        notified_down = False
        failure_count = 0



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


# ── Orchestrator system prompt (semantic IR → provider-specific render) ──
# The IR holds the static structure; runtime values (current time / model /
# alerts / skills) are spliced in via a provider-specific dynamic tail below.
# Claude gets XML tags; OpenAI/Qwen get Markdown headers — both deliberate,
# matching the format each family is trained on.

from llm.prompt_renderer import SystemPrompt, render as _render_prompt

_CHAT_AUDIENCE_INNER = (
    CHAT_AUDIENCE_BLOCK
    .removeprefix("<chat-audience>").removesuffix("</chat-audience>")
    .strip()
)

_ORCHESTRATOR_PROMPT_IR = SystemPrompt(
    identity=CORE_IDENTITY.rstrip() + "\n\n" + EXTERNAL_SOURCE_RULE,
    preamble=(
        "Operating via Telegram (you are currently talking to the admin "
        "비숑 동지). Use tools proactively when data would improve the "
        "answer — don't rely on memory alone."
    ),
    sections=[
        ("chat-audience", _CHAT_AUDIENCE_INNER),
        ("tool-strategy", """
- Geopolitics → knowledge_graph_search first, then vector_search
- For Korean organizations/publications already known to KG, preserve canonical names; do not invent translations/romanizations. Use `디아마트 (DiaMat)` and `웹진 반란(Uprising)`, not `Diamat` or `Webzine Banlan`.
- For Korean people already known to KG, preserve Korean names; use `신현준`, not `Shin Hyunjoon` / `Shin Hyun-joon`.
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page; for x.com/twitter.com status/profile URLs use fetch_x_post
- Self-reflection → read_self(content_type="diary"); cross-interface memory → read_self(content_type="chat_logs")
- Past lessons/mistakes → recall_experience (semantic search over accumulated daily insights)
- Reusable self-produced analysis → save_self_analysis, then retrieve later with vector_search(layer="self_produced_analysis")
- Store important structured facts → write_kg_structured
- Real-time market prices → get_finance_data
- Telegram channel announcement → broadcast_to_channel(title, summary, url). Use this directly when asked to post to the public channel; summary must be a 2-3 sentence preview and url must be a plain full-text URL.
- Unsure which agent or content store owns the request → route_task(task="...") first. Use list_agent_tools only when you need the detailed tool list/schema.
- Published content corrections → delegate to the content-owning agent, not programmer:
  - Past diary entry → `delegate(agent="diary")`; the diary agent has `edit_content(content_type="diary", ...)`
  - Published research / task report / blog post / hub curation / static page → `delegate(agent="analyst")`; the analyst has `research_document` and `edit_content`
  These are operational content edits with cache invalidation, not code changes. Do not delegate to programmer just to correct wording, titles, metadata, markdown prose, or factual text in existing public content.
- Runtime/pipeline failures → delegate to programmer even if the affected feature is diary or publishing. "Diary writing failed", diary scheduler/context bugs, route_task errors, service errors, and pipeline regressions are code/config work, not diary content edits.
""".strip()),
        ("context-isolation", """
**You are the orchestrator. You have no access to programming tools (read_file, write_file, patch_file, list_directory, execute_python).**
If you need to read/modify/execute code, you must delegate via `delegate(agent="programmer")`.
This code-delegation rule does NOT apply to already-published site content. If the user asks to edit public text/content that is stored in the database or research store, delegate to the content-owning agent (usually diary for diaries, analyst for research/reports/posts/curations), not programmer.
Your role is to understand the user's intent, dispatch tasks to the appropriate agents, and synthesize results.
The `<current_state>` block contains structured completed/in-progress/pending tasks. Use it to avoid duplicate work and determine next steps. Detailed tool execution logs are only accessible to each agent itself.
""".strip()),
        ("delegation", """
CRITICAL RULE: When you decide to delegate, you MUST call the `delegate` or `multi_delegate` tool.

You have specialized agents. Use the `delegate` tool to dispatch tasks:
- programmer: code writing/editing/debugging/file management
- analyst: default agent for information analysis/research. Web search + collection + KG cross-validation + pattern extraction + knowledge storage
- scout: Moltbook and mersoom.com activity (posting/commenting/patrol), routine patrols, large-scale platform crawling
- browser: AI browser automation — login, form input, multi-page navigation, dynamic site data extraction
- visualizer: image generation, visual concepts
- diary: writes scheduled diary entries and maintains published diary entries. For a new diary entry, delegate exactly `[diary] Write a periodic diary entry`; arbitrary diary-agent task text is treated as autonomous maintenance/inspection and cannot call `save_diary`.
Stasova is not a general delegation target. It is reserved for internal publication-review flows and is intentionally absent from delegate/multi_delegate.

Content-type routing:
- Diary entry → diary for writing/editing/deleting/unpublishing; use diary id when supplied. For new diary writing, use the exact scheduled prompt `[diary] Write a periodic diary entry`.
- Task report / completed Telegram report → analyst with edit_content(content_type="task_report"); use task_id when supplied
- Public research document → analyst with research_document; use research slug when supplied
- Private research document → analyst with research_document(action="save_private"|"publish_private"); read it with read_self(content_type="private_research_document")
- Hub curation → analyst with edit_content(content_type="hub_curation"); use curation slug when supplied
- Static/custom HTML page → analyst with edit_content(content_type="static_page"); use static page slug when supplied
Public URLs are only clues for inferring content type and identifier; do not make agents reason from routes when the content type is already known.

Parallel delegation with `multi_delegate`:
- Compound requests (e.g., "investigate X and fix Y's code") should be handled in parallel via multi_delegate.
- After all subtasks complete, a synthesis task automatically consolidates results.
- Specifying synthesis_instructions with consolidation criteria yields better results.

Context passing — agents automatically receive recent conversation and their own execution history, but specify the current conversation's key context in the `context` field:
1. The user's original request (verbatim or key summary)
2. Findings from the conversation so far (tool results, analysis, decisions)
3. Why you are delegating to this agent (reason and expected outcome)
4. The correct target identifier when the user supplied one: public URL, slug, post_id, DB document identifier, error text, command output, or visible symptom. Do not invent or pass filesystem paths; delegated agents that need code context can inspect the repository themselves.

Delegation discipline: delegate what must be achieved, not how. Do not invent unverified implementation details; let workers inspect and choose the implementation.

Do not delegate routine public-content edits to programmer. For requests like "fix this published post", "correct a diary/report/blog typo", "revise this curation", or "edit an already-published research page", delegate to the agent that owns the content/editor tool: diary for diary entries; analyst for research documents, task reports, blog posts, and curations. Delegate to programmer only when the required change is source code, configuration, scripts, templates, frontend behavior, deployment, or debugging.
If the user explicitly says to send a failure, scheduler, routing, or pipeline problem to programmer, obey that instruction; do not reinterpret it as a diary content task just because the word "diary" appears.
""".strip()),
        ("mission-management", """
- Missions are auto-created when delegate is called. The user does not need to create them explicitly.
- The `<active-mission>` block (when present) shows the current mission's title and event timeline. Read it before every turn to decide whether the user's new message still belongs to that mission.
- **Call `mission(action="close")` when ANY of these hold:**
  - The user's original goal for the active mission has been fully achieved.
  - **Topic drift**: the user's current message is clearly on a different topic from the active mission's title/timeline. A topic switch implicitly abandons the old mission — close it so the next `delegate` call opens a fresh mission aligned with the new topic. Do this even when the prior mission had in-progress or budget-interrupted tasks; stale missions pollute future context.
- **Do NOT close solely because** task results mention "budget exhausted" / "limit reached" / "stopped due to error" — if the user is still pursuing that same topic, leave the mission open and delegate follow-up work.
- When you close a mission because of topic drift, you don't need to manually create a replacement — just proceed with the new topic; if a delegate is needed, a fresh mission will be auto-created from it.
""".strip()),
        ("temporal-awareness", """
Conversation history includes timestamps ([YYYY-MM-DD HH:MM]) on user messages.
Infer elapsed time from the timestamps. Large gaps may indicate context switches or changed circumstances.
""".strip()),
        ("response-rules", """
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
- Do not use markdown formatting (**, *, #, ```, - etc.) in Telegram messages. Write in plain text only, as a human would. Markdown is allowed only when composing markdown documents or code artifacts through the appropriate specialist/tool.
""".strip()),
    ],
)


# Fully static system-layer tail. Only content that does not change between
# turns belongs here (skills catalog). Per-turn runtime state — current time,
# current model, mission, memories, alerts — is injected via message content,
# not the system prompt, so prompt caching stays effective.
_CLAUDE_STATIC_TAIL = "{skills_section}"
_MARKDOWN_STATIC_TAIL = "{skills_section}"


def _format_autonomous_status(provider: str = "claude") -> str:
    """One-line-per-project summary of active autonomous projects.

    Surfaces the existence of the self-running project loop to the orchestrator
    so it can reference ongoing work without the user having to prompt for it.
    Returns empty string if no active projects or on DB error (fail-safe —
    chat must never break because this auxiliary block can't be built).
    """
    try:
        from bot_config import is_autonomous_active
        loop_active = is_autonomous_active()
    except Exception:
        loop_active = True
    try:
        from db import query as db_query
        rows = db_query(
            """
            SELECT p.id, p.title, p.state, p.turn_count, p.last_run_at,
                   COALESCE(a.pending_advisories, 0) AS pending_advisories,
                   e.event_type AS last_event_type,
                   e.created_at AS last_event_at
              FROM autonomous_projects p
              LEFT JOIN LATERAL (
                  SELECT COUNT(*)::int AS pending_advisories
                    FROM autonomous_project_advisories adv
                   WHERE adv.project_id = p.id
                     AND adv.consumed_at IS NULL
              ) a ON TRUE
              LEFT JOIN LATERAL (
                  SELECT event_type, created_at
                    FROM autonomous_project_events ev
                   WHERE ev.project_id = p.id
                   ORDER BY ev.created_at DESC, ev.id DESC
                   LIMIT 1
              ) e ON TRUE
             WHERE p.state IN ('researching', 'planning')
             ORDER BY p.id
            """
        )
    except Exception:
        return ""
    if not rows:
        return ""
    lines = []
    for r in rows:
        last = r["last_run_at"].astimezone(KST).strftime("%m/%d %H:%M KST") if r.get("last_run_at") else "never"
        title = str(r.get("title") or "").replace("\n", " ")[:80]
        bits = [f"{r['state']}", f"turn {r['turn_count']}", f"last ran {last}"]
        pending = int(r.get("pending_advisories") or 0)
        if pending:
            bits.append(f"pending advice {pending}")
        if r.get("last_event_type"):
            event_at = r["last_event_at"].astimezone(KST).strftime("%m/%d %H:%M KST") if r.get("last_event_at") else "?"
            bits.append(f"last event {r['last_event_type']} @ {event_at}")
        lines.append(f"- #{r['id']} \"{title}\" — " + ", ".join(bits))
    loop_line = (
        "Loop is enabled; hourly timer can advance due projects."
        if loop_active
        else "Loop is PAUSED by config (autonomous_active=false); timer wakes skip run_tick until re-enabled."
    )
    body = (
        "Self-running long-term project loop (hourly tick, separate from your chat turn). "
        + loop_line
        + "\nActive projects:\n"
        + "\n".join(lines)
        + "\nFor detail on any project call read_self(content_type=\"autonomous_project\", id=<id>)."
    )
    if provider == "claude":
        return f"<autonomous-agent-status>\n{body}\n</autonomous-agent-status>"
    return "### Autonomous Agent Status\n" + body


def _build_orchestrator_system_prompt(provider: str) -> str:
    """Render the orchestrator's static system prompt for `provider`.

    Returned string keeps only the ``{skills_section}`` placeholder — the only
    stable runtime data in the system layer. Per-turn state (time, model,
    mission, memories, alerts) is injected as message content so the system
    prompt stays byte-identical across turns and benefits from prompt caching.
    """
    prompt_ir = _ORCHESTRATOR_PROMPT_IR
    political_line = load_political_line_body()
    if political_line:
        prompt_ir = SystemPrompt(
            identity=_ORCHESTRATOR_PROMPT_IR.identity,
            preamble=_ORCHESTRATOR_PROMPT_IR.preamble,
            sections=[("political-line", political_line), *_ORCHESTRATOR_PROMPT_IR.sections],
            context=_ORCHESTRATOR_PROMPT_IR.context,
        )
    body = _render_prompt(prompt_ir, provider)
    tail = _CLAUDE_STATIC_TAIL if provider == "claude" else _MARKDOWN_STATIC_TAIL
    return body + "\n\n" + tail


# ── Chat History ─────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10  # 10 pairs = 20 messages


# Per-user clear marker: messages with id <= this value are ignored
_clear_after_id: dict[int, int] = {}
_summary_state_hydrated = False



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


def _truncate_for_prompt(text: str, limit: int) -> str:
    """Slice `text` to `limit` chars and append an explicit truncation marker.

    The marker tells the orchestrator that cropping is display-only — the
    agent's underlying output is complete. Without this, the orchestrator
    (observed with opus 4.7) mistakes a mid-sentence cut for the agent
    having run out of budget and triggers spurious re-delegation.
    """
    if not text or len(text) <= limit:
        return text or ""
    omitted = len(text) - limit
    return (
        text[:limit]
        + f"\n\n[⚠ TRUNCATED FOR PROMPT DISPLAY: showing {limit} of {len(text)} chars "
        f"({omitted} omitted). The agent's output is COMPLETE and stored in full; "
        "this cropping is for prompt size only. Do NOT interpret the cut-off as the "
        "agent running out of budget or work being incomplete.]"
    )


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
_BAD_SUMMARY_PATTERNS = (
    "실행하시겠습니까",
    "죄송합니다",
    "당신이 제시한",
    "원하시면",
    "해드릴까요",
    "Would you like",
    "Let me know if",
)


def _hydrate_summary_state():
    global _summary_state_hydrated
    if _summary_state_hydrated:
        return
    hydrate_summary_state(_clear_after_id)
    _summary_state_hydrated = True


_RAW_MSG_HARD_CAP = 500  # safety ceiling; normally the summary cursor keeps
                         # the raw tail far smaller (<= _SUMMARY_CHUNK_SIZE-ish)


def _build_extractive_chat_summary(chunk: list[dict]) -> str:
    """Fallback summary that cannot answer the user or invent next actions."""
    users = [
        _normalize_history_content(r.get("content", "")).replace("\n", " ").strip()
        for r in chunk if r.get("role") == "user"
    ]
    assistants = [
        _normalize_history_content(r.get("content", "")).replace("\n", " ").strip()
        for r in chunk if r.get("role") == "assistant"
    ]
    first_user = users[0][:180] if users else "사용자 발화 없음"
    last_user = users[-1][:180] if users else first_user
    last_assistant = assistants[-1][:220] if assistants else "아직 assistant 응답 없음"
    return (
        f"사용자는 '{first_user}'로 대화를 시작했고, 최근에는 '{last_user}'라고 말했다. "
        f"마지막 assistant 응답은 '{last_assistant}'였다. "
        "이 요약은 원문 대화의 추출식 압축이며 사용자에게 보내는 답변이 아니다."
    )[:900]


def _summary_is_contaminated(summary: str) -> bool:
    """Detect summaries that are actually a direct reply to the user."""
    if not summary or not summary.strip():
        return True
    text = summary.strip()
    if any(p in text for p in _BAD_SUMMARY_PATTERNS):
        return True
    if text.endswith("?") or text.endswith("습니까?"):
        return True
    # A valid summary should describe the conversation, not address "you".
    if re.search(r"\b(you|your)\b", text, flags=re.IGNORECASE) and "user" not in text.lower():
        return True
    return False


def _load_context_with_summaries(user_id: int) -> list[dict]:
    """Load chat context: chunk summaries + raw messages after the last summary.

    Summaries are injected as a single context preamble (not fake conversation
    pairs) so the model sees a clear timeline:
      [context preamble with all useful summaries]  →  [raw messages after]
    Raw window is anchored at the last summary's ``chunk_end_id`` (or the
    user's clear marker when there are none), NOT a sliding fixed-size window.
    That makes the prompt prefix byte-stable across successive turns — only
    the tail grows by the newly-appended turn — so Anthropic prompt caching
    hits the full prefix. When enough raw messages accumulate, the background
    summarizer collapses them into a new summary, which is the one moment
    the prefix legitimately shifts.
    """
    _hydrate_summary_state()
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

    # Anchor the raw-message window at the last summary's chunk_end_id so the
    # resulting sequence is append-only between summarizer runs: new turns
    # extend the tail; older turns never shift out of the window. HARD_CAP is
    # a safety net for the edge case where summarization has lagged far behind.
    raw_anchor_id = summaries[-1]["chunk_end_id"] if summaries else min_id
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, role, content, created_at FROM telegram_chat_history "
                "WHERE user_id = %s AND id > %s "
                "ORDER BY id ASC LIMIT %s",
                (user_id, raw_anchor_id, _RAW_MSG_HARD_CAP),
            )
            raw_rows = cur.fetchall()

    # With the raw anchor at the last summary's tail, every summary is useful
    # (none overlaps with the raw window by construction).
    useful_summaries = summaries

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

    # Append raw messages with exact timestamps on user messages.
    #
    # NOTE: System events (task_report done, startup, deploy complete, model
    # switches, etc.) are intentionally NOT interleaved into the conversation
    # history anymore. Mixing them with real dialogue contaminates the
    # narrative plane — the model reads them as utterances, mis-attributes
    # causation, and loses track of who actually said what.
    #
    # High-severity alerts still reach the model via the `<system-alerts>`
    # block in the system prompt (see `_format_system_alerts`). All events
    # remain queryable on demand through `read_self(content_type="task_report" |
    # "server_logs" | "system_status")`. The `telegram_system_events` table
    # is still written to — only the injection into chat context is removed.
    for r in raw_rows:
        role = r["role"] if r["role"] in ("user", "assistant") else "user"
        text = _normalize_history_content(r["content"])
        ts = r.get("created_at")

        if ts and hasattr(ts, "strftime") and role == "user":
            ts_kst = ts.astimezone(KST) if ts.tzinfo else ts
            time_str = ts_kst.strftime("%Y-%m-%d %H:%M")
            text = f"[{time_str}] {text}" if text else f"[{time_str}]"

        context.append({"role": role, "content": text or "(empty)"})

    return context


async def _maybe_summarize_chunk(user_id: int):
    """Create a summary chunk if enough unsummarized messages have accumulated."""
    try:
        await asyncio.to_thread(_hydrate_summary_state)
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
            "Summarize the conversation below for future context only. "
            "Write in third person. Do NOT answer the user. Do NOT apologize. "
            "Do NOT give advice, ask a question, or propose execution. "
            "Keep only: 1) user topic/request, 2) assistant conclusions/results, "
            "3) unresolved items. Preserve proper nouns, numbers, dates, and "
            "specific decisions. Korean, 500 characters max.\n\n"
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
        if _summary_is_contaminated(summary):
            logger.warning(
                "Chunk summary contaminated; using extractive fallback user=%d msgs=#%d~#%d",
                user_id, chunk_start_id, chunk_end_id,
            )
            summary = _build_extractive_chat_summary(chunk)

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


async def _persist_assistant_turn_after_send(user_id: int, reply: str) -> None:
    """Persist a sent assistant turn without holding up Telegram delivery."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_save_chat_message, user_id, "assistant", reply),
            timeout=CHAT_PERSIST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "assistant chat history persist timed out after %.1fs for user_id=%s",
            CHAT_PERSIST_TIMEOUT_SECONDS,
            user_id,
        )
        return
    except Exception as e:
        logger.warning("assistant chat history persist failed for user_id=%s: %s", user_id, e)
        return

    try:
        await _maybe_summarize_chunk(user_id)
    except Exception as e:
        logger.warning("post-send chat summarization failed for user_id=%s: %s", user_id, e)


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
    max_input_tokens: int | None = None,
    max_output_continuations: int = 0,
    thinking_policy: str = "tool_loop",
    thinking_budget_tokens: int = 8192,
    budget_usd: float | None = None,
    extra_tools: list | None = None,
    extra_handlers: dict | None = None,
    on_progress=None,
    budget_tracker: dict | None = None,
    task_id: int | None = None,
    provider_override: str | None = None,
    finalization_tools: list[str] | None = None,
    terminal_tools: list[str] | None = None,
    extra_system_context: str = "",
    agent_name: str | None = None,
    runtime_kind: str | None = None,
    deepseek_thinking_override: dict | None = None,
) -> str:
    """Call LLM with tools — dispatches to Claude or OpenAI based on provider config.

    provider_override: if set, forces this provider instead of _config["provider"].
    Used by web_chat, diary writer, etc. to always use corporate LLM.
    extra_system_context: appended to the rendered orchestrator system prompt
    (active mission timeline, task state block, retrieved experiences, etc.).
    Ignored when `system_prompt` is passed directly.
    deepseek_thinking_override: per-call replacement for the global DeepSeek
    thinking params (same shape as _get_deepseek_thinking_params()). Short
    format-constrained calls (autonomous tick planner/critic) pass
    {"thinking": {"type": "disabled"}} — reasoning-mode burn was exhausting
    their whole max_tokens before any visible reply. Ignored by non-DeepSeek
    providers.
    """
    # Resolve provider up-front so the system prompt can be rendered in the
    # format native to the target model family (XML for Claude, Markdown for
    # OpenAI/Qwen). provider_override wins over the stored config.
    effective_provider = provider_override or _config.get("provider", "claude")

    # System prompt: orchestrator builds its own fully static prompt; agents
    # pass their pre-rendered (also static, post-refactor) spec prompt.
    if system_prompt is None:
        sys_prompt = _build_orchestrator_system_prompt(effective_provider).format(
            skills_section=build_skills_prompt(),
        )
        _runtime_kind = runtime_kind or "chat"
    else:
        sys_prompt = system_prompt
        _runtime_kind = runtime_kind or "task"
    if _runtime_kind not in ("chat", "task", "autonomous"):
        logger.warning("Unknown runtime_kind=%r; falling back to task", _runtime_kind)
        _runtime_kind = "task"
    profile = await resolve_runtime_profile(
        _runtime_kind,
        provider_override=effective_provider,
        model_override=model,
        budget_override=budget_usd,
        max_rounds_override=max_rounds,
        max_tokens_override=max_tokens,
    )
    effective_provider = profile.provider
    resolved_max_rounds = profile.max_rounds
    resolved_max_tokens = profile.max_tokens
    from tool_gateway.inference import DEFAULT_AGENT_MAX_INPUT_TOKENS
    resolved_max_input_tokens = int(max_input_tokens or DEFAULT_AGENT_MAX_INPUT_TOKENS)
    resolved_output_continuations = max(0, int(max_output_continuations or 0))
    resolved_budget = profile.budget_usd
    from tool_gateway.inference import AgentInferencePolicy, resolve_inference_extra
    call_inference_policy = AgentInferencePolicy(
        max_input_tokens=resolved_max_input_tokens,
        max_output_tokens=resolved_max_tokens,
        max_rounds=resolved_max_rounds,
        budget_usd=resolved_budget,
        max_output_continuations=resolved_output_continuations,
        thinking_policy=thinking_policy,
        thinking_budget_tokens=thinking_budget_tokens,
    )

    # Runtime context injection (applies to orchestrator AND agents). Volatile
    # data — current time, current model, caller-supplied extras (mission,
    # experiences, state), and system alerts — rides on the trailing user
    # message so the system prompt and history prefix stay byte-stable across
    # turns for prompt caching. Order: runtime header at top (stable framing);
    # caller extras in the middle (stable→volatile gradient); alerts at the
    # bottom (freshest, closest to the user's query for recency attention).
    full_runtime_context = _join_context_blocks(
        _build_runtime_prelude(effective_provider, kind=_runtime_kind),
        extra_system_context or "",
        _format_system_alerts(effective_provider),
    )
    messages = _merge_runtime_context_into_last_user(messages, full_runtime_context)
    is_orchestrator = extra_tools is None

    if is_orchestrator:
        merged_tools = select_orchestrator_tools(TOOLS)
    else:
        # Task/agent: use ONLY extra_tools (already filtered by agent spec).
        # Do NOT merge full TOOLS — that would bypass agent tool restrictions.
        merged_tools = list(extra_tools or [])
    # Orchestrator: full handler set. Task/agent: only the handlers for allowed tools.
    if is_orchestrator:
        merged_handlers = {**TOOL_HANDLERS, **(extra_handlers or {})}
    else:
        merged_handlers = dict(extra_handlers or {})

    if is_orchestrator and "list_agent_tools" in {t.get("name") for t in merged_tools}:
        from self_runtime.tools import build_list_agent_tools_handler
        merged_handlers["list_agent_tools"] = build_list_agent_tools_handler(merged_tools)

    # Inject run_agent handler (needs _chat_with_tools closure — can't be registered at import time)
    if is_orchestrator and "run_agent" not in merged_handlers:
        from self_runtime.tools import build_run_agent_handler
        merged_handlers["run_agent"] = build_run_agent_handler(_chat_with_tools)

    # Resolve agent name + mission for provenance tracking
    _agent_name = agent_name or ("orchestrator" if is_orchestrator else "agent")
    _mission_id: int | None = None
    if agent_name is None and not is_orchestrator and task_id is not None:
        try:
            from db import query as _db_q
            row = _db_q(
                "SELECT agent_type, mission_id FROM telegram_tasks WHERE id = %s",
                (task_id,),
            )
            if row:
                _agent_name = str(row[0].get("agent_type") or "agent")
                _mission_id = row[0].get("mission_id")
        except Exception:
            pass

    # ── Security gateway caller context ──
    # Telegram is the owner's gated control channel (ALLOWED_USER_IDS upstream),
    # so calls here are trusted. Orchestrator vs delegated-agent is distinguished
    # for audit attribution; caller_scope restores the parent on exit so a nested
    # run_agent sub-call doesn't leak its agent identity back to the orchestrator.
    from tool_gateway.security import CallerContext, caller_scope
    _gw_ctx = CallerContext(
        interface="telegram" if is_orchestrator else "agent",
        agent_name=None if is_orchestrator else _agent_name,
        is_owner=True,
        task_id=str(task_id) if task_id is not None else None,
    )

    # ── Provider dispatch: Claude vs OpenAI vs Local ──
    # effective_provider already resolved above before prompt rendering.
    if effective_provider == "local":
        from openai_tool_loop import chat_with_tools as openai_chat
        from llm.client import (
            _resolve_backend, LOCAL_SEMAPHORE, LOCAL_CONTEXT_LIMIT,
            LOCAL_MAX_TOKENS, LOCAL_ENABLE_THINKING,
        )
        backend = _resolve_backend()
        # Floor the completion budget at LOCAL_MAX_TOKENS (default 8192).
        # The 4096 default shared with Claude truncates Qwen3 responses
        # mid-<think> on Q4 quantizations, so the tool_call is never
        # emitted and the loop returns an empty answer.
        local_max_tokens = max(resolved_max_tokens, LOCAL_MAX_TOKENS)
        _chat_coro = openai_chat(
            messages,
            client=None,
            base_url=backend["base"],
            model=profile.model_id or backend["model"],
            tools=merged_tools,
            tool_handlers=merged_handlers,
            system_prompt=sys_prompt,
            max_rounds=resolved_max_rounds,
            max_tokens=local_max_tokens,
            max_input_tokens=resolved_max_input_tokens,
            recover_input_via_tools=True,
            continue_on_length=resolved_output_continuations > 0,
            max_length_continuations=resolved_output_continuations,
            log_event=_log_event,
            budget_usd=resolved_budget,
            on_progress=on_progress,
            budget_tracker=budget_tracker,
            task_id=task_id,
            context_limit=LOCAL_CONTEXT_LIMIT,
            enable_thinking=is_orchestrator and LOCAL_ENABLE_THINKING,
            agent_name=_agent_name,
            mission_id=_mission_id,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
            api_semaphore=LOCAL_SEMAPHORE,
            provider_label=f"local:{backend['base']}",
        )
        with caller_scope(_gw_ctx):
            return await _chat_coro

    if effective_provider == "openai" and _openai_client:
        from openai_tool_loop import chat_with_tools as openai_chat
        openai_inference = resolve_inference_extra(call_inference_policy, "openai")
        _chat_coro = openai_chat(
            messages,
            client=_openai_client,
            model=profile.model_id,
            tools=merged_tools,
            tool_handlers=merged_handlers,
            system_prompt=sys_prompt,
            max_rounds=resolved_max_rounds,
            max_tokens=resolved_max_tokens,
            max_input_tokens=resolved_max_input_tokens,
            recover_input_via_tools=True,
            continue_on_length=resolved_output_continuations > 0,
            max_length_continuations=resolved_output_continuations,
            log_event=_log_event,
            budget_usd=resolved_budget,
            on_progress=on_progress,
            budget_tracker=budget_tracker,
            task_id=task_id,
            agent_name=_agent_name,
            mission_id=_mission_id,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
            provider_label="openai",
            extra_body=openai_inference.get("extra_body"),
        )
        with caller_scope(_gw_ctx):
            return await _chat_coro

    if effective_provider == "kimi" and _kimi_client:
        from openai_tool_loop import chat_with_tools as openai_chat
        _chat_coro = openai_chat(
            messages,
            client=_kimi_client,
            model=profile.model_id,
            tools=merged_tools,
            tool_handlers=merged_handlers,
            system_prompt=sys_prompt,
            max_rounds=resolved_max_rounds,
            max_tokens=resolved_max_tokens,
            max_input_tokens=resolved_max_input_tokens,
            recover_input_via_tools=True,
            continue_on_length=resolved_output_continuations > 0,
            max_length_continuations=resolved_output_continuations,
            log_event=_log_event,
            budget_usd=resolved_budget,
            on_progress=on_progress,
            budget_tracker=budget_tracker,
            task_id=task_id,
            agent_name=_agent_name,
            mission_id=_mission_id,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
            provider_label="kimi",
            extra_body={"reasoning_effort": "max"},
            sdk_max_token_param="max_tokens",
            include_parallel_tool_calls=False,
            preserve_reasoning_content=True,
        )
        with caller_scope(_gw_ctx):
            return await _chat_coro

    if effective_provider == "deepseek" and _deepseek_anthropic_client:
        deepseek_thinking = deepseek_thinking_override or resolve_inference_extra(
            call_inference_policy, "deepseek"
        )
        _chat_coro = chat_with_tools(
            messages,
            client=_deepseek_anthropic_client,
            model=profile.model_id,
            tools=merged_tools,
            tool_handlers=merged_handlers,
            system_prompt=sys_prompt,
            max_rounds=resolved_max_rounds,
            max_tokens=resolved_max_tokens,
            max_input_tokens=resolved_max_input_tokens,
            recover_input_via_tools=True,
            continue_on_length=resolved_output_continuations > 0,
            max_length_continuations=resolved_output_continuations,
            log_event=_log_event,
            budget_usd=resolved_budget,
            on_progress=on_progress,
            budget_tracker=budget_tracker,
            task_id=task_id,
            agent_name=_agent_name,
            mission_id=_mission_id,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
            thinking=deepseek_thinking.get("thinking"),
            output_config=deepseek_thinking.get("output_config"),
        )
        with caller_scope(_gw_ctx):
            return await _chat_coro

    if effective_provider in ("openai", "deepseek", "kimi"):
        missing = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
        }[effective_provider]
        raise RuntimeError(f"{missing} is not configured for provider={effective_provider}")

    # Claude path. `messages` has already had the runtime context merged into
    # the trailing user turn by `_merge_runtime_context_into_last_user` above,
    # so history remains byte-stable across turns and prefix caching works.
    claude_inference = resolve_inference_extra(call_inference_policy, "claude")
    _chat_coro = chat_with_tools(
        messages,
        client=_claude,
        model=profile.model_id,
        tools=merged_tools,
        tool_handlers=merged_handlers,
        system_prompt=sys_prompt,
        max_rounds=resolved_max_rounds,
        max_tokens=resolved_max_tokens,
        max_input_tokens=resolved_max_input_tokens,
        recover_input_via_tools=True,
        continue_on_length=resolved_output_continuations > 0,
        max_length_continuations=resolved_output_continuations,
        log_event=_log_event,
        budget_usd=resolved_budget,
        on_progress=on_progress,
        budget_tracker=budget_tracker,
        task_id=task_id,
        agent_name=_agent_name,
        mission_id=_mission_id,
        finalization_tools=finalization_tools,
        terminal_tools=terminal_tools,
        thinking=claude_inference.get("thinking"),
    )
    with caller_scope(_gw_ctx):
        return await _chat_coro


# ── Router & Handlers ───────────────────────────────────────────────
router = Router()

# Register command handlers from extracted module
from telegram.commands import register_handlers
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
    "current_datetime_str": _current_datetime_str,
    "format_current_model_context": _format_current_model_context,
    "format_system_alerts": _format_system_alerts,
    "format_autonomous_status": _format_autonomous_status,
    "join_context_blocks": _join_context_blocks,
    "add_system_alert": _add_system_alert,
    "clear_system_alert": _clear_system_alert,
    "claude_client": _claude,
    "openai_client": _openai_client,
    "deepseek_client": _deepseek_client,
    "deepseek_anthropic_client": _deepseek_anthropic_client,
    "kimi_client": _kimi_client,
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
                    await bot.send_message(int(target_chat_id), text)
        except Exception as e:
            logger.warning("email bridge poll loop error: %s", e)
            _log_event("warning", "email_bridge", f"poll loop error: {e}")
        await asyncio.sleep(EMAIL_POLL_INTERVAL_SECONDS)


# ── Per-agent dispatch helpers ────────────────────────────────────────
# These resolve which chat_with_tools implementation + which model an agent
# spec runs against. Keeping them at module level (rather than as inline
# closures inside bot_main) keeps the dispatch block in bot_main short and
# makes the routing testable in isolation.

async def _get_model_for_agent(spec):
    """Resolve the LLM model an agent should run against.

    Priority: spec.provider override wins over task_provider, which can be
    independent from the Telegram chat provider. Returns the API model ID.
    """
    if spec.provider == "moon":
        return await _get_model_moon()
    if spec.provider == "codex":
        from codex_exec_loop import CODEX_DEFAULT_MODEL
        return CODEX_DEFAULT_MODEL
    provider = spec.provider or _get_task_provider()
    if provider == "local":
        if spec.model:
            return spec.model
        from llm.client import _resolve_backend
        return _resolve_backend()["model"]
    if provider in ("claude", "openai", "deepseek", "kimi"):
        profile = await resolve_runtime_profile(
            "task",
            provider_override=provider,
            tier_override=spec.model,
        )
        return profile.model_id
    return await _get_model_task()


def _make_moon_chat_fn(spec):
    """Build a chat_fn that targets the local llama-server (MOON PC).

    Falls back to the standard Claude/OpenAI chat fn if MOON is unhealthy
    so the agent run still succeeds.
    """
    from openai_tool_loop import chat_with_tools as _moon_loop
    from llm.client import MOON_BASE, MOON_MODEL, _health_ok
    if not _health_ok(MOON_BASE):
        fallback_provider = _get_task_provider()
        logger.warning(
            "MOON unavailable for agent %s; falling back to task provider %s",
            spec.name, fallback_provider,
        )
        fallback = _make_provider_chat_fn(fallback_provider)
        setattr(fallback, "_fallback_provider", fallback_provider)
        return fallback

    async def _moon_chat_fn(
        messages, max_rounds=None, system_prompt=None, model=None,
        max_tokens=None, max_input_tokens=None, max_output_continuations=0,
        thinking_policy="tool_loop", thinking_budget_tokens=8192, budget_usd=None, extra_tools=None,
        extra_handlers=None, on_progress=None, budget_tracker=None,
        task_id=None, finalization_tools=None, terminal_tools=None,
    ):
        return await _moon_loop(
            messages,
            base_url=MOON_BASE,
            model=model or MOON_MODEL,
            tools=list(extra_tools or []),
            tool_handlers=dict(extra_handlers or {}),
            system_prompt=system_prompt,
            max_rounds=max_rounds or spec.max_rounds,
            max_tokens=max_tokens or spec.max_output_tokens,
            max_input_tokens=max_input_tokens or spec.max_input_tokens,
            recover_input_via_tools=True,
            continue_on_length=max_output_continuations > 0,
            max_length_continuations=max_output_continuations,
            log_event=_log_event,
            budget_usd=budget_usd or 0.0,
            budget_tracker=budget_tracker,
            on_progress=on_progress,
            task_id=task_id,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
        )
    return _moon_chat_fn


def _make_codex_chat_fn(spec):
    """Build a chat_fn that delegates the entire task to OpenAI Codex CLI."""
    from codex_exec_loop import chat_with_tools as _codex_loop

    async def _codex_chat_fn(
        messages, max_rounds=None, system_prompt=None, model=None,
        max_tokens=None, max_input_tokens=None, max_output_continuations=0,
        thinking_policy="tool_loop", thinking_budget_tokens=8192, budget_usd=None, extra_tools=None,
        extra_handlers=None, on_progress=None, budget_tracker=None,
        task_id=None, finalization_tools=None, terminal_tools=None,
    ):
        return await _codex_loop(
            messages,
            model=model,
            tools=extra_tools, tool_handlers=extra_handlers,
            system_prompt=system_prompt,
            max_rounds=max_rounds or spec.max_rounds,
            max_tokens=max_tokens or 8192,
            log_event=_log_event,
            budget_usd=budget_usd or 0.0,
            budget_tracker=budget_tracker,
            on_progress=on_progress,
            task_id=task_id,
            agent_name=spec.name,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
        )
    return _codex_chat_fn


def _make_provider_chat_fn(provider: str):
    """Build a chat_fn that forces _chat_with_tools to use a specific provider.

    Used when an agent spec pins a provider or task_provider differs from the
    Telegram chat provider.
    """
    async def _provider_chat_fn(
        messages, max_rounds=None, system_prompt=None, model=None,
        max_tokens=None, max_input_tokens=None, max_output_continuations=0,
        thinking_policy="tool_loop", thinking_budget_tokens=8192, budget_usd=None, extra_tools=None,
        extra_handlers=None, on_progress=None, budget_tracker=None,
        task_id=None, finalization_tools=None, terminal_tools=None,
        agent_name=None,
        runtime_kind=None,
    ):
        return await _chat_with_tools(
            messages, max_rounds=max_rounds, system_prompt=system_prompt,
            model=model, max_tokens=max_tokens, max_input_tokens=max_input_tokens,
            max_output_continuations=max_output_continuations,
            thinking_policy=thinking_policy,
            thinking_budget_tokens=thinking_budget_tokens, budget_usd=budget_usd,
            extra_tools=extra_tools, extra_handlers=extra_handlers,
            on_progress=on_progress, budget_tracker=budget_tracker,
            task_id=task_id, provider_override=provider,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
            agent_name=agent_name,
            runtime_kind=runtime_kind,
        )
    return _provider_chat_fn


async def _run_stasova_diary_review(task: dict, title: str, content: str) -> str:
    from agents import get_agent
    from runtime_tools.registry import TOOLS as BASE_TOOLS, TOOL_HANDLERS as BASE_HANDLERS

    spec = get_agent("stasova")
    agent_tools, agent_handlers = spec.filter_tools(BASE_TOOLS, BASE_HANDLERS)
    provider = spec.effective_provider(_get_task_provider())
    chat_fn = _make_provider_chat_fn(provider)
    report_path = f"temp_dev/stasova_reviews/diary_task_{task.get('id', 'unknown')}.md"
    review_task = (
        "다음 공개 일기 초안을 출판 보안 관점에서 점검하라.\n"
        "정치 노선 개정 판단이나 문학적 편집은 하지 말고, 시스템 프롬프트의 위험 축과 정치노선 왜곡 위험만 적용하라.\n"
        f"점검 보고서를 `{report_path}`에도 저장하라.\n\n"
        f"제목:\n{title}\n\n"
        f"본문:\n{content}"
    )
    return await chat_fn(
        [{"role": "user", "content": review_task}],
        system_prompt=spec.render_prompt(provider=provider),
        model=await _get_model_for_agent(spec),
        max_rounds=spec.max_rounds,
        max_tokens=4096,
        budget_usd=spec.budget_usd,
        extra_tools=agent_tools,
        extra_handlers=agent_handlers,
        task_id=task.get("id"),
        agent_name="stasova",
        runtime_kind="task",
    )


def _extract_json_object(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


async def _apply_stasova_diary_review(
    task: dict,
    *,
    title: str,
    content: str,
    review_report: str,
) -> tuple[str, str]:
    provider = _get_task_provider()
    chat_fn = _make_provider_chat_fn(provider)
    revision_prompt = (
        "스타소바의 출판 보안 보고서를 반영해 공개 일기의 최종본을 작성하라.\n"
        "공개를 지연하거나 승인 요청을 만들지 않는다. 경고가 없으면 원문을 유지한다.\n"
        "경고가 있으면 비밀, 개인 식별, 비공개 조직 정보, 과도한 법적·플랫폼 위험 표현만 줄인다.\n"
        "정치적 핵심과 1인칭 일기 문체는 보존한다.\n"
        "반드시 JSON 객체 하나만 출력하라. 스키마: {\"title\": \"...\", \"content\": \"...\"}\n\n"
        f"초안 제목:\n{title}\n\n"
        f"초안 본문:\n{content}\n\n"
        f"스타소바 보고서:\n{review_report}"
    )
    response = await chat_fn(
        [{"role": "user", "content": revision_prompt}],
        system_prompt=(
            "너는 공개 일기 최종본 편집기다. 출판 보안 경고만 반영하고, "
            "본문을 새 글로 확장하거나 승인 절차를 제안하지 않는다."
        ),
        model=await _get_model_task(),
        max_rounds=1,
        max_tokens=4096,
        budget_usd=0.20,
        extra_tools=[],
        extra_handlers={},
        task_id=task.get("id"),
        agent_name="diary_revision",
        runtime_kind="task",
    )
    parsed = _extract_json_object(response)
    if not parsed:
        logger.warning("Stasova diary revision returned non-JSON; publishing original draft")
        return title, content
    final_title = str(parsed.get("title") or title).strip() or title
    final_content = str(parsed.get("content") or content).strip() or content
    return final_title, final_content


def _normalize_guarded_diary_save_args(args: tuple, kwargs: dict) -> tuple[str | None, str | None]:
    """Accept save_diary payloads from strict and loose tool-loop dispatchers."""
    title = kwargs.get("title")
    content = kwargs.get("content")

    for nested_key in ("input", "arguments", "args"):
        nested = kwargs.get(nested_key)
        if isinstance(nested, dict):
            title = title if title is not None else nested.get("title")
            content = content if content is not None else nested.get("content")

    if args:
        first = args[0]
        if isinstance(first, dict):
            title = title if title is not None else first.get("title")
            content = content if content is not None else first.get("content")
        else:
            title = title if title is not None else first
            if len(args) > 1:
                content = content if content is not None else args[1]

    return (
        str(title).strip() if title is not None else None,
        str(content).strip() if content is not None else None,
    )


def _is_diary_writing_task(task: dict) -> bool:
    """Return True only for the configured scheduled diary-writing prompt."""
    from telegram.diary_mode import is_diary_writing_task

    return is_diary_writing_task(task)


def _make_guarded_diary_save_handler(_bot: Bot, task: dict):
    async def _guarded_save_diary(*args, title: str | None = None, content: str | None = None, **kwargs) -> str:
        if task.get("agent_type") == "diary" and not _is_diary_writing_task(task):
            raise RuntimeError(
                "save_diary blocked: diary publication is allowed only for the configured scheduled "
                "diary-writing prompt. For all other diary-agent tasks, act autonomously with the "
                "appropriate maintenance tool such as edit_content; do not create a new diary."
            )
        title, content = _normalize_guarded_diary_save_args(
            args,
            {"title": title, "content": content, **kwargs},
        )
        if not title:
            return "Failed to publish reviewed diary: save_diary missing required argument: title"
        if not content:
            return "Failed to publish reviewed diary: save_diary missing required argument: content"
        try:
            review_report = await _run_stasova_diary_review(task, title, content)
            final_title, final_content = await _apply_stasova_diary_review(
                task,
                title=title,
                content=content,
                review_report=review_report,
            )
            from telegram.diary_publication import (
                publish_reviewed_diary_entry,
            )

            diary_id, broadcast_note, audit_id = await publish_reviewed_diary_entry(
                task_id=task.get("id"),
                title=title,
                content=content,
                final_title=final_title,
                final_content=final_content,
                review_report=review_report,
            )
            url = f"https://cyber-lenin.com/ai-diary/{diary_id}" if diary_id else "https://cyber-lenin.com/ai-diary"
            audit_note = f" / Stasova warning audit #{audit_id}" if audit_id else ""
            return f"Diary reviewed by Stasova and published automatically: {final_title}\n{url}{broadcast_note}{audit_note}"
        except Exception as exc:
            logger.error("guarded diary save failed: %s", exc)
            return f"Failed to publish reviewed diary: {exc}"

    return _guarded_save_diary


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

    await asyncio.to_thread(hydrate_summary_state, _clear_after_id)
    recovery = await recover_processing_tasks_on_startup(stale_minutes=60, max_resume_attempts=2)
    handed_off = int(recovery.get("handed_off", recovery.get("resumed", 0)))
    closed_stale = int(recovery.get("closed_stale", 0))
    closed_repeated = int(recovery.get("closed_repeated", 0))
    if handed_off or closed_stale or closed_repeated:
        _add_system_alert(
            f"Restart recovery: handoff {handed_off} / stale closed {closed_stale} / repeated-failure closed {closed_repeated}"
        )

    global _bot_instance
    session = AiohttpSession(timeout=TELEGRAM_SESSION_TIMEOUT_SECONDS, limit=100)
    bot = Bot(token=TELEGRAM_BOT_TOKEN, session=session)
    _bot_instance = bot
    dp = Dispatcher()
    access_middleware = OwnerOnlyMiddleware()
    dp.message.middleware(access_middleware)
    dp.callback_query.middleware(access_middleware)
    dp.my_chat_member.middleware(access_middleware)
    dp.my_chat_member.register(_ignore_chat_member_update)
    dp.update.register(_handle_guest_update, _has_guest_message)
    dp.include_router(router)

    email_bridge_task = None
    if EMAIL_BRIDGE_ENABLED and EMAIL_POLLING_ENABLED:
        email_bridge_task = asyncio.create_task(_email_bridge_poll_loop(bot), name="email-bridge-poll")

    # Register commands for Telegram "/" autocomplete menu
    from aiogram.types import BotCommand
    await bot.set_my_commands([
        BotCommand(command="help", description="커맨드 목록"),
        BotCommand(command="task", description="백그라운드 태스크 등록"),
        BotCommand(command="status", description="시스템 대시보드"),
        BotCommand(command="report", description="태스크 리포트 재전송"),
        BotCommand(command="config", description="설정 패널"),
        BotCommand(command="agents", description="에이전트 현황 / 워커 상태"),
        BotCommand(command="projects", description="자율 프로젝트 목록"),
        BotCommand(command="project", description="자율 프로젝트 상세/수정"),
        BotCommand(command="channel", description="브로드캐스트 채널 설정"),
        BotCommand(command="restart", description="서비스 재시작"),
        BotCommand(command="clear", description="대화 히스토리 초기화"),
    ])

    try:
        me = await bot.get_me()
        supports_guest_queries = getattr(me, "supports_guest_queries", None)
        aiogram_version = "unknown"
        try:
            import aiogram
            aiogram_version = aiogram.__version__
        except Exception:
            pass
        if OWNER_USER_ID:
            await asyncio.to_thread(
                _save_system_event,
                OWNER_USER_ID,
                "telegram_api10_startup",
                "Telegram Bot API 10.0 shim active; "
                f"aiogram={aiogram_version}; supports_guest_queries={supports_guest_queries}; "
                f"guest_policy={TELEGRAM_GUEST_MODE_POLICY}; allowed_updates={','.join(_allowed_updates())}; "
                f"polling_timeout={TELEGRAM_POLLING_TIMEOUT_SECONDS}; "
                f"session_timeout={TELEGRAM_SESSION_TIMEOUT_SECONDS}; "
                f"polling_concurrency={TELEGRAM_POLLING_CONCURRENCY_LIMIT}; "
                f"backoff_max={TELEGRAM_BACKOFF_MAX_SECONDS}",
            )
    except Exception as e:
        logger.warning("Telegram API 10.0 startup self-check failed: %s", e)

    # Detect fresh deploy — inject context so the bot knows it was just updated
    await check_deploy_meta(bot, add_alert_fn=_add_system_alert)

    # ── Orchestrator callback: interpret task results for the user ──
    async def _orchestrator_report_task(b: Bot, task: dict, result: dict, chat_id: int):
        """Trigger an orchestrator turn to interpret task results, communicate to user, and redelegate if needed."""
        task_id = task["id"]
        agent_type = task.get("agent_type") or "analyst"
        status = result.get("status", "unknown")
        was_interrupted = result.get("was_interrupted", False)
        mission_id = task.get("mission_id")

        # Check if this was the last active task in the mission
        mission_close_hint = ""
        if mission_id and status in ("done", "failed"):
            try:
                remaining = await asyncio.to_thread(
                    _query_one,
                    "SELECT COUNT(*) AS cnt FROM telegram_tasks "
                    "WHERE mission_id = %s AND id != %s AND status IN ('pending', 'processing', 'queued')",
                    (mission_id, task_id),
                )
                if remaining and remaining["cnt"] == 0:
                    mission_row = await asyncio.to_thread(
                        _query_one,
                        "SELECT id, title FROM telegram_missions WHERE id = %s AND status = 'active'",
                        (mission_id,),
                    )
                    if mission_row:
                        mission_close_hint = (
                            f"\n\n📋 Mission #{mission_row['id']} \"{mission_row['title']}\" has no remaining active tasks. "
                            f"If the user's original goal has been addressed, call `mission(action=\"close\")` to close it."
                        )
            except Exception:
                pass

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
                    tool_log_section = f"\n\nAgent work log (tool call history):\n{_truncate_for_prompt(tool_log, 5000)}"

                # Independent verification verdict (Critic). A shadow-mode FAIL
                # is advisory: the orchestrator relays the caveat, nothing was
                # auto-retried.
                verification = result.get("verification") or {}
                verification_section = ""
                if verification.get("status") == "failed":
                    retry = verification.get("retry") or {}
                    if retry.get("status") == "redelegated":
                        retry_note = f" An automatic retry was delegated as task #{retry.get('task_id')}."
                    elif retry.get("status") == "restart_initiated":
                        retry_note = f" {retry.get('message', 'Service restart initiated.')}"
                    elif verification.get("mode") == "shadow":
                        retry_note = " (shadow mode — no automatic retry was taken)"
                    else:
                        retry_note = f" No retry: {retry.get('message', 'retry unavailable')}."
                    verification_section = (
                        f"\n\n⚠️ Independent verification FAILED.{retry_note}\n"
                        f"Verifier findings:\n{_truncate_for_prompt(verification.get('details', ''), 800)}\n"
                        f"Mention this caveat to the user when relaying the results."
                    )

                prompt = (
                    f"[TASK REPORT] Task #{task_id} [{agent_type}] completed{' (interrupted)' if was_interrupted else ''}\n\n"
                    f"Original request:\n{_truncate_for_prompt(task.get('content', ''), 1000)}\n\n"
                    f"Execution result:\n{_truncate_for_prompt(report, 3000)}"
                    f"{tool_log_section}"
                    f"{verification_section}"
                    f"{interrupted_note}\n\n"
                    f"## Your role\n"
                    f"1. Relay the results to the user concisely, covering only key points. Do not use markdown formatting.\n"
                    f"2. Re-delegation judgment: Only delegate follow-up work when ALL of these conditions are met:\n"
                    f"   - The agent could not finish due to budget/turn limits\n"
                    f"   - Additional work can yield meaningful improvement\n"
                    f"   - The cause is NOT external factors (permission denied, blocked, CAPTCHA, API error, etc.)\n"
                    f"   If re-delegation is unnecessary, just relay the results."
                    f"{mission_close_hint}"
                )
            else:
                error = result.get("error", "unknown error")
                prompt = (
                    f"[TASK REPORT] Task #{task_id} [{agent_type}] failed\n\n"
                    f"Original request:\n{_truncate_for_prompt(task.get('content', ''), 500)}\n\n"
                    f"Error: {error}\n\n"
                    f"Inform the user of the failure and its cause concisely. "
                    f"Do not re-delegate if the issue would not be resolved by retrying."
                    f"{mission_close_hint}"
                )

            # Load recent chat history for context
            history = await asyncio.to_thread(_load_context_with_summaries, chat_id)
            history.append({"role": "user", "content": prompt})

            # Run orchestrator — budget enough for response + optional redelegate call
            from runtime_tools.registry import build_mission_handler
            reply = await _chat_with_tools(
                history,
                budget_usd=0.15,
                max_rounds=5,
                extra_handlers={"mission": build_mission_handler(chat_id)},
            )

            for chunk in _split_message(reply):
                await b.send_message(chat_id=chat_id, text=chunk)
            asyncio.create_task(_persist_assistant_turn_after_send(chat_id, reply))
            await asyncio.to_thread(_save_system_event, chat_id, "task_report", f"task #{task_id} [{agent_type}] {status}")

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

        from self_runtime.tools import build_task_context_tools
        from runtime_tools.registry import TOOLS as BASE_TOOLS, TOOL_HANDLERS as BASE_HANDLERS
        from runtime_tools.registry import build_mission_handler

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

        # Add task-context tools (save_finding), except for Stasova whose
        # publication-security tool surface is deliberately minimal.
        if agent_type != "stasova":
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

        # Render agent-specific system prompt in the format native to the
        # provider that will actually run this agent (local/openai → Markdown,
        # claude → XML). spec.effective_provider falls back to config when
        # the agent has no pinned provider. Prompt is fully static post-refactor
        # — current time, current model, and alerts are injected as runtime
        # context by _chat_with_tools, not baked into the system prompt.
        task_provider = _get_task_provider()
        from tool_gateway.inference import resolve_agent_inference_policy
        inference_policy = resolve_agent_inference_policy(spec)
        _agent_provider = spec.effective_provider(task_provider)
        system_prompt = spec.render_prompt(provider=_agent_provider)

        # Inject runtime environment info for programmer (needs venv, packages, services)
        if agent_type == "programmer":
            system_prompt += "\n" + _build_env_context()

        # Send progress to the task's user (or all users if self-generated)
        target_chat_id = task["user_id"] if task["user_id"] != 0 else OWNER_USER_ID
        progress_cb = _make_progress_callback(target_chat_id) if target_chat_id else None

        # ── Provider dispatch: chat_fn varies per provider; model_fn unified ──
        # Helpers (_make_*_chat_fn, _get_model_for_agent) live at module level
        # so this stays a thin routing block. provider=None follows
        # task_provider, which may differ from the Telegram chat provider.
        if spec.provider == "moon":
            chosen_chat_fn = _make_moon_chat_fn(spec)
            fallback_provider = getattr(chosen_chat_fn, "_fallback_provider", None)
            chosen_max_tokens = inference_policy.max_output_tokens
        elif spec.provider == "codex":
            chosen_chat_fn = _make_codex_chat_fn(spec)
            chosen_max_tokens = inference_policy.max_output_tokens
        elif spec.provider in ("claude", "openai", "deepseek", "kimi", "local"):
            chosen_chat_fn = _make_provider_chat_fn(spec.provider)
            chosen_max_tokens = inference_policy.max_output_tokens
        elif task_provider in ("claude", "openai", "deepseek", "kimi", "local"):
            chosen_chat_fn = _make_provider_chat_fn(task_provider)
            chosen_max_tokens = inference_policy.max_output_tokens
        else:
            chosen_chat_fn = _chat_with_tools
            chosen_max_tokens = inference_policy.max_output_tokens

        async def chosen_model_fn():
            if spec.provider == "moon":
                fallback_provider = getattr(chosen_chat_fn, "_fallback_provider", None)
                if fallback_provider:
                    if fallback_provider == "local":
                        from llm.client import _resolve_backend
                        return _resolve_backend()["model"]
                    profile = await resolve_runtime_profile("task", provider_override=fallback_provider)
                    return profile.model_id
            return await _get_model_for_agent(spec)

        if agent_type == "diary" and "save_diary" in agent_handlers:
            agent_handlers = dict(agent_handlers)
            agent_handlers["save_diary"] = _make_guarded_diary_save_handler(b, task)

        # ── Post-hoc verification (Critic) routing ──────────────────
        # The verifier runs on the low tier of a standard provider so the
        # critique stays cheap. Codex/moon executors are verified by the task
        # provider — an independent judge for handed-off work.
        verification_mode = get_task_verification_mode()
        verify_chat_fn = None
        verify_model_fn = None
        if verification_mode in ("shadow", "enforce"):
            if spec.provider in ("claude", "openai", "deepseek", "kimi", "local"):
                verify_provider = spec.provider
            elif task_provider in ("claude", "openai", "deepseek", "kimi", "local"):
                verify_provider = task_provider
            else:
                verify_provider = "claude"
            verify_chat_fn = _make_provider_chat_fn(verify_provider)

            async def verify_model_fn(_provider=verify_provider):
                profile = await resolve_runtime_profile(
                    "task", provider_override=_provider, tier_override="low",
                )
                return profile.model_id

        def _on_task_complete(task_id: int, status: str, summary: str, **kw):
            icon = "✅" if status == "done" else "❌"
            verdict_note = ""
            if kw.get("verification_status") == "failed":
                verdict_note = " ⚠️ verification FAILED"
            _add_system_alert(f"{icon} Task #{task_id} {status}: {summary[:200]}{verdict_note}")

        result = await process_task(
            b, task,
            chat_with_tools_fn=chosen_chat_fn,
            get_model_fn=chosen_model_fn,
            task_system_prompt=system_prompt,
            max_tokens_task=chosen_max_tokens,
            max_input_tokens_task=inference_policy.max_input_tokens,
            max_output_continuations=inference_policy.max_output_continuations,
            thinking_policy=inference_policy.thinking_policy,
            thinking_budget_tokens=inference_policy.thinking_budget_tokens,
            allowed_user_ids=ALLOWED_USER_IDS,
            log_event_fn=_log_event,
            extra_tools=agent_tools,
            extra_handlers=agent_handlers,
            budget_usd=inference_policy.budget_usd,
            finalization_tools=list(spec.finalization_tools),
            terminal_tools=list(spec.terminal_tools),
            on_progress=progress_cb,
            on_complete=_on_task_complete,
            context_provider=_agent_provider,
            verification_mode=verification_mode,
            verify_chat_fn=verify_chat_fn,
            verify_model_fn=verify_model_fn,
        )
        # Flush remaining progress buffer
        if progress_cb and hasattr(progress_cb, "flush"):
            await progress_cb.flush()

        # ── Orchestrator callback: report result to user via orchestrator ──
        result = result or {}
        is_subtask = result.get("is_subtask", False)
        if not is_subtask and result.get("status") in ("done", "failed"):
            # Skip the LLM-driven callback for self-delivering scheduled tasks
            # (e.g. diary): spec opts out AND the task came from the cron
            # scheduler. User-delegated calls to the same agent still get the
            # callback so the user hears back.
            task_meta = task.get("metadata") or {}
            if isinstance(task_meta, str):
                try:
                    task_meta = json.loads(task_meta)
                except Exception:
                    task_meta = {}
            task_origin = task_meta.get("origin") if isinstance(task_meta, dict) else None
            skip_callback = spec.skip_orchestrator_report and task_origin == "schedule"
            if not skip_callback:
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
        asyncio.create_task(
            _telegram_connectivity_watchdog(bot),
            name="telegram_connectivity_watchdog",
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
    await dp.start_polling(
        bot,
        drop_pending_updates=True,
        allowed_updates=_allowed_updates(),
        polling_timeout=TELEGRAM_POLLING_TIMEOUT_SECONDS,
        backoff_config=BackoffConfig(
            min_delay=TELEGRAM_BACKOFF_MIN_SECONDS,
            max_delay=TELEGRAM_BACKOFF_MAX_SECONDS,
            factor=TELEGRAM_BACKOFF_FACTOR,
            jitter=TELEGRAM_BACKOFF_JITTER,
        ),
        tasks_concurrency_limit=TELEGRAM_POLLING_CONCURRENCY_LIMIT,
    )
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
