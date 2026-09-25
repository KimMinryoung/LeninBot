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
from shared import KST
from db import query as _query, execute as _execute, query_one as _query_one, get_conn as _get_conn
from psycopg2.extras import RealDictCursor

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import CallbackQuery, ChatMemberUpdated, Message
from aiogram.utils.backoff import BackoffConfig

from secrets_loader import get_secret

# Extracted modules
from bot_config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    _claude,
    _openai_client,
    _deepseek_client,
    _kimi_client,
    _CLAUDE_MAX_TOKENS_TASK,
    _config,
    _save_config,
    _CONFIG_DEFAULTS,
    _CONFIG_META,
    _resolved_models,
    _tier_to_display,
    _get_model,
    _get_model_task,
    _get_model_light,
    _get_task_provider,
    get_current_model_selection,
    get_task_verification_mode,
    _extract_text,
)
from llm.runtime_profile import resolve_runtime_profile
from telegram.schema import hydrate_summary_state
from telegram.task_reporting import report_for_callback, RESULT_RELAY_GUIDANCE
from telegram._send_utils import make_progress_callback, split_message
from ops.logs import log_event
from llm.json_utils import extract_json_object
from llm.claude_loop import dedupe_tools_by_name
from llm.provider_registry import CHAT_PROVIDERS
from telegram.bot_api10 import TelegramBotApi10Client, TelegramBotApiError
from telegram.tasks import (
    process_task, system_monitor,
    task_worker, schedule_worker, check_deploy_meta,
    recover_processing_tasks_on_startup,
    checkpoint_task_on_shutdown, persist_task_restart_state,
    _delegate_to_browser_worker, check_browser_worker_alive,
    _load_task_metadata,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

_runtime_state: dict = {"active_task_ids": set()}

# Per-coroutine task context — allows concurrent tasks to know their own task_id.
# Defined in llm.runtime_context so non-bot processes can share the same object.
from llm.runtime_context import current_task_ctx

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
_log_event = log_event


# ── Light LLM — llm/call_registry 경유 (config/llm_call_sites.json 관리) ──
from llm.call_registry import generate as _registry_generate


async def _light_generate(feature: str, prompt: str) -> str | None:
    """Run a registered one-shot generation (chunk_summary, conversation_reflection).

    Model/options come from config/llm_call_sites.json (hot-reloaded).
    Returns None on any failure — callers keep their own fallbacks.
    """
    return await _registry_generate(feature, prompt)


from llm.runtime_context import (
    build_runtime_prelude as _build_runtime_prelude,
    current_datetime_str as _current_datetime_str,
    format_current_model_context as _format_current_model_context,
    join_context_blocks as _join_context_blocks,
)


# ── System Alerts (injected into system prompt) ─────────────────────

from telegram.chat_runtime import (
    add_system_alert as _add_system_alert,
    clear_system_alert as _clear_system_alert,
    _owner_run_context,
    chat_with_tools as _chat_with_tools,
    get_model_for_agent as _get_model_for_agent,
    make_provider_chat_fn as _make_provider_chat_fn,
)

# Each alert: (monotonic_timestamp, formatted_string)


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
    from ops.paths import PROJECT_ROOT
    venv = str(PROJECT_ROOT / "venv")
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



# Fully static system-layer tail. Only content that does not change between
# turns belongs here (skills catalog). Per-turn runtime state — current time,
# current model, mission, memories, alerts — is injected via message content,
# not the system prompt, so prompt caching stays effective.


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

    Summaries are attached as attributed historical metadata (not conversation
    pairs). Raw messages retain their order; summary metadata travels beside
    the current request.
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

    # Preserve summaries as attributed metadata without creating dialogue turns.
    summary_records = []
    if useful_summaries:
        from llm.execution_context import context_record
        summary_lines = []
        for s in useful_summaries:
            summary_lines.append(
                f"• (msgs #{s['chunk_start_id']}~#{s['chunk_end_id']}): {s['summary']}"
            )
        summary_records = [context_record(
            "conversation_summary", "chat_history_summaries", "\n".join(summary_lines),
            scope=f"telegram:{user_id}", temporal_scope="historical",
            reference="original messages via read_self(chat_logs, chat_source=telegram)",
            coverage="model-compressed dialogue; claims and promises are not receipts",
        )]

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

    if raw_rows and context:
        from llm.execution_context import RUNTIME_EVENTS_KEY
        from telegram.execution_history import recent_execution_events
        try:
            context[-1][RUNTIME_EVENTS_KEY] = recent_execution_events(
                user_id, raw_rows[0]["created_at"],
            )
        except Exception as exc:
            logger.warning("Telegram execution history unavailable: %s", exc)
            context[-1][RUNTIME_EVENTS_KEY] = [{
                "source": "tool_audit_log", "coverage": "unavailable",
            }]

    if summary_records:
        from llm.execution_context import attach_context
        context = attach_context(context, summary_records)
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
            "Keep only: 1) user requests and explicit corrections, 2) assistant claims/proposals "
            "attributed as claims, 3) pending requests and unresolved items. A promise to act "
            "is not an execution receipt; never turn 'I will delegate' into 'delegated'. "
            "Preserve speaker, proper nouns, numbers, dates, and "
            "specific decisions. Korean, 500 characters max.\n\n"
            + conversation_text
        )

        summary = await _light_generate("chunk_summary", summary_prompt)
        if not summary or _summary_is_contaminated(summary):
            logger.warning(
                "Chunk summary %s; using extractive fallback user=%d msgs=#%d~#%d",
                "generation failed" if not summary else "contaminated",
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
_split_message = split_message


# ── Progress Callback (live tool progress via Telegram) ──────────────

_bot_instance: Bot | None = None  # set in bot_main()


def _make_progress_callback(chat_id: int):
    return make_progress_callback(lambda: _bot_instance, chat_id)


def _is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS


# ── _chat_with_tools helpers: profile/toolset, context, provider dispatch ──


# ── Thin wrapper: _chat_with_tools (injects module-level dependencies) ──


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
    "format_autonomous_status": _format_autonomous_status,
    "join_context_blocks": _join_context_blocks,
    "add_system_alert": _add_system_alert,
    "claude_client": _claude,
    "openai_client": _openai_client,
    "deepseek_client": _deepseek_client,
    "kimi_client": _kimi_client,
    "extract_text": _extract_text,
    "light_generate": _light_generate,
    "get_model_light": _get_model_light,
    "maybe_summarize_chunk": _maybe_summarize_chunk,
})


# ── Entry Point ──────────────────────────────────────────────────────

# ── Per-agent dispatch helpers ────────────────────────────────────────
# These resolve which chat_with_tools implementation + which model an agent
# spec runs against. Keeping them at module level (rather than as inline
# closures inside bot_main) keeps the dispatch block in bot_main short and
# makes the routing testable in isolation.


def _make_moon_chat_fn(spec):
    """Build a chat_fn that targets the local llama-server (MOON PC).

    Falls back to the standard Claude/OpenAI chat fn if MOON is unhealthy
    so the agent run still succeeds.
    """
    from llm.openai_tool_loop import chat_with_tools as _moon_loop
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
        agent_name=None, runtime_kind=None, user_id=None, session_id=None,
        request_id=None, parent_request_id=None, scope_type=None, scope_id=None,
    ):
        from tool_gateway.security import caller_scope
        ctx = _owner_run_context(
            "agent", agent_name or spec.name,
            request_id=request_id, user_id=user_id, task_id=task_id,
            session_id=session_id, parent_request_id=parent_request_id,
            scope_type=scope_type, scope_id=scope_id,
        )
        with caller_scope(ctx):
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
    from llm.codex_exec_loop import chat_with_tools as _codex_loop

    async def _codex_chat_fn(
        messages, max_rounds=None, system_prompt=None, model=None,
        max_tokens=None, max_input_tokens=None, max_output_continuations=0,
        thinking_policy="tool_loop", thinking_budget_tokens=8192, budget_usd=None, extra_tools=None,
        extra_handlers=None, on_progress=None, budget_tracker=None,
        task_id=None, finalization_tools=None, terminal_tools=None,
        agent_name=None, runtime_kind=None, user_id=None, session_id=None,
        request_id=None, parent_request_id=None, scope_type=None, scope_id=None,
    ):
        from tool_gateway.security import caller_scope
        ctx = _owner_run_context(
            "agent", agent_name or spec.name,
            request_id=request_id, user_id=user_id, task_id=task_id,
            session_id=session_id, parent_request_id=parent_request_id,
            scope_type=scope_type, scope_id=scope_id,
        )
        with caller_scope(ctx):
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


_extract_json_object = extract_json_object


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
        if (status == "done" and not was_interrupted
                and (result.get("verification") or {}).get("status") != "failed"):
            from mail_runtime import store as mail_store
            from mail_runtime.delivery import deliver as deliver_mail_briefing
            mail_items = await asyncio.to_thread(mail_store.briefing_items, task_id, chat_id)
            if mail_items:
                try:
                    await deliver_mail_briefing(b, task_id, chat_id, mail_items,
                                               _persist_assistant_turn_after_send)
                    await asyncio.to_thread(_save_system_event, chat_id, "task_report",
                                            f"task #{task_id} mail briefing delivery checked")
                except Exception:
                    logger.exception("Mail briefing delivery failed for task #%d; unsent items remain unbriefed", task_id)
                return
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
                    f"Report goal completion separately from execution quality. Blocked or unverified work is not complete; relay the remaining requirements and retry conditions."
                )

            prompt = (
                f"[TASK REPORT] Task #{task_id} [{agent_type}] execution ended{' (interrupted)' if was_interrupted else ''}; goal completion is not implied\n\n"
                f"Original request:\n{_truncate_for_prompt(task.get('content', ''), 1000)}\n\n"
                f"Agent's report (claims, not independent verification):\n{report_for_callback(task_id, report)}"
                f"{tool_log_section}"
                f"{verification_section}"
                f"{interrupted_note}\n\n"
                f"## Your role\n"
                f"1. Relay the results to the user concisely, covering only key points. Do not use markdown formatting.\n"
                f"   {RESULT_RELAY_GUIDANCE}\n"
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
        from llm.execution_context import RUNTIME_EVENTS_KEY
        history.append({
            "role": "user",
            "content": (
                "Relay the runtime-supplied task outcome to the user concisely, without markdown. "
                "Distinguish recorded task status from the agent's claims and goal completion. "
                + RESULT_RELAY_GUIDANCE + " "
                + "Only delegate follow-up when the agent was interrupted by budget/turn limits, "
                "further work can improve the result, and the cause is not an external blocker "
                "such as permissions, CAPTCHA or API failure. Otherwise relay the outcome."
                + mission_close_hint
            ),
            RUNTIME_EVENTS_KEY: [{
                "source": "telegram_task_callback",
                "task_id": task_id,
                "agent": agent_type,
                "recorded_status": status,
                "callback_context": prompt,
            }],
        })

        # Run orchestrator — budget enough for response + optional redelegate call
        from runtime_tools.registry import build_mission_handler
        reply = await _chat_with_tools(
            history,
            budget_usd=0.15,
            max_rounds=5,
            extra_handlers={"mission": build_mission_handler(chat_id)},
            user_id=str(chat_id),
            session_id=f"telegram:{chat_id}",
            scope_type="telegram_task_callback",
            scope_id=str(task_id),
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

    from agents import get_agent
    try:
        spec = get_agent(agent_type)
    except (ValueError, ImportError):
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
    elif spec.provider == "codex":
        chosen_chat_fn = _make_codex_chat_fn(spec)
    elif spec.provider in CHAT_PROVIDERS:
        chosen_chat_fn = _make_provider_chat_fn(spec.provider)
    elif task_provider in CHAT_PROVIDERS:
        chosen_chat_fn = _make_provider_chat_fn(task_provider)
    else:
        chosen_chat_fn = _chat_with_tools

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
    if agent_type == "hub_curator" and "publish_hub_curation" in agent_handlers:
        from telegram.curate import make_guarded_publish_handler

        agent_handlers = dict(agent_handlers)
        agent_handlers["publish_hub_curation"] = make_guarded_publish_handler(
            agent_handlers["publish_hub_curation"], task
        )

    # ── Post-hoc verification (Critic) routing ──────────────────
    # The verifier runs on the low tier of a standard provider so the
    # critique stays cheap. Codex/moon executors are verified by the task
    # provider — an independent judge for handed-off work.
    verification_mode = get_task_verification_mode()
    verify_chat_fn = None
    verify_model_fn = None
    if verification_mode in ("shadow", "enforce"):
        if spec.provider in CHAT_PROVIDERS:
            verify_provider = spec.provider
        elif task_provider in CHAT_PROVIDERS:
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
        max_tokens_task=inference_policy.max_output_tokens,
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
    if agent_type == "hub_curator" and result.get("status") in ("done", "failed"):
        # /curate outcome is judged by the hub_curations row, not by the
        # agent's text, and needs no LLM turn to relay.
        from telegram.curate import report_curation_outcome

        target_uid = task["user_id"] if task["user_id"] != 0 else OWNER_USER_ID
        if target_uid:
            await report_curation_outcome(
                b, task, result, chat_id=target_uid, save_system_event=_save_system_event
            )
    elif not is_subtask and result.get("status") in ("done", "failed"):
        # Skip the LLM-driven callback for self-delivering scheduled tasks
        # (e.g. diary): spec opts out AND the task came from the cron
        # scheduler. User-delegated calls to the same agent still get the
        # callback so the user hears back.
        task_origin = _load_task_metadata(task).get("origin")
        skip_callback = spec.skip_orchestrator_report and task_origin == "schedule"
        if not skip_callback:
            target_uid = task["user_id"] if task["user_id"] != 0 else OWNER_USER_ID
            if target_uid:
                await _orchestrator_report_task(b, task, result, target_uid)


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

    # Register commands for Telegram "/" autocomplete menu
    from aiogram.types import BotCommand
    from telegram.commands import bot_menu_commands
    await bot.set_my_commands([
        BotCommand(command=command, description=description)
        for command, description in bot_menu_commands()
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

    # Graceful shutdown: notify + stop polling cleanly when SIGTERM received (Render deploy)
    import signal

    def _handle_sigterm(*_):
        logger.info("SIGTERM received — stopping polling gracefully")
        # Schedule shutdown notification before stopping
        async def _shutdown_notify_and_checkpoint():
            # Merge in-memory and Redis active task sets for comprehensive checkpoint
            active_ids = set(_runtime_state.get("active_task_ids", set()))
            try:
                from memory_store.redis_state import get_active_task_ids
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
