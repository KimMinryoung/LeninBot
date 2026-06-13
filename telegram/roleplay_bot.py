"""roleplay_bot.py — Standalone DeepSeek roleplay companion (independent of Cyber-Lenin).

A lightweight second Telegram bot for free-form character roleplay. It deliberately
reuses only the verified low-level building blocks — DB pool, secret loading, the
DeepSeek client, the OpenAI-compatible tool loop, and a curated subset of read-only
knowledge tools — without dragging in Cyber-Lenin's agent stack (tasks, missions,
autonomous loop, KG writes, identity prompt).

Sessions live in their own tables (``roleplay_chat_history`` / ``roleplay_clear_markers``)
so this bot's conversation is fully isolated from the Cyber-Lenin Telegram bot.

Run: ``python -m telegram.roleplay_bot`` (see systemd/leninbot-roleplay.service).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message

from secrets_loader import get_secret
from db import query as _query, execute as _execute
from bot_config import _deepseek_anthropic_client, _resolve_deepseek_model
from claude_loop import chat_with_tools
from runtime_tools.registry import TOOLS, TOOL_HANDLERS
from identity.prompts import EXTERNAL_SOURCE_RULE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] roleplay: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
ROLEPLAY_BOT_TOKEN = get_secret("ROLEPLAY_BOT_TOKEN", "") or ""

# Owner gate: dedicated allowlist, falling back to the main bot's allowlist.
_ALLOWED_RAW = os.getenv("ROLEPLAY_ALLOWED_USER_IDS", "") or os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip()) for uid in _ALLOWED_RAW.split(",") if uid.strip()
}

PERSONA_PATH = Path(__file__).resolve().parent.parent / "identity" / "roleplay_persona.md"

# Roleplay tuning: flash + thinking ON for answer quality. We call DeepSeek over
# its Anthropic-compatible endpoint via claude_loop, which keeps reasoning as
# replay-only "thinking" blocks — used for quality but excluded from the
# user-facing reply (see _REPLAY_ONLY_BLOCK_TYPES in claude_loop). This is what
# separates the inner monologue from the final answer; the OpenAI path instead
# prepends reasoning to the reply, which is why it leaked.
ROLEPLAY_MODEL = _resolve_deepseek_model("deepseek_flash")  # "deepseek-v4-flash"
ROLEPLAY_MAX_TOKENS = int(os.getenv("ROLEPLAY_MAX_TOKENS", "4096"))
ROLEPLAY_MAX_ROUNDS = int(os.getenv("ROLEPLAY_MAX_ROUNDS", "8"))
ROLEPLAY_BUDGET_USD = float(os.getenv("ROLEPLAY_BUDGET_USD", "0.50"))
HISTORY_CAP = int(os.getenv("ROLEPLAY_HISTORY_CAP", "40"))  # messages kept in context

# Curated read-only toolset (no task execution, no KG writes).
_TOOL_NAMES = ["vector_search", "knowledge_graph_search", "web_search", "fetch_url"]


def _select_tools() -> tuple[list[dict], dict]:
    tools = [t for t in TOOLS if t.get("name") in _TOOL_NAMES]
    missing = set(_TOOL_NAMES) - {t["name"] for t in tools}
    handlers = {n: TOOL_HANDLERS[n] for n in _TOOL_NAMES if n in TOOL_HANDLERS}
    missing |= set(_TOOL_NAMES) - set(handlers)
    if missing:
        logger.warning("roleplay toolset missing definitions/handlers: %s", sorted(missing))
    return tools, handlers


RP_TOOLS, RP_HANDLERS = _select_tools()


# ── Persistence (own tables → session isolation) ─────────────────────
def ensure_tables() -> None:
    _execute("""
        CREATE TABLE IF NOT EXISTS roleplay_chat_history (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            role        VARCHAR(10) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_rp_history_user
        ON roleplay_chat_history (user_id, id DESC)
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS roleplay_clear_markers (
            user_id        BIGINT PRIMARY KEY,
            clear_after_id BIGINT NOT NULL DEFAULT 0
        )
    """)


def _clear_after_id(user_id: int) -> int:
    rows = _query(
        "SELECT clear_after_id FROM roleplay_clear_markers WHERE user_id = %s",
        (user_id,),
    )
    return int(rows[0]["clear_after_id"]) if rows else 0


def save_message(user_id: int, role: str, content: str) -> None:
    _execute(
        "INSERT INTO roleplay_chat_history (user_id, role, content) VALUES (%s, %s, %s)",
        (user_id, role, content),
    )


def load_history(user_id: int) -> list[dict]:
    """Recent turns after the last /new marker, oldest-first, capped at HISTORY_CAP."""
    min_id = _clear_after_id(user_id)
    rows = _query(
        "SELECT role, content FROM roleplay_chat_history "
        "WHERE user_id = %s AND id > %s ORDER BY id DESC LIMIT %s",
        (user_id, min_id, HISTORY_CAP),
    )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def reset_session(user_id: int) -> None:
    rows = _query(
        "SELECT COALESCE(MAX(id), 0) AS max_id FROM roleplay_chat_history WHERE user_id = %s",
        (user_id,),
    )
    max_id = int(rows[0]["max_id"]) if rows else 0
    _execute(
        "INSERT INTO roleplay_clear_markers (user_id, clear_after_id) VALUES (%s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET clear_after_id = EXCLUDED.clear_after_id",
        (user_id, max_id),
    )


# ── Persona / system prompt (hot-reloaded every turn) ────────────────
def build_system_prompt() -> str:
    try:
        persona = PERSONA_PATH.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.error("persona file unreadable (%s): %s", PERSONA_PATH, e)
        persona = "너는 사용자와 자유롭게 대화하는 친근한 캐릭터다. 일관된 말투를 유지한다."
    return persona + "\n\n" + EXTERNAL_SOURCE_RULE


# ── Telegram plumbing ────────────────────────────────────────────────
def _is_allowed(user_id: int | None) -> bool:
    return user_id is not None and user_id in ALLOWED_USER_IDS


class OwnerOnlyMiddleware(BaseMiddleware):
    """Drop messages from anyone but the configured owner(s)."""

    async def __call__(self, handler, event, data):
        user_id = getattr(getattr(event, "from_user", None), "id", None)
        if _is_allowed(user_id):
            return await handler(event, data)
        logger.info("blocked unauthorized roleplay message user_id=%s", user_id)
        return None


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


def _make_progress_callback(bot: Bot, chat_id: int):
    """on_progress callback that streams reasoning/tool steps as separate messages.

    Mirrors the Cyber-Lenin bot: buffer events per round, flush one code-block
    message per round so the final answer stays clean prose. Expose ``.flush``
    for the trailing round after the loop returns.
    """
    _buf: list[str] = []
    _current_round = [0]

    async def _flush():
        if not _buf:
            return
        text = "\n".join(_buf)
        _buf.clear()
        try:
            for chunk in _split_message(f"```\n{text}\n```"):
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
        except Exception as e:
            logger.debug("progress send failed: %s", e)

    async def _on_progress(event: str, detail: str):
        round_num = 0
        if detail.startswith("["):
            try:
                round_num = int(detail[1:detail.index("]")])
            except (ValueError, IndexError):
                pass
        if round_num > _current_round[0] and _current_round[0] > 0:
            await _flush()
        if round_num > 0:
            _current_round[0] = round_num

        # Stream ONLY tool steps. The model's in-character prose arrives as
        # "thinking"/"text_delta" but is also folded into the final reply by
        # the loop — streaming it here would duplicate it. Budget ("💰") is
        # mechanics noise. So a plain chat turn sends no progress at all (just
        # the reply); a tool turn shows the 🔧 steps, then the clean answer.
        if event in ("tool_call", "tool_result"):
            _buf.append(detail)

    _on_progress.flush = _flush
    return _on_progress


router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "역할극 봇이 준비됐어. 그냥 말을 걸어줘.\n"
        "/new 로 대화를 처음부터 다시 시작할 수 있고, /help 로 사용법을 볼 수 있어."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "이 봇은 캐릭터와 1:1 역할극을 하는 독립 봇이야 (Cyber-Lenin과 별개).\n"
        "• 그냥 메시지를 보내면 캐릭터가 답해.\n"
        "• /new — 현재 대화 맥락을 끊고 새 세션 시작\n"
        "• 캐릭터 설정은 identity/roleplay_persona.md 파일에서 편집 (재시작 불필요)\n"
        f"• 모델: {ROLEPLAY_MODEL} (thinking on, 추론은 답변에 미포함)"
    )


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    await asyncio.to_thread(reset_session, message.from_user.id)
    await message.answer("새 대화를 시작할게. (이전 맥락은 잊었어)")


@router.message(F.text)
async def handle_message(message: Message) -> None:
    user_id = message.from_user.id
    user_text = message.text or ""
    if not user_text.strip():
        return

    await asyncio.to_thread(save_message, user_id, "user", user_text)
    history = await asyncio.to_thread(load_history, user_id)

    try:
        await message.bot.send_chat_action(message.chat.id, "typing")
    except Exception:
        pass

    # Stream reasoning/tool steps as separate messages; keep the final reply clean.
    progress_cb = _make_progress_callback(message.bot, message.chat.id)
    try:
        reply = await chat_with_tools(
            history,
            client=_deepseek_anthropic_client,
            model=ROLEPLAY_MODEL,
            tools=RP_TOOLS,
            tool_handlers=RP_HANDLERS,
            system_prompt=build_system_prompt(),
            max_rounds=ROLEPLAY_MAX_ROUNDS,
            max_tokens=ROLEPLAY_MAX_TOKENS,
            budget_usd=ROLEPLAY_BUDGET_USD,
            on_progress=progress_cb,
            agent_name="roleplay",
            thinking={"type": "enabled"},
            output_config={"effort": "high"},
        )
    except Exception as e:
        logger.exception("roleplay turn failed: %s", e)
        await message.answer("…(잠깐 말이 막혔어. 다시 한 번 말해줄래?)")
        return
    finally:
        await progress_cb.flush()

    await asyncio.to_thread(save_message, user_id, "assistant", reply)
    for chunk in _split_message(reply):
        await message.answer(chunk)


async def bot_main() -> None:
    if not ROLEPLAY_BOT_TOKEN:
        raise RuntimeError("ROLEPLAY_BOT_TOKEN is not set (.env or systemd credential).")
    if not ALLOWED_USER_IDS:
        raise RuntimeError("No allowed users: set ROLEPLAY_ALLOWED_USER_IDS or ALLOWED_USER_IDS.")
    if _deepseek_anthropic_client is None:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured; roleplay bot needs DeepSeek.")

    await asyncio.to_thread(ensure_tables)

    session = AiohttpSession()
    bot = Bot(token=ROLEPLAY_BOT_TOKEN, session=session)
    dp = Dispatcher()
    dp.message.middleware(OwnerOnlyMiddleware())
    dp.include_router(router)

    await bot.set_my_commands([
        BotCommand(command="new", description="새 대화 시작 (맥락 초기화)"),
        BotCommand(command="help", description="사용법"),
    ])

    me = await bot.get_me()
    logger.info(
        "roleplay bot @%s up — model=%s tools=%s owners=%s",
        me.username, ROLEPLAY_MODEL, [t["name"] for t in RP_TOOLS], sorted(ALLOWED_USER_IDS),
    )

    try:
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(bot_main())
