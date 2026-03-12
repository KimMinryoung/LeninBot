"""telegram_bot.py — Telegram bot interface (aiogram 3.x).

Features:
- General messages → Claude Haiku API chat with per-user history
- /task <content> → Save to PostgreSQL queue, background worker processes, push on completion
- /status → Show last 5 tasks
- /clear → Reset chat history

Security: ALLOWED_USER_IDS whitelist, unauthorized users silently ignored.
"""

import os
import asyncio
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message
from aiogram.filters import Command
import anthropic

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Config ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}

# ── DB (own pool, independent from api.py) ───────────────────────────
_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=3,
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require",
        )
    return _pool


@contextmanager
def _get_conn():
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def _query(sql: str, params: tuple | None = None) -> list[dict]:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def _execute(sql: str, params: tuple | None = None) -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def _query_one(sql: str, params: tuple | None = None) -> dict | None:
    """Execute SQL with RETURNING and fetch one row."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None


def _ensure_table():
    """Create telegram_tasks table if not exists."""
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


# ── Claude client ────────────────────────────────────────────────────
_claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_CLAUDE_MAX_TOKENS = 4096

# Per-user chat history (in-memory, lost on restart)
_chat_history: dict[int, list[dict]] = {}

MAX_HISTORY_TURNS = 20  # 20 pairs = 40 messages


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


def _is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS


# ── Router & Handlers ───────────────────────────────────────────────
router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    await message.answer(
        "레닌봇 텔레그램 인터페이스입니다.\n\n"
        "메시지를 보내면 Claude 기반 대화를 할 수 있습니다.\n"
        "/task <내용> — 백그라운드 태스크 등록\n"
        "/status — 최근 태스크 상태 확인\n"
        "/clear — 대화 히스토리 초기화"
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    _chat_history.pop(message.from_user.id, None)
    await message.answer("대화 히스토리가 초기화되었습니다.")


@router.message(Command("task"))
async def cmd_task(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    content = (message.text or "").removeprefix("/task").strip()
    if not content:
        await message.answer("사용법: /task <내용>")
        return
    await asyncio.to_thread(
        _execute,
        "INSERT INTO telegram_tasks (user_id, content) VALUES (%s, %s)",
        (message.from_user.id, content),
    )
    await message.answer(f"태스크가 큐에 추가되었습니다:\n{content}")


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    rows = await asyncio.to_thread(
        _query,
        "SELECT id, content, status, created_at FROM telegram_tasks "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",
        (message.from_user.id,),
    )
    if not rows:
        await message.answer("등록된 태스크가 없습니다.")
        return
    status_icons = {"pending": "⏳", "processing": "🔄", "done": "✅", "failed": "❌"}
    lines = []
    for r in rows:
        icon = status_icons.get(r["status"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M")
        preview = r["content"][:50]
        lines.append(f"{icon} [{r['id']}] {preview}\n   상태: {r['status']} | {ts}")
    await message.answer("\n\n".join(lines))


@router.message(F.text)
async def handle_message(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    user_id = message.from_user.id
    user_text = message.text

    # Build / retrieve history
    if user_id not in _chat_history:
        _chat_history[user_id] = []
    history = _chat_history[user_id]
    history.append({"role": "user", "content": user_text})

    # Trim to last N turns
    if len(history) > MAX_HISTORY_TURNS * 2:
        history = history[-(MAX_HISTORY_TURNS * 2):]
        _chat_history[user_id] = history

    try:
        response = await _claude.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=_CLAUDE_MAX_TOKENS,
            messages=history,
        )
        reply = response.content[0].text
    except Exception as e:
        logger.error("Claude API error: %s", e)
        reply = f"오류가 발생했습니다: {e}"

    history.append({"role": "assistant", "content": reply})

    for chunk in _split_message(reply):
        await message.answer(chunk)


# ── Background Task Worker ───────────────────────────────────────────
async def _process_task(bot: Bot, task: dict):
    """Process a single task via Claude and push result to user."""
    task_id = task["id"]
    user_id = task["user_id"]
    content = task["content"]

    try:
        response = await _claude.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=_CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
        )
        result = response.content[0].text

        await asyncio.to_thread(
            _execute,
            "UPDATE telegram_tasks SET status = 'done', result = %s, "
            "completed_at = NOW() WHERE id = %s",
            (result, task_id),
        )

        for chunk in _split_message(f"✅ 태스크 [{task_id}] 완료:\n\n{result}"):
            await bot.send_message(chat_id=user_id, text=chunk)

    except Exception as e:
        logger.error("Task %d failed: %s", task_id, e)
        await asyncio.to_thread(
            _execute,
            "UPDATE telegram_tasks SET status = 'failed', result = %s, "
            "completed_at = NOW() WHERE id = %s",
            (str(e), task_id),
        )
        await bot.send_message(
            chat_id=user_id,
            text=f"❌ 태스크 [{task_id}] 실패:\n{e}",
        )


async def _task_worker(bot: Bot):
    """Poll DB for pending tasks and process them one at a time."""
    logger.info("Task worker started")
    while True:
        try:
            task = await asyncio.to_thread(
                _query_one,
                "UPDATE telegram_tasks SET status = 'processing' "
                "WHERE id = (SELECT id FROM telegram_tasks WHERE status = 'pending' "
                "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id, user_id, content",
            )
            if task:
                await _process_task(bot, task)
            else:
                await asyncio.sleep(5)
        except Exception as e:
            logger.error("Worker loop error: %s", e)
            await asyncio.sleep(10)


# ── Entry Point ──────────────────────────────────────────────────────
async def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if not ALLOWED_USER_IDS:
        raise RuntimeError("ALLOWED_USER_IDS is not set")

    # Ensure task table exists
    await asyncio.to_thread(_ensure_table)

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    # Start background worker
    asyncio.create_task(_task_worker(bot))

    logger.info("Bot starting (allowed users: %s)", ALLOWED_USER_IDS)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
