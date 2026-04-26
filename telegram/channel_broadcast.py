"""Telegram channel broadcast helpers.

The bot cannot create Telegram channels through the Bot API. Operators create a
channel manually, add the bot as an administrator with post permission, then
store the target here.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from secrets_loader import get_secret

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
CONFIG_PATH = PROJECT_ROOT / "data" / "channel_config.json"
MESSAGE_CHUNK_LIMIT = 3800

current_autonomous_project_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_autonomous_project_id",
    default=None,
)


@dataclass
class BroadcastResult:
    ok: bool
    message: str
    sent_count: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_config() -> dict[str, Any]:
    return {
        "chat_id": "",
        "title": "",
        "username": "",
        "enabled": True,
        "created_at": "",
        "updated_at": "",
        "last_error": "",
        "requested_private_channel_name": "",
    }


def load_channel_config() -> dict[str, Any]:
    env_channel_ref = normalize_channel_ref(
        get_secret("CHANNEL_ID", "")
        or get_secret("CHANNEL_USERNAME", "")
        or get_secret("TELEGRAM_CHANNEL_ID", "")
        or get_secret("TELEGRAM_CHANNEL_USERNAME", "")
        or ""
    )
    if not CONFIG_PATH.exists():
        cfg = _default_config()
    else:
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("channel config load failed: %s", e)
            data = {}
        cfg = _default_config()
        if isinstance(data, dict):
            cfg.update(data)

    if env_channel_ref:
        cfg["chat_id"] = env_channel_ref
        cfg["username"] = env_channel_ref if env_channel_ref.startswith("@") else ""
    return cfg


def save_channel_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = _default_config()
    cfg.update(config)
    if not cfg.get("created_at"):
        cfg["created_at"] = _now_iso()
    cfg["updated_at"] = _now_iso()
    fd, tmp_path = tempfile.mkstemp(prefix=".channel_config.", suffix=".json", dir=str(CONFIG_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, CONFIG_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def normalize_channel_ref(raw: str) -> str:
    value = (raw or "").strip()
    if value.startswith("https://t.me/"):
        value = value.removeprefix("https://t.me/")
    elif value.startswith("http://t.me/"):
        value = value.removeprefix("http://t.me/")
    elif value.startswith("t.me/"):
        value = value.removeprefix("t.me/")
    value = value.strip().split()[0] if value.strip() else ""
    if not value:
        return ""
    if value.startswith("@") or value.startswith("-100"):
        return value
    if value.lstrip("-").isdigit():
        return value
    return f"@{value}"


def configured_chat_id() -> str:
    return str(load_channel_config().get("chat_id") or "").strip()


def _format_chat(chat: Any) -> dict[str, str]:
    username = getattr(chat, "username", "") or ""
    chat_type = getattr(chat, "type", "") or ""
    if hasattr(chat_type, "value"):
        chat_type = chat_type.value
    return {
        "id": str(getattr(chat, "id", "") or ""),
        "title": getattr(chat, "title", "") or "",
        "username": f"@{username}" if username else "",
        "type": str(chat_type),
    }


async def validate_channel(bot: Bot, channel_ref: str) -> tuple[bool, str, dict[str, str]]:
    target = normalize_channel_ref(channel_ref)
    if not target:
        return False, "채널 핸들 또는 chat_id가 비어 있습니다.", {}
    try:
        chat = await bot.get_chat(target)
        info = _format_chat(chat)
        if info["type"] != "channel":
            return False, f"대상이 채널이 아닙니다(type={info['type'] or 'unknown'}).", info

        me = await bot.get_me()
        member = await bot.get_chat_member(chat.id, me.id)
        status_value = getattr(member, "status", "") or ""
        status = str(status_value.value if hasattr(status_value, "value") else status_value)
        can_post = bool(getattr(member, "can_post_messages", False))
        if status != "administrator" or not can_post:
            return (
                False,
                "봇이 이 채널의 게시 권한 있는 관리자가 아닙니다. 채널 설정에서 봇을 관리자로 추가하고 '메시지 게시' 권한을 켜십시오.",
                info,
            )
        return True, "ok", info
    except TelegramForbiddenError:
        return False, "봇이 채널에 접근할 수 없습니다. 채널에 봇을 관리자로 추가해야 합니다.", {}
    except TelegramBadRequest as e:
        return False, f"채널을 찾거나 검증할 수 없습니다: {e.message}", {}
    except Exception as e:
        return False, f"채널 검증 실패: {e}", {}


def chunk_message(text: str, limit: int = MESSAGE_CHUNK_LIMIT) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    chunks: list[str] = []
    while len(s) > limit:
        split_at = s.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(s[:split_at].strip())
        s = s[split_at:].strip()
    if s:
        chunks.append(s)
    return chunks


async def broadcast_to_configured_channel(
    bot: Bot,
    text: str,
    *,
    parse_mode: str | None = None,
    disable_web_page_preview: bool = False,
) -> BroadcastResult:
    cfg = load_channel_config()
    if not cfg.get("enabled", True):
        return BroadcastResult(False, "채널 브로드캐스트가 비활성화되어 있습니다.")
    chat_id = str(cfg.get("chat_id") or "").strip()
    if not chat_id:
        return BroadcastResult(False, "브로드캐스트 대상 채널이 설정되지 않았습니다. `/channel_set <@채널핸들 또는 -100...chat_id>`를 먼저 실행하십시오.")

    parts = chunk_message(text)
    if not parts:
        return BroadcastResult(False, "전송할 메시지가 비어 있습니다.")

    sent = 0
    try:
        for part in parts:
            await bot.send_message(
                chat_id=chat_id,
                text=part,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
            sent += 1
        if cfg.get("last_error"):
            cfg["last_error"] = ""
            save_channel_config(cfg)
        return BroadcastResult(True, f"채널 전송 완료 ({sent}개 메시지).", sent)
    except TelegramForbiddenError:
        msg = "전송 실패: 봇이 채널 관리자에서 제거되었거나 게시 권한이 없습니다."
    except TelegramBadRequest as e:
        msg = f"전송 실패: {e.message}"
    except Exception as e:
        msg = f"전송 실패: {e}"

    cfg["last_error"] = msg
    save_channel_config(cfg)
    logger.warning("channel broadcast failed: %s", msg)
    return BroadcastResult(False, msg, sent)


async def broadcast_with_token(
    text: str,
    *,
    parse_mode: str | None = None,
    disable_web_page_preview: bool = False,
) -> BroadcastResult:
    token = get_secret("TELEGRAM_BOT_TOKEN", "") or ""
    if not token:
        return BroadcastResult(False, "TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
    bot = Bot(token=token)
    try:
        return await broadcast_to_configured_channel(
            bot,
            text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
    finally:
        await bot.session.close()


def broadcast_with_token_sync(
    text: str,
    *,
    parse_mode: str | None = None,
    disable_web_page_preview: bool = False,
) -> BroadcastResult:
    async def _run() -> BroadcastResult:
        return await broadcast_with_token(
            text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    holder: dict[str, BroadcastResult | BaseException] = {}

    def _thread_main() -> None:
        try:
            holder["result"] = asyncio.run(_run())
        except BaseException as e:
            holder["error"] = e

    thread = Thread(target=_thread_main, daemon=True)
    thread.start()
    thread.join(timeout=20)
    if "error" in holder:
        raise holder["error"]  # type: ignore[misc]
    if "result" in holder:
        return holder["result"]  # type: ignore[return-value]
    return BroadcastResult(False, "전송 실패: Telegram 응답 시간 초과.")


def should_broadcast_diary() -> bool:
    return os.getenv("TELEGRAM_BROADCAST_DIARY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def should_broadcast_moltbook() -> bool:
    return os.getenv("TELEGRAM_BROADCAST_MOLTBOOK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def should_broadcast_autonomous_project(project_id: int | None) -> bool:
    if project_id is None:
        return False
    raw = os.getenv("TELEGRAM_BROADCAST_AUTONOMOUS_PROJECT_IDS", "all").strip()
    if raw.lower() in {"", "none", "off", "false", "0"}:
        return False
    if raw.lower() in {"all", "*"}:
        return True
    allowed = {p.strip() for p in raw.split(",") if p.strip()}
    return str(project_id) in allowed


def should_broadcast_site_publication(project_id: int | None = None) -> bool:
    if os.getenv("TELEGRAM_BROADCAST_SITE_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if project_id is None:
        return True
    return should_broadcast_autonomous_project(project_id)


async def maybe_broadcast_autonomous_publication(
    *,
    title: str,
    url: str,
    body: str = "",
    source: str = "cyber-lenin.com",
) -> BroadcastResult:
    project_id = current_autonomous_project_id.get()
    if not should_broadcast_site_publication(project_id):
        return BroadcastResult(False, "site publication broadcast disabled")

    excerpt = " ".join((body or "").split())
    if len(excerpt) > 2400:
        excerpt = excerpt[:2399].rstrip() + "…"
    text = f"[사이버-레닌 새 글]\n{title.strip()}\n{source}\n{url.strip()}"
    if excerpt:
        text += f"\n\n{excerpt}"
    return await broadcast_with_token(text)
