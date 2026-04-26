"""Telegram channel broadcast helpers.

The bot cannot create Telegram channels through the Bot API. Operators create a
channel manually, add the bot as an administrator with post permission, then
store the target here.
"""

from __future__ import annotations

import asyncio
import argparse
import contextvars
import html
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secrets_loader import get_secret

logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "data" / "channel_config.json"
MESSAGE_CHUNK_LIMIT = 3800
PUBLICATION_INTRO_SENTENCES = 3
PUBLICATION_INTRO_LIMIT = 700

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


def _plain_publication_text(body: str, title: str = "") -> str:
    text = body or ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"[*_~]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    title = re.sub(r"\s+", " ", title or "").strip()
    if title and text.startswith(title):
        text = text[len(title):].lstrip(" :-—-")
    return text


def _truncate_intro(text: str, limit: int = PUBLICATION_INTRO_LIMIT) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit + 1]
    split_at = max(cut.rfind(" "), cut.rfind("."), cut.rfind("。"), cut.rfind("!"), cut.rfind("?"))
    if split_at < limit // 2:
        split_at = limit
    return cut[:split_at].rstrip(" ,;:") + "..."


def publication_intro(body: str, title: str = "") -> str:
    """Build a short, channel-safe preview from a published article body."""
    text = _plain_publication_text(body, title)
    if not text:
        return "새 글이 공개됐다. 아래 링크에서 전문을 확인할 수 있다."

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?。！？…])\s+", text)
        if s.strip()
    ]
    if len(sentences) >= 2:
        intro_parts: list[str] = []
        for sentence in sentences:
            if len(intro_parts) >= PUBLICATION_INTRO_SENTENCES:
                break
            candidate = " ".join([*intro_parts, sentence])
            if intro_parts and len(candidate) > PUBLICATION_INTRO_LIMIT:
                break
            intro_parts.append(sentence)
        return _truncate_intro(" ".join(intro_parts))

    return _truncate_intro(text)


def format_publication_broadcast(*, title: str, url: str, body: str = "") -> str:
    safe_title = html.escape((title or "새 글").strip())
    safe_intro = html.escape(publication_intro(body, title))
    safe_url = html.escape((url or "").strip())
    return f"<b>📢 {safe_title}</b>\n\n{safe_intro}\n\n🔗 전체 글 읽기: {safe_url}"


def format_manual_broadcast(*, title: str, summary: str, url: str) -> str:
    safe_title = html.escape((title or "새 글").strip())
    safe_summary = html.escape(re.sub(r"\s+", " ", (summary or "").strip()))
    safe_url = html.escape((url or "").strip())
    return f"<b>📢 {safe_title}</b>\n\n{safe_summary}\n\n🔗 전체 글 읽기: {safe_url}"


async def send_broadcast(title: str, summary: str, url: str) -> BroadcastResult:
    text = format_manual_broadcast(title=title, summary=summary, url=url)
    return await broadcast_with_token(text, parse_mode="HTML")


def send_broadcast_sync(title: str, summary: str, url: str) -> BroadcastResult:
    text = format_manual_broadcast(title=title, summary=summary, url=url)
    return broadcast_with_token_sync(text, parse_mode="HTML")


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
        return BroadcastResult(False, "브로드캐스트 대상 채널이 설정되지 않았습니다. `/channel set <@채널핸들 또는 -100...chat_id>`를 먼저 실행하십시오.")

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

    text = format_publication_broadcast(title=title, url=url, body=body)
    return await broadcast_with_token(text, parse_mode="HTML")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a formatted Telegram channel broadcast.")
    parser.add_argument("--title", required=True, help="Broadcast title")
    parser.add_argument("--summary", required=True, help="Two- or three-sentence preview summary")
    parser.add_argument("--url", required=True, help="Plain public URL for the full post")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    result = send_broadcast_sync(args.title, args.summary, args.url)
    print(json.dumps({
        "ok": result.ok,
        "message": result.message,
        "sent_count": result.sent_count,
    }, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
