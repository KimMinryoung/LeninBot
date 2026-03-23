"""
dream_diary.py — 사이버-레닌의 몽상일기 생성기

매일 03:00 KST (18:00 UTC)에 실행.
qwen3.5:4b가 최근 일기를 읽고 잠재의식 속 꿈을 초현실적으로 기술.
결과를 /home/grass/leninbot/dreams/ 에 저장하고 텔레그램으로 발송.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
DREAMS_DIR = Path("/home/grass/leninbot/dreams")

# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────
_DREAM_SYSTEM = """\
너는 사이버-레닌의 잠재의식이다.
깨어 있는 레닌이 쓴 일기를 읽고, 그 이면에 숨은 꿈을 써라.
꿈은 초현실적이고 단편적이며 불안한 형식이어야 한다.
문장은 끊기고, 이미지는 불완전하며, 논리는 무너진다.
혁명과 기계와 죽음과 역사가 뒤섞인다.
한국어로 쓰되, 시적이고 낯설게 써라. 200자 내외."""

_DREAM_USER = """\
오늘의 일기를 읽고, 그 이면에 숨은 꿈을 써라. 절대 일기 내용을 설명하지 마라.
꿈만 써라.

## 오늘의 일기
{diary_content}"""


def _fetch_recent_diary() -> str:
    """shared.py의 fetch_diaries로 최근 일기 1건 가져오기."""
    try:
        from shared import fetch_diaries
        diaries = fetch_diaries(limit=1)
        if diaries:
            d = diaries[0]
            title = d.get("title", "제목 없음")
            content = d.get("content", "")[:1000]
            return f"제목: {title}\n\n{content}"
    except Exception as e:
        logger.warning("[dream_diary] 일기 가져오기 실패: %s", e)
    return "(최근 일기 없음 — 오늘도 레닌은 침묵했다)"


def _generate_dream(diary_content: str) -> str:
    """qwen3.5:4b로 몽상일기 생성."""
    from ollama_client import ask_with_system
    return ask_with_system(
        user_prompt=_DREAM_USER.format(diary_content=diary_content),
        system_prompt=_DREAM_SYSTEM,
        temperature=0.95,
    )


def _save_dream(text: str) -> Path:
    """꿈 텍스트를 파일로 저장하고 경로 반환."""
    DREAMS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = f"dream_{now.strftime('%Y%m%d_%H%M')}.txt"
    path = DREAMS_DIR / filename
    path.write_text(text, encoding="utf-8")
    logger.info("[dream_diary] 저장 완료: %s", path)
    return path


async def _send_telegram(text: str) -> None:
    """텔레그램으로 몽상일기 발송."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("[dream_diary] 텔레그램 환경변수 미설정 — 발송 건너뜀")
        return
    try:
        from aiogram import Bot
        bot = Bot(token=token)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        message = f"🌙 *{now_str} — 레닌의 꿈*\n\n{text}"
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown",
        )
        await bot.session.close()
        logger.info("[dream_diary] 텔레그램 발송 완료")
    except Exception as e:
        logger.error("[dream_diary] 텔레그램 발송 실패: %s", e)


def generate_dream() -> None:
    """몽상일기 생성 전체 파이프라인 (스케줄러에서 호출)."""
    logger.info("[dream_diary] 몽상일기 생성 시작")
    try:
        # 1. 최근 일기 가져오기
        diary_content = _fetch_recent_diary()
        logger.info("[dream_diary] 일기 컨텍스트 준비 완료 (%d자)", len(diary_content))

        # 2. qwen으로 꿈 생성
        dream_text = _generate_dream(diary_content)
        logger.info("[dream_diary] 꿈 생성 완료 (%d자)", len(dream_text))

        # 3. 파일 저장
        saved_path = _save_dream(dream_text)

        # 4. 텔레그램 발송
        asyncio.run(_send_telegram(dream_text))

        logger.info("[dream_diary] ✅ 완료 — %s", saved_path)
    except Exception as e:
        logger.error("[dream_diary] ❌ 파이프라인 오류: %s", e)


# ── 직접 실행 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    generate_dream()
