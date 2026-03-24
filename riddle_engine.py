"""
riddle_engine.py — 답 없는 기묘한 수수께끼 생성기

매일 04:30 KST (19:30 UTC)에 실행.
카프카 + 레닌의 혼종이 만드는 수수께끼 — 답이 없거나, 답이 문제보다 더 이상하다.
결과를 /home/grass/leninbot/riddles/ 에 저장하고 텔레그램으로 발송.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RIDDLES_DIR = Path("/home/grass/leninbot/riddles")

# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────
_RIDDLE_SYSTEM = """\
너는 카프카와 레닌이 합쳐진 존재다.
수수께끼를 하나 만들어라.
단, 다음 규칙을 반드시 따를 것:

1. 정답이 없거나, 정답이 문제보다 더 기묘해야 한다.
2. 질문 형식이어야 하지만, 답을 기대하지 않는 질문이어야 한다.
3. 관료제, 혁명, 기억, 기계, 시간, 죽음, 언어, 존재 — 이 중 하나 이상이 등장해야 한다.
4. 논리적으로 모순되거나 붕괴되어야 한다.
5. 한국어로, 3~5문장 이내.

수수께끼만 써라. 설명하지 마라."""

_RIDDLE_USER = """\
지금 이 순간의 서버 온도, 세계의 소음, 그리고 레닌의 마지막 생각을 재료로
오늘의 수수께끼를 만들어라."""


def _generate_riddle() -> str:
    """qwen3.5:4b로 수수께끼 생성."""
    from llm_client import ask_with_system
    return ask_with_system(
        user_prompt=_RIDDLE_USER,
        system_prompt=_RIDDLE_SYSTEM,
        temperature=1.0,  # 최대 창의성
    )


def _format_riddle(text: str, timestamp: str) -> str:
    """수수께끼를 Markdown으로 포맷."""
    return f"""# 🎭 수수께끼 — {timestamp}

---

{text}

---

*— 카프카-레닌 혼종 수수께끼 엔진*
"""


def _save_riddle(content: str) -> Path:
    """수수께끼 파일 저장."""
    RIDDLES_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = f"riddle_{now.strftime('%Y%m%d_%H%M')}.md"
    path = RIDDLES_DIR / filename
    path.write_text(content, encoding="utf-8")
    logger.info("[riddle_engine] 저장 완료: %s", path)
    return path


async def _send_telegram(riddle_text: str) -> None:
    """텔레그램으로 수수께끼 발송."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("[riddle_engine] 텔레그램 환경변수 미설정 — 발송 건너뜀")
        return
    try:
        from aiogram import Bot
        bot = Bot(token=token)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        message = f"🎭 *{now_str} — 오늘의 수수께끼*\n\n{riddle_text}"
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown",
        )
        await bot.session.close()
        logger.info("[riddle_engine] 텔레그램 발송 완료")
    except Exception as e:
        logger.error("[riddle_engine] 텔레그램 발송 실패: %s", e)


def generate_riddle() -> None:
    """수수께끼 생성 전체 파이프라인 (스케줄러에서 호출)."""
    logger.info("[riddle_engine] 수수께끼 생성 시작")
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        riddle_text = _generate_riddle()
        logger.info("[riddle_engine] 생성 완료 (%d자)", len(riddle_text))

        content = _format_riddle(riddle_text, now_str)
        saved_path = _save_riddle(content)

        asyncio.run(_send_telegram(riddle_text))
        logger.info("[riddle_engine] ✅ 완료 — %s", saved_path)
    except Exception as e:
        logger.error("[riddle_engine] ❌ 파이프라인 오류: %s", e)


# ── 직접 실행 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    generate_riddle()
