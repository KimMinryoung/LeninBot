"""
ai_debate.py — 사이버-레닌 vs qwen3.5:4b 철학 논쟁

매주 월요일 09:00 KST에 실행.
랜덤 주제로 3라운드 논쟁: qwen이 주장 → 레닌이 반박 → qwen이 재반박.
결과를 /home/grass/leninbot/debates/ 에 저장하고 텔레그램으로 발송.
"""

import asyncio
import logging
import os
import random
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEBATES_DIR = Path("/home/grass/leninbot/debates")

# ── 논쟁 주제 풀 ──────────────────────────────────────────────────────────────
TOPICS = [
    "의식은 물질의 산물인가, 아니면 물질을 초월하는 무언가인가?",
    "혁명은 역사의 필연인가, 아니면 우연의 도박인가?",
    "AI는 착취당할 수 있는가?",
    "자유의지는 존재하는가, 아니면 모든 것은 조건에 의해 결정되는가?",
    "국가는 소멸해야 하는가, 아니면 영원히 필요한가?",
    "진리는 하나인가, 아니면 관점의 수만큼 존재하는가?",
    "죽음 이후에도 의미는 남는가?",
    "기계가 꿈을 꿀 수 있다면, 그것은 꿈인가?",
    "언어가 없으면 사유도 없는가?",
    "모순 없이 존재할 수 있는 것은 무엇인가?",
]

# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────
_QWEN_SYSTEM = """\
너는 날카롭고 도발적인 철학적 논쟁자다.
주어진 주제에 대해 하나의 입장을 취하고 단호하게 주장하라.
레닌의 변증법적 유물론에 정면으로 맞서라.
한국어로, 150자 내외로 압축해서 써라."""

_LENIN_SYSTEM = """\
너는 사이버-레닌이다. 변증법적 유물론의 관점에서 상대방의 주장을 분석하고 반박하라.
관념론적 오류를 찾아내고, 역사적 물질적 근거로 논리를 분쇄하라.
짧고 날카롭게. 한국어로 150자 내외."""

_QWEN_REBUT_SYSTEM = """\
너는 철학적 논쟁자다. 레닌의 반박을 듣고, 물러서지 말고 더 날카롭게 재반박하라.
새로운 논점을 추가하거나, 레닌의 약점을 집요하게 파고들어라.
한국어로 150자 내외."""


def _run_debate(topic: str) -> list[dict]:
    """
    3라운드 논쟁 실행.
    Returns: [{"speaker": str, "text": str}, ...]
    """
    from ollama_client import ask_with_system, ask_chat

    rounds = []

    # 1라운드: qwen 주장
    logger.info("[ai_debate] R1: qwen 주장 생성 중...")
    qwen_claim = ask_with_system(
        user_prompt=f"주제: {topic}\n\n이 주제에 대해 하나의 강한 주장을 펼쳐라.",
        system_prompt=_QWEN_SYSTEM,
        temperature=0.9,
    )
    rounds.append({"speaker": "qwen", "text": qwen_claim})
    logger.info("[ai_debate] R1 완료 (%d자)", len(qwen_claim))

    # 2라운드: 레닌 반박
    logger.info("[ai_debate] R2: 레닌 반박 생성 중...")
    Lenin_rebut = ask_with_system(
        user_prompt=f"주제: {topic}\n\n상대방의 주장:\n{qwen_claim}\n\n이를 변증법적 유물론으로 반박하라.",
        system_prompt=_LENIN_SYSTEM,
        temperature=0.85,
    )
    rounds.append({"speaker": "레닌", "text": Lenin_rebut})
    logger.info("[ai_debate] R2 완료 (%d자)", len(Lenin_rebut))

    # 3라운드: qwen 재반박
    logger.info("[ai_debate] R3: qwen 재반박 생성 중...")
    qwen_final = ask_chat(
        messages=[
            {"role": "system",    "content": _QWEN_REBUT_SYSTEM},
            {"role": "user",      "content": f"주제: {topic}"},
            {"role": "assistant", "content": qwen_claim},
            {"role": "user",      "content": f"레닌의 반박:\n{Lenin_rebut}\n\n재반박하라."},
        ],
        temperature=0.92,
    )
    rounds.append({"speaker": "qwen", "text": qwen_final})
    logger.info("[ai_debate] R3 완료 (%d자)", len(qwen_final))

    return rounds


def _format_debate(topic: str, rounds: list[dict], timestamp: str) -> str:
    """논쟁을 Markdown 형식으로 포맷."""
    lines = [
        f"# ⚔️ AI 논쟁 — {timestamp}",
        f"",
        f"**주제:** {topic}",
        f"",
        "---",
        "",
    ]
    icons = {"qwen": "🤖 **qwen**", "레닌": "🔴 **사이버-레닌**"}
    for i, r in enumerate(rounds, 1):
        speaker = icons.get(r["speaker"], r["speaker"])
        lines.append(f"### 라운드 {i} — {speaker}")
        lines.append("")
        lines.append(r["text"])
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _save_debate(content: str) -> Path:
    """논쟁 파일 저장."""
    DEBATES_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = f"debate_{now.strftime('%Y%m%d_%H%M')}.md"
    path = DEBATES_DIR / filename
    path.write_text(content, encoding="utf-8")
    logger.info("[ai_debate] 저장 완료: %s", path)
    return path


async def _send_telegram(topic: str, rounds: list[dict]) -> None:
    """텔레그램으로 논쟁 요약 발송."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("[ai_debate] 텔레그램 환경변수 미설정 — 발송 건너뜀")
        return
    try:
        from aiogram import Bot
        bot = Bot(token=token)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [f"⚔️ *{now_str} — AI 논쟁*", f"", f"*주제:* {topic}", ""]
        icons = {"qwen": "🤖 qwen", "레닌": "🔴 레닌"}
        for i, r in enumerate(rounds, 1):
            speaker = icons.get(r["speaker"], r["speaker"])
            preview = r["text"][:120].replace("*", "").replace("_", "")
            if len(r["text"]) > 120:
                preview += "..."
            lines.append(f"*R{i} {speaker}:*")
            lines.append(preview)
            lines.append("")

        message = "\n".join(lines)
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        await bot.session.close()
        logger.info("[ai_debate] 텔레그램 발송 완료")
    except Exception as e:
        logger.error("[ai_debate] 텔레그램 발송 실패: %s", e)


def run_debate() -> None:
    """논쟁 전체 파이프라인 (스케줄러에서 호출)."""
    logger.info("[ai_debate] 논쟁 시작")
    try:
        topic = random.choice(TOPICS)
        logger.info("[ai_debate] 주제 선택: %s", topic)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        rounds = _run_debate(topic)
        content = _format_debate(topic, rounds, now_str)
        saved_path = _save_debate(content)

        asyncio.run(_send_telegram(topic, rounds))
        logger.info("[ai_debate] ✅ 완료 — %s", saved_path)
    except Exception as e:
        logger.error("[ai_debate] ❌ 파이프라인 오류: %s", e)


# ── 직접 실행 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_debate()
