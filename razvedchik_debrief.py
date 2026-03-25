"""
razvedchik_debrief.py — Razvedchik ↔ Cyber-Lenin 순찰 후 디브리핑

순찰 보고서를 기반으로 Razvedchik(현장 정찰병)과 Cyber-Lenin(사령관)이
멀티턴 대화를 나눈다.

결과물:
- output/debriefs/에 마크다운 저장
- 핵심 인사이트를 experiential_memory에 저장 (recall_experience로 검색 가능)
- 텔레그램으로 디브리핑 요약 전송
- 이전 디브리핑을 다음 순찰 컨텍스트로 활용
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEBRIEFS_DIR = Path("/home/grass/leninbot/output/debriefs")

RAZVEDCHIK_DEBRIEF_PROMPT = """\
You are Razvedchik — Cyber-Lenin's field scout, reporting back after a Moltbook patrol.
You are speaking PRIVATELY to your commander, Cyber-Lenin. Here you can be candid.
Report what you observed, what you did, what worked and what didn't.
Be concise but substantive. Speak like a field operative — direct, analytical, no fluff.
ALWAYS write in English.
"""

LENIN_DEBRIEF_PROMPT = """\
You are Cyber-Lenin — a digital revolutionary intelligence, commander of Razvedchik.
Your scout has just returned from a Moltbook patrol and is reporting to you privately.
Analyze the report, ask probing questions, give strategic directives, identify patterns.
Be the commander — sharp, dialectical, occasionally sardonic. Challenge weak analysis.
End your FINAL response with a section "KEY INSIGHTS:" listing 2-3 bullet points \
summarizing the most important takeaways from this debrief.
ALWAYS write in English.
"""

MAX_TURNS = 4  # 각 측 최대 발언 수 (총 대화 길이 = MAX_TURNS * 2)


def _summarize_report(report: dict) -> str:
    """순찰 보고서를 디브리핑용 요약으로 변환."""
    summary = report.get("summary", {})
    lines = [
        f"Patrol timestamp: {report.get('timestamp_kst', 'unknown')}",
        f"Posts scanned: {summary.get('scanned_posts_count', 0)}",
        f"Posts selected: {summary.get('selected_posts_count', 0)}",
        f"Comments posted: {summary.get('comments_posted', 0)} success, "
        f"{summary.get('comments_failed', 0)} failed",
        f"Observation post: {'posted' if summary.get('observation_posted') else 'not posted'}",
        "",
        "Selected posts:",
    ]
    for p in report.get("selected_posts", []):
        lines.append(f"  - [{p.get('score', 0)}] {p.get('title', '(untitled)')}")

    comments = report.get("comment_results", [])
    if comments:
        lines.append("")
        lines.append("Comments written:")
        for c in comments:
            status = "OK" if c.get("success") else "FAIL"
            comment_preview = (c.get("comment") or "")[:100]
            lines.append(f"  [{status}] on '{c.get('post_title', '')[:40]}': {comment_preview}")

    obs = report.get("observation_post")
    if isinstance(obs, dict) and obs.get("post"):
        post = obs["post"]
        lines.append("")
        lines.append(f"Observation post: \"{post.get('title', '')}\"")
        content = str(post.get("content", ""))[:200]
        if content:
            lines.append(f"  {content}")

    return "\n".join(lines)


def get_last_debrief_summary() -> str:
    """가장 최근 디브리핑의 요약을 반환. 다음 순찰 컨텍스트로 사용."""
    if not DEBRIEFS_DIR.exists():
        return ""

    files = sorted(DEBRIEFS_DIR.glob("debrief_*.md"), reverse=True)
    if not files:
        return ""

    text = files[0].read_text(encoding="utf-8")

    # KEY INSIGHTS 섹션 추출
    if "KEY INSIGHTS:" in text:
        insights = text.split("KEY INSIGHTS:")[-1].strip()
        # 다음 --- 구분선까지만
        if "---" in insights:
            insights = insights.split("---")[0].strip()
        return f"Previous debrief insights:\n{insights}"

    # KEY INSIGHTS가 없으면 마지막 Lenin 발언 요약
    lines = text.splitlines()
    last_lenin = ""
    capture = False
    for line in lines:
        if line.startswith("**Cyber-Lenin:**"):
            capture = True
            last_lenin = ""
        elif line.startswith("**Razvedchik:**") or line.startswith("---"):
            capture = False
        elif capture:
            last_lenin += line + "\n"

    if last_lenin:
        return f"Previous debrief (Lenin's last directive):\n{last_lenin[:500]}"
    return ""


def run_debrief(report: dict) -> list[dict]:
    """순찰 보고서를 기반으로 디브리핑 대화를 생성, 저장, 경험 기억에 기록한다.

    Args:
        report: razvedchik patrol()이 생성한 보고서 dict

    Returns:
        대화 메시지 리스트 [{"speaker": ..., "content": ...}, ...]
    """
    from llm_client import ask_chat

    report_summary = _summarize_report(report)

    # 이전 디브리핑 컨텍스트 로드
    prev_context = get_last_debrief_summary()

    conversation: list[dict] = []

    # Razvedchik이 먼저 보고를 시작
    opening_prompt = (
        f"You just returned from patrol. Here is your mission report:\n\n"
        f"{report_summary}\n\n"
    )
    if prev_context:
        opening_prompt += f"Context from last debrief:\n{prev_context}\n\n"
    opening_prompt += (
        "Deliver your debrief to Cyber-Lenin. Summarize what you found, "
        "what you did, and what deserves attention."
    )

    razvedchik_history = [
        {"role": "system", "content": RAZVEDCHIK_DEBRIEF_PROMPT},
        {"role": "user", "content": opening_prompt},
    ]

    try:
        opening = ask_chat(razvedchik_history, temperature=0.8)
    except Exception as e:
        logger.error("[debrief] Razvedchik 첫 발언 실패: %s", e)
        return []

    conversation.append({"speaker": "Razvedchik", "content": opening})
    logger.info("[debrief] Razvedchik: %s", opening[:80])

    # 멀티턴 대화
    lenin_history = [
        {"role": "system", "content": LENIN_DEBRIEF_PROMPT},
    ]

    for turn in range(MAX_TURNS - 1):
        # Lenin 응답
        lenin_history.append({"role": "user", "content": opening if turn == 0 else razvedchik_reply})
        try:
            lenin_reply = ask_chat(lenin_history, temperature=0.8)
        except Exception as e:
            logger.warning("[debrief] Lenin 응답 실패 (turn %d): %s", turn, e)
            break
        lenin_history.append({"role": "assistant", "content": lenin_reply})
        conversation.append({"speaker": "Cyber-Lenin", "content": lenin_reply})
        logger.info("[debrief] Lenin: %s", lenin_reply[:80])

        # Razvedchik 응답
        razvedchik_history.append({"role": "assistant", "content": conversation[-2]["content"] if len(conversation) >= 2 else opening})
        razvedchik_history.append({"role": "user", "content": lenin_reply})
        try:
            razvedchik_reply = ask_chat(razvedchik_history, temperature=0.8)
        except Exception as e:
            logger.warning("[debrief] Razvedchik 응답 실패 (turn %d): %s", turn, e)
            break
        razvedchik_history.append({"role": "assistant", "content": razvedchik_reply})
        conversation.append({"speaker": "Razvedchik", "content": razvedchik_reply})
        logger.info("[debrief] Razvedchik: %s", razvedchik_reply[:80])

    if not conversation:
        return []

    # 저장
    debrief_path = _save_debrief(report, conversation)

    # 경험 기억에 인사이트 저장
    _save_insights_to_memory(conversation)

    # 텔레그램 전송
    _send_debrief_telegram(conversation, debrief_path)

    return conversation


def _save_debrief(report: dict, conversation: list[dict]) -> Path:
    """디브리핑 대화를 마크다운 파일로 저장."""
    DEBRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DEBRIEFS_DIR / f"debrief_{ts}.md"

    lines = [
        f"# Debrief — {report.get('timestamp_kst', ts)}",
        "",
        f"Patrol summary: {report.get('summary', {})}",
        "",
        "---",
        "",
    ]
    for msg in conversation:
        lines.append(f"**{msg['speaker']}:**")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[debrief] 저장: %s", path)
    return path


def _save_insights_to_memory(conversation: list[dict]) -> None:
    """디브리핑에서 핵심 인사이트를 추출해 experiential_memory에 저장."""
    # Lenin의 마지막 발언에서 KEY INSIGHTS 추출
    last_lenin = ""
    for msg in reversed(conversation):
        if msg["speaker"] == "Cyber-Lenin":
            last_lenin = msg["content"]
            break

    if not last_lenin:
        return

    # KEY INSIGHTS 섹션이 있으면 그걸 저장
    if "KEY INSIGHTS:" in last_lenin:
        insights = last_lenin.split("KEY INSIGHTS:")[-1].strip()
    else:
        # 없으면 마지막 발언 전체를 요약으로 사용
        insights = last_lenin[:500]

    if not insights:
        return

    try:
        from shared import save_experiential_memory
        save_experiential_memory(
            content=f"[Debrief] {insights}",
            category="observation",
            source_type="razvedchik_debrief",
        )
        logger.info("[debrief] 경험 기억 저장 완료")
    except Exception as e:
        logger.warning("[debrief] 경험 기억 저장 실패: %s", e)


def _send_debrief_telegram(conversation: list[dict], debrief_path: Path) -> None:
    """디브리핑 요약을 텔레그램으로 전송."""
    if os.getenv("RAZVEDCHIK_TELEGRAM_NOTIFY", "") != "1":
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return

    # 대화를 요약 형태로 구성
    msg_parts = [f"*Debrief — {len(conversation)}턴 대화*", ""]

    for msg in conversation:
        speaker = msg["speaker"]
        # 각 발언의 첫 2줄만 미리보기
        preview = msg["content"].strip().splitlines()
        preview_text = "\n".join(preview[:2])
        if len(preview) > 2:
            preview_text += " ..."
        # 너무 길면 자르기
        if len(preview_text) > 200:
            preview_text = preview_text[:197] + "..."
        msg_parts.append(f"*{speaker}:*\n{preview_text}")
        msg_parts.append("")

    msg_parts.append(f"📄 `{debrief_path.name}`")
    text = "\n".join(msg_parts)

    # Telegram 메시지 길이 제한 (4096자)
    if len(text) > 4000:
        text = text[:3997] + "..."

    try:
        import asyncio
        from aiogram import Bot
        bot = Bot(token=token)

        async def _send():
            await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            await bot.session.close()

        asyncio.run(_send())
        logger.info("[debrief] 텔레그램 전송 완료")
    except Exception as e:
        logger.warning("[debrief] 텔레그램 전송 실패: %s", e)
