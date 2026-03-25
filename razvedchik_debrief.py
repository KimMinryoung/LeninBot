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
End your FINAL response with a section "KEY INSIGHTS:" listing 2-3 bullet points summarizing the most important takeaways from this debrief.
ALWAYS write in English.
"""

MAX_TURNS = 3

# 컨텍스트 트리밍 — role collapse 방지
MAX_HISTORY_MESSAGES = 6  # system/opening 제외, 최근 6개 메시지 (= 3턴 분량)


def _trim_history(history: list[dict], keep_indices: int = 2) -> list[dict]:
    """히스토리 컨텍스트 트리밍.

    - history[:keep_indices] 는 항상 보존 (system, opening 등)
    - 나머지는 최근 MAX_HISTORY_MESSAGES 개만 유지
    """
    protected = history[:keep_indices]
    sliding = history[keep_indices:]
    if len(sliding) <= MAX_HISTORY_MESSAGES:
        return history
    trimmed = sliding[-MAX_HISTORY_MESSAGES:]
    return protected + trimmed


def _summarize_report(report: dict) -> str:
    """순찰 보고서를 디브리핑용 요약으로 변환."""
    summary = []
    lines = []

    lines.append("Patrol timestamp: " + str(report.get("timestamp_kst", "unknown")))
    lines.append("Posts scanned: " + str(report.get("scanned_posts_count", 0)))
    lines.append("Posts selected: " + str(report.get("selected_posts_count", 0)))
    lines.append(
        "Comments posted: "
        + str(report.get("comments_posted", 0))
        + " success, "
        + str(report.get("comments_failed", 0))
        + " failed"
    )
    lines.append(
        "Observation post: "
        + ("posted" if report.get("observation_posted") else "not posted")
    )

    selected = report.get("selected_posts", [])
    if selected:
        lines.append("Selected posts:")
        for p in selected:
            title = p.get("post_title", "(untitled)") if isinstance(p, dict) else str(p)
            lines.append(f"  - {title}")
            comments = p.get("comment_results", []) if isinstance(p, dict) else []
            if comments:
                lines.append("  Comments written:")
                for c in comments:
                    status = c.get("success", False) if isinstance(c, dict) else False
                    comment_preview = str(c.get("comment", ""))[:80] if isinstance(c, dict) else ""
                    lines.append(f"    [{'+' if status else 'FAIL'}] {comment_preview}")

    obs = report.get("observation_post")
    if obs and isinstance(obs, dict):
        content = obs.get("content", "")
        lines.append(f'Observation post: "{str(content)[:120]}"')

    return "\n".join(lines)


def get_last_debrief_summary() -> str:
    """가장 최근 디브리핑의 요약을 반환. 다음 순찰 컨텍스트로 사용."""
    if not DEBRIEFS_DIR.exists():
        return ""

    files = sorted(DEBRIEFS_DIR.glob("debrief_*.md"))
    if not files:
        return ""

    text = files[-1].read_text()

    # KEY INSIGHTS 섹션 추출
    if "KEY INSIGHTS:" in text:
        insights = text.split("KEY INSIGHTS:")[1].strip()
        return "Previous debrief insights:\n" + insights[:600]

    # 없으면 Lenin의 마지막 발언 추출
    lines = text.splitlines()
    last_lenin = []
    capture = False
    for line in lines:
        if line.startswith("**Cyber-Lenin:**"):
            last_lenin = []
            capture = True
        elif line.startswith("**Razvedchik:**"):
            capture = False
        elif capture:
            last_lenin.append(line)

    if last_lenin:
        return "Previous debrief (Lenin's last directive):\n" + "\n".join(last_lenin).strip()[:600]

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
    prev_context = get_last_debrief_summary()

    conversation: list[dict] = []

    # ── Razvedchik 첫 발언 ────────────────────────────────────────────────────
    opening_prompt = (
        "You just returned from patrol. Here is your mission report:\n\n"
        + report_summary
    )
    if prev_context:
        opening_prompt += "\n\nContext from last debrief:\n" + prev_context

    opening_prompt += "\n\nDeliver your debrief to Cyber-Lenin. Summarize what you found, what you did, and what deserves attention."

    razvedchik_history = [
        {"role": "system", "content": RAZVEDCHIK_DEBRIEF_PROMPT},
        {"role": "user", "content": opening_prompt},
    ]

    try:
        opening = ask_chat(razvedchik_history)
        razvedchik_history.append({"role": "assistant", "content": opening})
        conversation.append({"speaker": "Razvedchik", "content": opening})
        logger.info("[debrief] Razvedchik: %s", opening[:120])
    except Exception as e:
        logger.error("[debrief] Razvedchik 첫 발언 실패: %s", e)
        return conversation

    # ── Lenin 히스토리 초기화 ─────────────────────────────────────────────────
    lenin_history = [
        {"role": "system", "content": LENIN_DEBRIEF_PROMPT},
    ]

    # ── 멀티턴 대화 ───────────────────────────────────────────────────────────
    for turn in range(MAX_TURNS):
        is_final = turn == MAX_TURNS - 1

        # Lenin 응답
        final_hint = (
            ' Remember to end with "KEY INSIGHTS:" section.' if is_final else ""
        )
        lenin_history.append(
            {"role": "user", "content": razvedchik_history[-1]["content"] + final_hint}
        )

        try:
            lenin_reply = ask_chat(lenin_history)
            lenin_history.append({"role": "assistant", "content": lenin_reply})
            # 트리밍 — system(0) 보존, 이후 슬라이딩
            lenin_history = _trim_history(lenin_history, keep_indices=1)
            conversation.append({"speaker": "Cyber-Lenin", "content": lenin_reply})
            logger.info("[debrief] Lenin: %s", lenin_reply[:120])
        except Exception as e:
            logger.warning("[debrief] Lenin 응답 실패 (turn %d): %s", turn, e)
            break

        if is_final:
            break

        # Razvedchik 응답
        razvedchik_history.append({"role": "user", "content": lenin_reply})

        try:
            razvedchik_reply = ask_chat(razvedchik_history)
            razvedchik_history.append({"role": "assistant", "content": razvedchik_reply})
            # 트리밍 — system(0) + opening_prompt(1) 보존, 이후 슬라이딩
            razvedchik_history = _trim_history(razvedchik_history, keep_indices=2)
            conversation.append({"speaker": "Razvedchik", "content": razvedchik_reply})
            logger.info("[debrief] Razvedchik: %s", razvedchik_reply[:120])
        except Exception as e:
            logger.warning("[debrief] Razvedchik 응답 실패 (turn %d): %s", turn, e)
            break

    # ── 저장 ──────────────────────────────────────────────────────────────────
    debrief_path = _save_debrief(report, conversation)
    _save_insights_to_memory(conversation)
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

    path.write_text("\n".join(lines))
    logger.info("[debrief] 저장: %s", path)
    return path


def _save_insights_to_memory(conversation: list[dict]) -> None:
    """디브리핑에서 핵심 인사이트를 추출해 experiential_memory에 저장."""
    last_lenin = None
    for msg in reversed(conversation):
        if msg["speaker"] == "Cyber-Lenin":
            last_lenin = msg["content"]
            break

    if not last_lenin:
        return

    insights = ""
    if "KEY INSIGHTS:" in last_lenin:
        insights = last_lenin.split("KEY INSIGHTS:")[1].strip()

    if not insights:
        return

    try:
        import shared
        shared.save_experiential_memory(
            observation=insights,
            source="razvedchik_debrief",
        )
        logger.info("[debrief] 경험 기억 저장 완료")
    except Exception as e:
        logger.warning("[debrief] 경험 기억 저장 실패: %s", e)


def _send_debrief_telegram(conversation: list[dict], debrief_path: Path) -> None:
    """디브리핑 요약을 텔레그램으로 전송."""
    if not os.getenv("RAZVEDCHIK_TELEGRAM_NOTIFY"):
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not len(token or ""):
        return

    msg_parts = [f"*Debrief — {debrief_path.name}*\n"]
    for msg in conversation:
        speaker = msg["speaker"]
        content = msg["content"]
        preview_text = content.strip().splitlines()
        preview = "\n".join(preview_text[:3])
        msg_parts.append(f"**{speaker}:** {preview[:200]}")

    msg = "\n\n".join(msg_parts)

    try:
        import asyncio
        from aiogram import Bot

        async def _send():
            session = Bot(token=token)
            await session.send_message(
                chat_id=os.getenv("TELEGRAM_CHAT_ID"),
                text=msg,
                parse_mode="Markdown",
            )
            await session.session.close()

        asyncio.run(_send())
        logger.info("[debrief] 텔레그램 전송 완료")
    except Exception as e:
        logger.warning("[debrief] 텔레그램 전송 실패: %s", e)
