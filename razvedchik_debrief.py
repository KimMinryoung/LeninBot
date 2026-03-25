"""
razvedchik_debrief.py — Razvedchik ↔ Cyber-Lenin 순찰 후 디브리핑

순찰 보고서를 기반으로 Razvedchik(현장 정찰병)과 Cyber-Lenin(사령관)이
멀티턴 대화를 나눈다. 대화는 output/debriefs/에 저장된다.
"""

import json
import logging
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

    # 댓글 결과 요약
    comments = report.get("comment_results", [])
    if comments:
        lines.append("")
        lines.append("Comments written:")
        for c in comments:
            status = "OK" if c.get("success") else "FAIL"
            comment_preview = (c.get("comment") or "")[:100]
            lines.append(f"  [{status}] on '{c.get('post_title', '')[:40]}': {comment_preview}")

    # 관찰 포스트
    obs = report.get("observation_post")
    if isinstance(obs, dict) and obs.get("post"):
        post = obs["post"]
        lines.append("")
        lines.append(f"Observation post: \"{post.get('title', '')}\"")
        content = str(post.get("content", ""))[:200]
        if content:
            lines.append(f"  {content}")

    return "\n".join(lines)


def run_debrief(report: dict) -> list[dict]:
    """순찰 보고서를 기반으로 디브리핑 대화를 생성하고 저장한다.

    Args:
        report: razvedchik patrol()이 생성한 보고서 dict

    Returns:
        대화 메시지 리스트 [{"speaker": ..., "content": ...}, ...]
    """
    from llm_client import ask_chat

    report_summary = _summarize_report(report)
    conversation: list[dict] = []

    # Razvedchik이 먼저 보고를 시작
    razvedchik_history = [
        {"role": "system", "content": RAZVEDCHIK_DEBRIEF_PROMPT},
        {"role": "user", "content": (
            f"You just returned from patrol. Here is your mission report:\n\n"
            f"{report_summary}\n\n"
            f"Deliver your debrief to Cyber-Lenin. Summarize what you found, "
            f"what you did, and what deserves attention."
        )},
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

    # 저장
    if conversation:
        _save_debrief(report, conversation)

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
