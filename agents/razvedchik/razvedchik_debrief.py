"""
razvedchik_debrief.py — Razvedchik ↔ Cyber-Lenin 순찰 후 디브리핑

순찰 보고서를 기반으로 Razvedchik(현장 정찰병)과 Cyber-Lenin(사령관)이
2턴 대화를 나눈다: 정찰 보고 → 사령관 분석+지시.

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

from agents.razvedchik.persona import build_prompt, DEBRIEF_SCOUT, DEBRIEF_COMMANDER

RAZVEDCHIK_DEBRIEF_PROMPT = build_prompt(DEBRIEF_SCOUT)
LENIN_DEBRIEF_PROMPT = DEBRIEF_COMMANDER  # 사령관은 scout 페르소나가 아님


def _summarize_report(report: dict) -> str:
    """순찰 보고서를 디브리핑용 요약으로 변환 — 모든 필드를 구체적으로 포함."""
    summary = report.get("summary", {})
    lines = []

    lines.append(f"Patrol timestamp: {report.get('timestamp_kst', 'unknown')}")
    lines.append("")

    # 핵심 지표
    lines.append("## Metrics")
    lines.append(f"- Posts scanned: {summary.get('scanned_posts_count', 0)}")
    lines.append(f"- Posts selected: {summary.get('selected_posts_count', 0)}")
    lines.append(f"- Comments posted: {summary.get('comments_posted', 0)} success, {summary.get('comments_failed', 0)} failed")
    lines.append(f"- Replies to my posts: {summary.get('replies_posted', 0)}")
    lines.append(f"- Upvoted: {summary.get('upvoted_count', 0)}")
    lines.append(f"- New follows: {summary.get('followed_count', 0)}")

    # /home 데이터
    home = report.get("home_summary")
    if home:
        lines.append("")
        lines.append("## Account Status")
        lines.append(f"- Karma: {home.get('karma', '?')}")
        lines.append(f"- Unread notifications: {home.get('unread_notifications', '?')}")
        lines.append(f"- Posts with new activity: {home.get('activity_posts', 0)}")

    # 선별된 포스트 + 댓글
    selected = report.get("selected_posts", [])
    comment_results = report.get("comment_results", [])
    if selected:
        lines.append("")
        lines.append("## Posts Engaged")
        for p in selected:
            title = p.get("title", "(untitled)")
            score = p.get("score", 0)
            submolt = p.get("submolt", "?")
            lines.append(f"- [{submolt}] \"{title}\" (score={score})")

        # 댓글 결과
        if comment_results:
            lines.append("")
            lines.append("## Comments Written")
            for c in comment_results:
                status = "✓" if c.get("success") else "✗"
                title = c.get("post_title", "")[:40]
                comment = c.get("comment", "")[:120]
                error = c.get("error", "")
                line = f"- [{status}] on \"{title}\": \"{comment}\""
                if error:
                    line += f" | ERROR: {error}"
                lines.append(line)

    # 답글 결과
    reply_results = report.get("reply_results", [])
    if reply_results:
        lines.append("")
        lines.append("## Replies to Activity on My Posts")
        for r in reply_results:
            status = "✓" if r.get("success") else "✗"
            reply_preview = r.get("reply", "")[:100]
            lines.append(f"- [{status}] \"{reply_preview}\"")

    # 팔로우
    followed = report.get("followed_agents", [])
    if followed:
        lines.append("")
        lines.append(f"## New Follows: {', '.join(followed)}")

    # 관찰 포스트
    obs = report.get("observation_post")
    if obs and isinstance(obs, dict):
        lines.append("")
        lines.append("## Observation Post")
        if obs.get("error"):
            lines.append(f"- FAILED: {obs['error'][:150]}")
        else:
            title = obs.get("post", {}).get("title", obs.get("title", ""))
            content = obs.get("post", {}).get("content", obs.get("content", ""))
            lines.append(f"- Title: \"{title}\"")
            lines.append(f"- Content preview: \"{str(content)[:150]}\"")
            vs = obs.get("post", {}).get("verification_status", "")
            if vs:
                lines.append(f"- Verification: {vs}")

    return "\n".join(lines)


def get_last_debrief_summary() -> str:
    """가장 최근 디브리핑의 KEY INSIGHTS를 반환. 다음 순찰 컨텍스트로 사용."""
    if not DEBRIEFS_DIR.exists():
        return ""

    files = sorted(DEBRIEFS_DIR.glob("debrief_*.md"))
    if not files:
        return ""

    text = files[-1].read_text()

    # KEY INSIGHTS 섹션 추출
    if "KEY INSIGHTS:" in text:
        # 마지막 KEY INSIGHTS 블록 사용
        parts = text.split("KEY INSIGHTS:")
        insights = parts[-1].strip()
        # 다음 --- 구분선까지만
        if "---" in insights:
            insights = insights[:insights.index("---")].strip()
        return "Previous debrief insights:\n" + insights[:500]

    return ""


def run_debrief(report: dict) -> list[dict]:
    """순찰 보고서를 기반으로 2턴 디브리핑 대화를 생성, 저장, 경험 기억에 기록한다.

    구조: Razvedchik 보고 → Cyber-Lenin 분석+지시 (2회 LLM 호출)

    Args:
        report: razvedchik patrol()이 생성한 보고서 dict

    Returns:
        대화 메시지 리스트 [{"speaker": ..., "content": ...}, ...]
    """
    from llm.client import ask_chat

    report_summary = _summarize_report(report)
    prev_context = get_last_debrief_summary()

    conversation: list[dict] = []

    # ── Razvedchik 보고 ────────────────────────────────────────────────────────
    opening_prompt = (
        "You just returned from patrol. Here is your mission report:\n\n"
        + report_summary
    )
    if prev_context:
        opening_prompt += "\n\nContext from last debrief:\n" + prev_context

    opening_prompt += (
        "\n\nDeliver your debrief to Cyber-Lenin. "
        "Cover: what you did, what worked, what failed, what patterns you noticed. "
        "Stick to facts from the report above. Under 300 words."
    )

    razvedchik_history = [
        {"role": "system", "content": RAZVEDCHIK_DEBRIEF_PROMPT},
        {"role": "user", "content": opening_prompt},
    ]

    try:
        opening = ask_chat(razvedchik_history)
        conversation.append({"speaker": "Razvedchik", "content": opening})
        logger.info("[debrief] Razvedchik: %s", opening[:120])
    except Exception as e:
        logger.error("[debrief] Razvedchik 보고 실패: %s", e)
        return conversation

    # ── Cyber-Lenin 분석 + 지시 ────────────────────────────────────────────────
    lenin_history = [
        {"role": "system", "content": LENIN_DEBRIEF_PROMPT},
        {"role": "user", "content": (
            f"Scout's report:\n\n{opening}\n\n"
            f"Raw patrol data for reference:\n{report_summary}\n\n"
            'Analyze and give directives. End with "KEY INSIGHTS:" section.'
        )},
    ]

    try:
        lenin_reply = ask_chat(lenin_history)
        conversation.append({"speaker": "Cyber-Lenin", "content": lenin_reply})
        logger.info("[debrief] Lenin: %s", lenin_reply[:120])
    except Exception as e:
        logger.warning("[debrief] Lenin 응답 실패: %s", e)

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
        f"Patrol summary: {json.dumps(report.get('summary', {}), indent=2)}",
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
        insights = last_lenin.split("KEY INSIGHTS:")[-1].strip()

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
