"""Public diary publication and Stasova audit helpers."""

from __future__ import annotations

import asyncio
import logging
import re

from db import execute as _execute, query_one as _query_one


logger = logging.getLogger(__name__)

_NO_WARNING_PATTERNS = (
    r"경고\s*항목\s*[:：]\s*(없음|없다|해당\s*없음|무)",
    r"위험\s*없음",
    r"특이\s*사항\s*없음",
    r"no\s+(warnings?|flags?|issues?)",
)


def ensure_diary_publication_audit_table() -> None:
    _execute("""
        CREATE TABLE IF NOT EXISTS diary_publication_audits (
            id SERIAL PRIMARY KEY,
            task_id INTEGER REFERENCES telegram_tasks(id),
            diary_id INTEGER REFERENCES ai_diary(id),
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            final_title TEXT NOT NULL,
            final_content TEXT NOT NULL,
            review_report TEXT NOT NULL,
            warning_detected BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_diary_publication_audits_created
        ON diary_publication_audits(created_at DESC)
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_diary_publication_audits_warning_created
        ON diary_publication_audits(warning_detected, created_at DESC)
    """)


def stasova_report_has_warning(review_report: str) -> bool:
    report = str(review_report or "").strip()
    if not report:
        return True

    risk_match = re.search(r"위험도\s*총평\s*[:：]\s*(낮음|주의|높음|심각)", report)
    if risk_match and risk_match.group(1) in {"주의", "높음", "심각"}:
        return True

    if re.search(r"권고\s*[:：]\s*(조건부\s*공개|보류|비공개)", report):
        return True

    if any(re.search(pattern, report, flags=re.IGNORECASE) for pattern in _NO_WARNING_PATTERNS):
        return False

    if risk_match and risk_match.group(1) == "낮음":
        return False

    # Ambiguous reports are kept for central committee after-review.
    return True


async def record_diary_publication_audit(
    *,
    task_id: int | None,
    diary_id: int | None,
    title: str,
    content: str,
    final_title: str,
    final_content: str,
    review_report: str,
    warning_detected: bool,
) -> int:
    row = await asyncio.to_thread(
        _query_one,
        """
        INSERT INTO diary_publication_audits
            (task_id, diary_id, title, content, final_title, final_content, review_report, warning_detected)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            task_id,
            diary_id,
            title,
            content,
            final_title,
            final_content,
            review_report,
            warning_detected,
        ),
    )
    return int(row["id"])


async def publish_diary_entry(title: str, content: str) -> tuple[int | None, str]:
    row = await asyncio.to_thread(
        _query_one,
        "INSERT INTO ai_diary (title, content) VALUES (%s, %s) RETURNING id",
        (title, content),
    )
    diary_id = row.get("id") if row else None
    broadcast_note = ""
    try:
        from telegram.channel_broadcast import should_broadcast_diary, send_broadcast

        if should_broadcast_diary():
            preview = re.sub(r"\s+", " ", (content or "").strip())
            if len(preview) > 500:
                cut = preview[:501]
                split_at = max(cut.rfind(" "), cut.rfind("."), cut.rfind("。"), cut.rfind("!"), cut.rfind("?"))
                if split_at < 250:
                    split_at = 500
                preview = cut[:split_at].rstrip(" ,;:") + "..."
            public_url = f"https://cyber-lenin.com/ai-diary/{diary_id}" if diary_id else "https://cyber-lenin.com/ai-diary"
            result = await send_broadcast(
                title=f"사이버-레닌 일기: {title}",
                summary=preview,
                url=public_url,
            )
            broadcast_note = f" / Telegram channel: {'sent' if result.ok else result.message}"
    except Exception as exc:
        broadcast_note = f" / Telegram channel failed: {exc}"
    return diary_id, broadcast_note


async def publish_reviewed_diary_entry(
    *,
    task_id: int | None,
    title: str,
    content: str,
    final_title: str,
    final_content: str,
    review_report: str,
) -> tuple[int | None, str, int | None]:
    diary_id, broadcast_note = await publish_diary_entry(final_title, final_content)
    warning_detected = stasova_report_has_warning(review_report)
    audit_id = None
    if warning_detected:
        try:
            ensure_diary_publication_audit_table()
            audit_id = await record_diary_publication_audit(
                task_id=task_id,
                diary_id=diary_id,
                title=title,
                content=content,
                final_title=final_title,
                final_content=final_content,
                review_report=review_report,
                warning_detected=warning_detected,
            )
        except Exception as exc:
            logger.error("failed to record Stasova diary audit: %s", exc)
            broadcast_note += f" / Stasova audit failed: {exc}"
    return diary_id, broadcast_note, audit_id
