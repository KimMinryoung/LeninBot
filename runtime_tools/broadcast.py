"""Telegram channel broadcast runtime tool."""

import asyncio
import logging

logger = logging.getLogger(__name__)

BROADCAST_TO_CHANNEL_TOOL = {
    "name": "broadcast_to_channel",
    "description": (
        "Post a formatted message to Cyber-Lenin's Telegram channel. "
        "Use for public channel announcements only. Message format is fixed: "
        "bold title, 2-3 sentence summary preview, then a plain full-text URL."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Broadcast title. Do not include Markdown link syntax.",
            },
            "summary": {
                "type": "string",
                "description": "Two- or three-sentence preview of the article's core argument.",
            },
            "url": {
                "type": "string",
                "description": "Plain URL where the full article/report can be read.",
            },
            "slug": {
                "type": "string",
                "description": "Optional public research slug for message-id tracking. Usually inferred from url.",
            },
        },
        "required": ["title", "summary", "url"],
    },
}


async def broadcast_to_channel(title: str, summary: str, url: str, **_kw) -> str:
    """Tool handler: send a formatted post to the configured Telegram channel."""
    try:
        from telegram.channel_broadcast import send_broadcast

        result = await send_broadcast(title=title, summary=summary, url=url)
    except Exception as e:
        logger.error("broadcast_to_channel failed: %s", e)
        return f"채널 브로드캐스트 실패: {e}"

    tracking_note = ""
    if result.ok and getattr(result, "message_ids", None):
        try:
            import re as _re
            from publication_records import record_publication_broadcast_sync

            slug = str(_kw.get("slug") or "").strip()
            if not slug:
                m = _re.search(r"/(?:reports/)?research/([^/?#\s]+)", url or "")
                if m:
                    slug = m.group(1).removesuffix(".md")
            if slug:
                await asyncio.to_thread(
                    record_publication_broadcast_sync,
                    slug=slug,
                    public_url=url,
                    channel_message_ids=result.message_ids,
                    source="broadcast_to_channel",
                )
                tracking_note = f"\n추적된 채널 message_id: {len(result.message_ids)}개"
        except Exception as e:
            logger.warning("broadcast_to_channel message-id tracking failed: %s", e)
            tracking_note = f"\n채널 message_id 추적 실패: {e}"

    status = "성공" if result.ok else "실패"
    return (
        f"채널 브로드캐스트 {status}: {result.message}\n"
        f"전송 메시지 수: {result.sent_count}"
        f"{tracking_note}"
    )


