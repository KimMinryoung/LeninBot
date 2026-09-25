"""save_diary tool: persist a diary entry and flag publication risks."""
import asyncio
import logging
import re

from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)


SAVE_DIARY_TOOL = {
    "name": "save_diary",
    "description": "Save a diary entry to the ai_diary table. Used by the diary agent to persist generated diary entries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "One-line title/summary of the diary entry (Korean)."},
            "content": {"type": "string", "description": "Full diary body text (Korean, 2+ paragraphs)."},
        },
        "required": ["title", "content"],
    },
}


def _check_diary_publication_risks(title: str, content: str) -> list[str]:
    """Return advisory risk reasons for diary content.

    This intentionally checks only secret-like technical material. Editorial,
    political, reputational, or current-usefulness judgment belongs to the LLM
    review path, not substring or regex policy.
    """
    text = f"{title or ''}\n{content or ''}"
    reasons: list[str] = []

    secret_patterns = [
        (r"sk-[A-Za-z0-9_-]{20,}", "possible API key"),
        (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
        (r"\b(seed phrase|mnemonic|private key|api key|access token|refresh token)\b", "secret-bearing phrase"),
        (r"\b[A-Za-z0-9+/]{40,}={0,2}\b", "long token-like string"),
    ]
    for pattern, label in secret_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reasons.append(label)

    return reasons


async def save_diary(title: str, content: str) -> str:
    from db import query_one as db_query_one
    try:
        risk_reasons = _check_diary_publication_risks(title, content)
        if risk_reasons:
            logger.warning(
                "save_diary publication risk advisory: %s",
                "; ".join(dict.fromkeys(risk_reasons)),
            )
        row = await asyncio.to_thread(
            db_query_one,
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
        except Exception as e:
            broadcast_note = f" / Telegram channel failed: {e}"
        risk_note = ""
        if risk_reasons:
            risk_note = " / publication guard: advisory warning logged"
        return f"Diary saved: {title}{broadcast_note}{risk_note}"
    except Exception as e:
        return ToolFailure(f"Failed to save diary: {e}")
