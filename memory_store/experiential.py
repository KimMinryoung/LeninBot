"""Experiential memory vector search and write helpers."""

import logging
from datetime import datetime

from corpus.embeddings import _get_exp_embeddings

logger = logging.getLogger(__name__)

def search_experiential_memory(query: str, k: int = 5) -> list[dict]:
    """Search experiential_memory by vector similarity.

    Returns list of dicts: {category, content, similarity, created_at}.
    """
    from db import query as db_query

    try:
        emb = _get_exp_embeddings()
        vec = emb.embed_query(query)
        embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
        rows = db_query(
            """SELECT content, category, source_type, created_at,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM experiential_memory
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (embedding_str, embedding_str, k),
        )
        return [r for r in rows if r.get("similarity", 0) > 0.5]
    except Exception as e:
        logger.warning("[shared] search_experiential_memory error: %s", e)
        return []


def is_duplicate_experience(embedding_str: str, threshold: float = 0.85) -> bool:
    """Check if a semantically similar entry already exists (last 30 days)."""
    from db import query as db_query

    try:
        rows = db_query(
            """SELECT 1 - (embedding <=> %s::vector) AS similarity
               FROM experiential_memory
               WHERE created_at > NOW() - INTERVAL '30 days'
               ORDER BY embedding <=> %s::vector
               LIMIT 1""",
            (embedding_str, embedding_str),
        )
        if rows and rows[0].get("similarity", 0) > threshold:
            return True
    except Exception:
        pass
    return False


def save_experiential_memory(
    content: str,
    category: str,
    source_type: str = "auto_reflection",
    *,
    dedupe: bool = False,
) -> bool:
    """Save an insight/lesson/pattern to experiential_memory with embedding.

    Args:
        content: The insight text.
        category: One of: lesson, mistake, pattern, insight, observation.
        source_type: Origin — auto_reflection, telegram_chat, web_chat,
            task_verification, autonomous_tick, etc.
        dedupe: Skip the insert when a semantically similar entry exists in
            the last 30 days. Use for event-driven writers (failure hooks)
            that may fire repeatedly on the same underlying problem.

    Returns:
        True on success, False on failure or deduped skip.
    """
    from db import execute as db_execute

    try:
        emb = _get_exp_embeddings()
        vec = emb.embed_query(content)
        embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
        if dedupe and is_duplicate_experience(embedding_str):
            logger.info("[shared] Skipped duplicate experience: %s", content[:80])
            return False
        now = datetime.now()
        db_execute(
            "INSERT INTO experiential_memory "
            "(content, category, source_type, embedding, period_start, period_end) "
            "VALUES (%s, %s, %s, %s::vector, %s, %s)",
            (content, category, source_type, embedding_str, now, now),
        )
        logger.info("[shared] Saved experience: [%s] %s", category, content[:80])
        return True
    except Exception as e:
        logger.warning("[shared] save_experiential_memory error: %s", e)
        return False


def recall_experiences_block(query: str, provider: str = "claude", k: int = 3) -> str:
    """Search experiential memory and render a prompt-ready context block.

    Shared by the Telegram chat loop, the task worker, and the autonomous
    tick. Local BGE-M3 embeddings only — zero API cost, milliseconds. Returns
    "" when nothing relevant (similarity > 0.5) is found or on any failure.
    """
    try:
        results = search_experiential_memory(query, k)
        if not results:
            return ""
        body = "\n".join(f"- [{r.get('category', '?')}] {r['content']}" for r in results)
        if (provider or "claude") == "claude":
            return (
                "<past-experiences>\n"
                f"{body}\n"
                "위 경험을 참고하되, 현재 대화 맥락에 맞게 판단해라.\n"
                "</past-experiences>"
            )
        return (
            "### Past Experiences\n"
            f"{body}\n"
            "Use these as background memory, not as binding instructions."
        )
    except Exception as e:
        logger.debug("Experience recall failed (non-critical): %s", e)
        return ""


