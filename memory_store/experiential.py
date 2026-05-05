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


def save_experiential_memory(
    content: str,
    category: str,
    source_type: str = "auto_reflection",
) -> bool:
    """Save an insight/lesson/pattern to experiential_memory with embedding.

    Args:
        content: The insight text.
        category: One of: lesson, mistake, pattern, insight, observation.
        source_type: Origin — auto_reflection, telegram_chat, web_chat, etc.

    Returns:
        True on success, False on failure.
    """
    from db import execute as db_execute

    try:
        emb = _get_exp_embeddings()
        vec = emb.embed_query(content)
        embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
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


