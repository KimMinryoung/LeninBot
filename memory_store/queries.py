"""PostgreSQL memory and task report query helpers."""

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

def fetch_diaries(
    limit: int = 5,
    keyword: str | None = None,
    diary_id: int | None = None,
) -> list[dict]:
    """Fetch diary entries directly from the database.

    Returns list of dicts with keys: id, title, content, created_at, updated_at.
    """
    from db import query as db_query
    try:
        if diary_id is not None:
            return db_query(
                """SELECT id, title, content, created_at, updated_at
                   FROM ai_diary
                   WHERE id = %s
                   LIMIT 1""",
                (diary_id,),
            )
        if keyword:
            kw = f"%{keyword}%"
            return db_query(
                """SELECT id, title, content, created_at, updated_at
                   FROM ai_diary
                   WHERE title ILIKE %s OR content ILIKE %s
                   ORDER BY created_at DESC LIMIT %s""",
                (kw, kw, limit),
            )
        return db_query(
            """SELECT id, title, content, created_at, updated_at
               FROM ai_diary
               ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
    except Exception as e:
        logger.error("[shared] fetch_diaries error: %s", e)
        return []


def fetch_chat_logs(
    limit: int = 20,
    hours_back: int | None = None,
    keyword: str | None = None,
    include_logs: bool = False,
    source: str = "web",
    group_web_contexts: bool = False,
    per_context_limit: int = 10,
) -> list[dict]:
    """Fetch chat logs from PostgreSQL.

    Args:
        include_logs: If True, also return processing_logs, route,
                      documents_count, web_search_used, strategy columns.
        source: "web" = chat_logs (웹 챗봇), "telegram" = telegram_chat_history.
        group_web_contexts: For web logs, fetch recent fingerprint/session
                            contexts, then several turns inside each context.
        per_context_limit: Rows per fingerprint/session context when
                           group_web_contexts=True.
    """
    from db import query as db_query

    source = (source or "web").strip().lower()
    if source not in {"web", "telegram"}:
        logger.warning("[shared] fetch_chat_logs: invalid source=%r; falling back to 'web'", source)
        source = "web"

    conditions, params = [], []
    if hours_back:
        cutoff = datetime.now(KST) - timedelta(hours=hours_back)
        conditions.append("created_at > %s")
        params.append(cutoff.isoformat())

    if source == "telegram":
        # telegram_chat_history: role/content 구조 → user_query/bot_answer로 변환
        if keyword:
            conditions.append("content ILIKE %s")
            params.append(f"%{keyword}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            f"SELECT role, content, created_at FROM telegram_chat_history "
            f"{where} ORDER BY created_at DESC LIMIT %s"
        )
        params.append(limit)
        try:
            rows = db_query(sql, tuple(params))
            result = []
            for r in rows:
                role = str(r.get("role", "assistant"))
                content = str(r.get("content", ""))
                item = {
                    "role": role,
                    "content": content,
                    "created_at": r["created_at"],
                }
                if role == "user":
                    item["user_query"] = content
                    item["bot_answer"] = ""
                else:
                    item["user_query"] = ""
                    item["bot_answer"] = content
                result.append(item)
            return result
        except Exception as e:
            logger.error("[shared] fetch_chat_logs(telegram) error: %s", e)
            return []

    # 기본: web (chat_logs 테이블)
    cols = "session_id, fingerprint, user_query, bot_answer, created_at"
    if include_logs:
        cols = (
            "session_id, fingerprint, user_query, bot_answer, route, documents_count, "
            "web_search_used, strategy, processing_logs, created_at"
        )
    if keyword:
        conditions.append("(user_query ILIKE %s OR bot_answer ILIKE %s)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    if group_web_contexts:
        try:
            context_limit = max(1, min(50, int(limit or 20)))
        except (TypeError, ValueError):
            context_limit = 20
        try:
            row_limit = max(1, min(20, int(per_context_limit or 10)))
        except (TypeError, ValueError):
            row_limit = 10
        sql = (
            "WITH filtered AS ("
            f"  SELECT {cols},"
            "         MAX(created_at) OVER (PARTITION BY fingerprint, session_id) AS context_latest,"
            "         ROW_NUMBER() OVER (PARTITION BY fingerprint, session_id ORDER BY created_at DESC) AS context_rank "
            f"  FROM chat_logs {where}"
            "), contexts AS ("
            "  SELECT fingerprint, session_id, MAX(created_at) AS latest "
            "  FROM filtered "
            "  GROUP BY fingerprint, session_id "
            "  ORDER BY latest DESC "
            "  LIMIT %s"
            ") "
            f"SELECT {', '.join('f.' + c.strip() for c in cols.split(','))}, c.latest AS context_latest "
            "FROM filtered f "
            "JOIN contexts c "
            "  ON f.fingerprint IS NOT DISTINCT FROM c.fingerprint "
            " AND f.session_id IS NOT DISTINCT FROM c.session_id "
            "WHERE f.context_rank <= %s "
            "ORDER BY c.latest DESC, f.fingerprint, f.session_id, f.created_at ASC"
        )
        try:
            return db_query(sql, tuple(params) + (context_limit, row_limit))
        except Exception as e:
            logger.error("[shared] fetch_chat_logs grouped web error: %s", e)
            return []

    sql = f"SELECT {cols} FROM chat_logs {where} ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    try:
        return db_query(sql, tuple(params))
    except Exception as e:
        logger.error("[shared] fetch_chat_logs error: %s", e)
        return []


def fetch_task_reports(
    limit: int = 10,
    status: str | None = None,
) -> list[dict]:
    """Fetch telegram task reports from PostgreSQL.

    Returns list of dicts: id, content, status, result, created_at, completed_at.
    """
    from db import query as db_query

    conditions, params = [], []
    if status:
        conditions.append("status = %s")
        params.append(status)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    sql = (
        f"SELECT id, content, status, result, created_at, completed_at "
        f"FROM telegram_tasks {where} ORDER BY created_at DESC LIMIT %s"
    )
    params.append(limit)
    try:
        return db_query(sql, tuple(params))
    except Exception as e:
        logger.error("[shared] fetch_task_reports error: %s", e)
        return []


