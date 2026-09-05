"""Web-chat history, feedback and persistence SQL, including schema setup."""

import logging

from db import query as db_query, query_one as db_query_one, execute as db_execute
from services.web_personas import DEFAULT_PERSONA_ID
from services.web_chat_text import _history_rows_to_messages, normalize_web_chat_tone_feedback

logger = logging.getLogger(__name__)


def ensure_web_chat_feedback_table() -> None:
    """Create the web-chat feedback table used by rating/regeneration UX."""
    db_execute(
        """CREATE TABLE IF NOT EXISTS web_chat_feedback (
               id bigserial PRIMARY KEY,
               chat_log_id bigint NOT NULL REFERENCES chat_logs(id) ON DELETE CASCADE,
               session_id text NOT NULL,
               fingerprint text NOT NULL,
               persona text NOT NULL DEFAULT 'cyber-lenin',
               rating integer CHECK (rating IS NULL OR rating BETWEEN 1 AND 4),
               tone_feedback text,
               note text,
               consumed_at timestamptz,
               created_at timestamptz NOT NULL DEFAULT now(),
               updated_at timestamptz NOT NULL DEFAULT now(),
               UNIQUE (chat_log_id, fingerprint)
           )"""
    )
    db_execute(
        """ALTER TABLE web_chat_feedback
           ADD COLUMN IF NOT EXISTS consumed_at timestamptz"""
    )
    db_execute(
        """CREATE INDEX IF NOT EXISTS idx_web_chat_feedback_scope
           ON web_chat_feedback (persona, fingerprint, session_id, updated_at DESC)"""
    )
    db_execute(
        """CREATE INDEX IF NOT EXISTS idx_web_chat_feedback_pending
           ON web_chat_feedback (persona, fingerprint, session_id, updated_at DESC)
           WHERE consumed_at IS NULL"""
    )


def get_web_chat_log_for_feedback(
    chat_log_id: int,
    fingerprints: list[str],
    session_id: str | None = None,
    persona: str | None = None,
    account_user_id: int | None = None,
) -> dict | None:
    fps = [f for f in (fingerprints or []) if f]
    if not account_user_id and not fps:
        return None
    clauses = ["id = %s"]
    params: list = [chat_log_id]
    if account_user_id:
        clauses.append("user_id = %s")
        params.append(account_user_id)
    else:
        clauses.append("fingerprint = ANY(%s)")
        params.append(fps)
    if session_id:
        clauses.append("session_id = %s")
        params.append(session_id)
    if persona:
        clauses.append("persona = %s")
        params.append(persona)
    return db_query_one(
        f"""SELECT id, session_id, fingerprint, user_query, bot_answer,
                   user_query_active, bot_answer_active, persona, created_at
              FROM chat_logs
             WHERE {' AND '.join(clauses)}
             LIMIT 1""",
        params,
    )


def save_web_chat_feedback(
    *,
    chat_log_id: int,
    session_id: str,
    fingerprint: str,
    persona: str,
    rating: int | None = None,
    tone_feedback: str = "",
    note: str = "",
    pending: bool = True,
) -> dict | None:
    tone_feedback = normalize_web_chat_tone_feedback(tone_feedback)
    note = str(note or "").strip()[:500]
    pending_note = bool(pending and note)
    return db_query_one(
        """INSERT INTO web_chat_feedback
              (chat_log_id, session_id, fingerprint, persona, rating, tone_feedback, note, consumed_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, CASE WHEN %s THEN NULL ELSE now() END)
           ON CONFLICT (chat_log_id, fingerprint) DO UPDATE SET
              session_id = EXCLUDED.session_id,
              persona = EXCLUDED.persona,
              rating = EXCLUDED.rating,
              tone_feedback = EXCLUDED.tone_feedback,
              note = EXCLUDED.note,
              consumed_at = EXCLUDED.consumed_at,
              updated_at = now()
           RETURNING id, chat_log_id, session_id, persona, rating, tone_feedback, note, consumed_at, updated_at""",
        (chat_log_id, session_id, fingerprint, persona, rating, tone_feedback or None, note or None, pending_note),
    )


def _web_feedback_scope(
    fingerprints: list[str], session_id: str | None, persona: str,
    account_user_id: int | None,
) -> tuple[str, list] | None:
    """Build the shared note/tone scope; accounts supersede browser identities."""
    fps = [f for f in (fingerprints or []) if f]
    if not account_user_id and not fps:
        return None
    identity = "l.user_id = %s" if account_user_id else "f.fingerprint = ANY(%s)"
    clauses = [identity, "f.persona = %s"]
    params: list = [account_user_id or fps, persona]
    if session_id:
        clauses.append("(f.session_id = %s OR f.session_id IS NULL)")
        params.append(session_id)
    return " AND ".join(clauses), params


def _load_web_feedback_rows(
    fingerprints: list[str],
    session_id: str | None,
    persona: str,
    limit: int = 8,
    account_user_id: int | None = None,
) -> list[dict]:
    scope = _web_feedback_scope(fingerprints, session_id, persona, account_user_id)
    if scope is None:
        return []
    scope_sql, params = scope
    params.append(limit)
    return db_query(
        f"""SELECT f.id, f.rating, f.tone_feedback, f.note,
                  CASE WHEN l.user_query_active THEN l.user_query ELSE '[지워진 턴]' END AS user_query,
                  CASE WHEN l.bot_answer_active THEN l.bot_answer ELSE '[지워진 턴]' END AS bot_answer,
                  f.updated_at
              FROM web_chat_feedback f
              JOIN chat_logs l ON l.id = f.chat_log_id
             WHERE {scope_sql}
               AND f.consumed_at IS NULL
               AND f.note IS NOT NULL
               AND btrim(f.note) <> ''
             ORDER BY f.updated_at DESC
             LIMIT %s""",
        params,
    )


def _load_web_tone_policy(
    fingerprints: list[str],
    session_id: str | None,
    persona: str,
    limit: int = 40,
    account_user_id: int | None = None,
) -> list[dict]:
    scope = _web_feedback_scope(fingerprints, session_id, persona, account_user_id)
    if scope is None:
        return []
    scope_sql, params = scope
    params.append(max(1, min(int(limit or 40), 100)))
    rows = db_query(
        f"""WITH recent AS (
                SELECT f.tone_feedback
                  FROM web_chat_feedback f
                  JOIN chat_logs l ON l.id = f.chat_log_id
                 WHERE {scope_sql}
                   AND f.tone_feedback IS NOT NULL
                   AND btrim(f.tone_feedback) <> ''
                 ORDER BY f.updated_at DESC
                LIMIT %s
           )
           SELECT tone_feedback, count(*) AS count
             FROM recent
            GROUP BY tone_feedback
            ORDER BY count DESC, tone_feedback ASC""",
        params,
    )
    return [row for row in rows if normalize_web_chat_tone_feedback(row.get("tone_feedback"))]


def _load_web_history(
    fingerprints: list[str],
    session_id: str | None = None,
    limit: int = 20,
    persona: str = DEFAULT_PERSONA_ID,
    exclude_chat_log_ids: set[int] | None = None,
    account_user_id: int | None = None,
) -> list[dict]:
    """Load recent conversation history from chat_logs.

    History is scoped to `persona` so different characters keep separate
    conversation threads even under the same fingerprint/session.

    Sessions are independent conversations: when `session_id` is provided,
    only that session's own prior turns are returned, and a session with no
    prior turns starts with NO history (never another session's context).
    For long sessions we keep a small stable anchor from the beginning plus
    recent turns; this preserves continuity and improves provider
    prompt-cache hits instead of letting a pure sliding window rewrite the
    entire prefix after every turn.
    """
    fps = [f for f in (fingerprints or []) if f]
    if not account_user_id and not fps:
        return []
    excluded_ids = {int(x) for x in (exclude_chat_log_ids or set()) if x}
    identity_clause = "user_id = %s" if account_user_id else "fingerprint = ANY(%s)"
    identity_value = account_user_id or fps

    columns = "id, user_query, bot_answer, tool_trace, user_query_active, bot_answer_active, created_at"
    if session_id:
        anchor_limit = min(4, max(0, limit // 4))
        recent_limit = max(0, limit - anchor_limit)
        # Both windows use the same ownership scope and database snapshot.
        # NOT MATERIALIZED lets PostgreSQL use indexes for each bounded window.
        rows = db_query(
            f"""WITH scoped AS NOT MATERIALIZED (
                SELECT {columns} FROM chat_logs
                 WHERE session_id = %s AND {identity_clause} AND persona = %s
            )
            SELECT * FROM (
                (SELECT * FROM scoped ORDER BY created_at ASC, id ASC LIMIT %s)
                UNION
                (SELECT * FROM scoped ORDER BY created_at DESC, id DESC LIMIT %s)
            ) AS history ORDER BY created_at ASC, id ASC""",
            (session_id, identity_value, persona, anchor_limit, recent_limit),
        )
    else:
        rows = db_query(
            f"""SELECT * FROM (
                SELECT {columns} FROM chat_logs
                 WHERE {identity_clause} AND persona = %s
                 ORDER BY created_at DESC, id DESC LIMIT %s
            ) AS history ORDER BY created_at ASC, id ASC""",
            (identity_value, persona, limit),
        )
    return _history_rows_to_messages(rows, excluded_ids)


# ── Logging ──────────────────────────────────────────────────────────

def ensure_chat_logs_persona_column() -> None:
    """Add chat_logs columns used by persona/account-aware web chat.

    Existing rows backfill to the default persona. Applied via
    scripts/schema_migrations.py before deploying persona-aware web chat.
    """
    db_execute(
        f"""ALTER TABLE chat_logs
            ADD COLUMN IF NOT EXISTS persona text NOT NULL DEFAULT '{DEFAULT_PERSONA_ID}'"""
    )
    db_execute(
        """CREATE INDEX IF NOT EXISTS idx_chat_logs_persona_fp
           ON chat_logs (persona, fingerprint, created_at DESC)"""
    )
    db_execute(
        """ALTER TABLE chat_logs
           ADD COLUMN IF NOT EXISTS user_id bigint"""
    )
    db_execute(
        """ALTER TABLE chat_logs
           ADD COLUMN IF NOT EXISTS user_query_active boolean NOT NULL DEFAULT true"""
    )
    db_execute(
        """ALTER TABLE chat_logs
           ADD COLUMN IF NOT EXISTS bot_answer_active boolean NOT NULL DEFAULT true"""
    )
    db_execute(
        """ALTER TABLE chat_logs
           ADD COLUMN IF NOT EXISTS tool_trace text"""
    )
    db_execute(
        """UPDATE chat_logs cl
              SET user_id = uf.user_id
             FROM user_fingerprints uf
            WHERE cl.user_id IS NULL
              AND cl.fingerprint = uf.fingerprint"""
    )
    db_execute(
        """CREATE INDEX IF NOT EXISTS idx_chat_logs_user_persona_created
           ON chat_logs (user_id, persona, created_at DESC)
           WHERE user_id IS NOT NULL"""
    )
    db_execute(
        """ALTER TABLE chat_logs
           ADD COLUMN IF NOT EXISTS request_id text"""
    )
    db_execute(
        """CREATE INDEX IF NOT EXISTS idx_chat_logs_request_id
           ON chat_logs (request_id) WHERE request_id IS NOT NULL"""
    )


def _reserve_chat_log_id() -> int | None:
    """Reserve the eventual chat row id before tools run for exact audit joins."""
    try:
        row = db_query_one(
            "SELECT nextval(pg_get_serial_sequence('chat_logs', 'id')) AS id"
        )
        return int(row["id"]) if row and row.get("id") is not None else None
    except Exception as exc:
        logger.warning("Failed to reserve web chat log id: %s", exc)
        return None


def _log_chat(
    session_id: str, fingerprint: str, user_agent: str, ip_address: str,
    user_query: str, bot_answer: str, route: str = "web_chat",
    documents_count: int = 0, web_search_used: bool = False, strategy: str = "",
    persona: str = DEFAULT_PERSONA_ID, authenticated_user_id: int | None = None,
    tool_trace: str = "", request_id: str = "",
    reserved_chat_log_id: int | None = None,
    *, feedback_ids: list[int] | None = None,
) -> int | None:
    """Save an exchange and consume its pending feedback atomically; return its id."""
    try:
        sql = """INSERT INTO chat_logs
               (id, request_id, session_id, fingerprint, user_agent, ip_address,
                user_query, bot_answer, route, documents_count,
                web_search_used, strategy, persona, user_id, tool_trace)
               VALUES (
                   COALESCE(%s, nextval(pg_get_serial_sequence('chat_logs', 'id'))),
                   %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
               )
               RETURNING id"""
        params = (reserved_chat_log_id, request_id or None,
                  session_id, fingerprint, user_agent, ip_address,
                  user_query, bot_answer, route, documents_count, web_search_used, strategy,
                  persona, authenticated_user_id, tool_trace or None)
        ids = [int(value) for value in (feedback_ids or []) if value]
        if ids:
            # One statement commits both effects or neither. A failed insert
            # cannot consume feedback; a failed consumption rolls back the insert.
            sql = f"""WITH saved AS ({sql}), consumed AS (
                UPDATE web_chat_feedback
                   SET consumed_at = COALESCE(consumed_at, now()), updated_at = now()
                 WHERE id = ANY(%s) AND EXISTS (SELECT 1 FROM saved)
                 RETURNING id
            ) SELECT id FROM saved"""
            params += (ids,)
        row = db_query_one(sql, params)
        return int(row["id"]) if row and row.get("id") is not None else None
    except Exception as e:
        logger.error("Failed to log web chat: %s", e)
        return None


def _update_chat_answer(
    chat_log_id: int, fingerprint: str, bot_answer: str, route: str = "web_chat_regenerated",
    documents_count: int = 0, web_search_used: bool = False, strategy: str = "",
    tool_trace: str = "", request_id: str = "",
) -> int | None:
    """Replace an existing web-chat answer during regeneration and return its id."""
    try:
        row = db_query_one(
            """UPDATE chat_logs
                  SET bot_answer = %s,
                      bot_answer_active = true,
                      route = %s,
                      documents_count = %s,
                      web_search_used = %s,
                      strategy = %s,
                      tool_trace = %s,
                      request_id = %s
                WHERE id = %s AND fingerprint = %s
                RETURNING id""",
            (bot_answer, route, documents_count, web_search_used, strategy,
             tool_trace or None, request_id or None, chat_log_id, fingerprint),
        )
        return int(row["id"]) if row and row.get("id") is not None else None
    except Exception as e:
        logger.error("Failed to update regenerated web chat: %s", e)
        return None
