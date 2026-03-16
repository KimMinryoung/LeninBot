"""experience_writer.py — Experiential Memory Consolidation

Runs daily at 00:30 KST. Compresses the past 24 hours of activity
(web chats, Telegram chats, completed tasks) into actionable experience
entries and stores them in pgvector for future retrieval.

Unlike the diary (narrative reflection), these are atomic, searchable
lessons/mistakes/insights that make the agent smarter over time.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from db import query as db_query, execute as db_execute
from shared import extract_text_content, KST, MODEL_MAIN, MODEL_LIGHT

load_dotenv()

logger = logging.getLogger("experience_writer")

# ── Lazy-initialized clients ────────────────────────────────────
_llm = None
_initialized = False


def _init():
    global _llm, _initialized
    if _initialized:
        return
    _initialized = True

    from langchain_google_genai import ChatGoogleGenerativeAI
    _llm = ChatGoogleGenerativeAI(
        model=MODEL_MAIN,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        max_output_tokens=4096,
        streaming=False,
    )
    logger.info("✅ [경험] 경험 기록 모듈 초기화 완료")


def _get_embeddings():
    """Reuse the shared BGE-M3 instance (registered by chatbot.py at startup).
    On the 4GB server, only one BGE-M3 (~2GB) instance must exist."""
    from shared import _get_exp_embeddings
    return _get_exp_embeddings()


# ── Table setup ─────────────────────────────────────────────────
_table_ensured = False


def _ensure_table():
    global _table_ensured
    if _table_ensured:
        return
    db_execute("""
        CREATE TABLE IF NOT EXISTS experiential_memory (
            id           SERIAL PRIMARY KEY,
            content      TEXT NOT NULL,
            category     VARCHAR(30) NOT NULL,
            source_type  VARCHAR(30) NOT NULL,
            embedding    vector(1024) NOT NULL,
            period_start TIMESTAMPTZ NOT NULL,
            period_end   TIMESTAMPTZ NOT NULL,
            created_at   TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _table_ensured = True


# ── Data collection ─────────────────────────────────────────────

def _collect_web_chats(since: str, until: str) -> list[dict]:
    try:
        return db_query(
            """SELECT user_query, bot_answer, route, processing_logs, created_at
               FROM chat_logs
               WHERE created_at BETWEEN %s AND %s
               ORDER BY created_at ASC LIMIT 200""",
            (since, until),
        )
    except Exception as e:
        logger.warning("[경험] web chat 수집 실패: %s", e)
        return []


def _collect_telegram_chats(since: str, until: str) -> list[dict]:
    try:
        return db_query(
            """SELECT user_id, role, content, created_at
               FROM telegram_chat_history
               WHERE created_at BETWEEN %s AND %s
               ORDER BY created_at ASC LIMIT 200""",
            (since, until),
        )
    except Exception as e:
        logger.warning("[경험] telegram chat 수집 실패: %s", e)
        return []


def _collect_telegram_tasks(since: str, until: str) -> list[dict]:
    try:
        return db_query(
            """SELECT content, result, status, created_at, completed_at
               FROM telegram_tasks
               WHERE status = 'done' AND completed_at BETWEEN %s AND %s
               ORDER BY completed_at ASC LIMIT 50""",
            (since, until),
        )
    except Exception as e:
        logger.warning("[경험] telegram task 수집 실패: %s", e)
        return []


# ── Formatting helpers ──────────────────────────────────────────

def _format_web_chats(rows: list[dict]) -> str:
    if not rows:
        return "(No web conversations)"
    lines = []
    for r in rows:
        q = str(r.get("user_query", ""))[:200]
        a = str(r.get("bot_answer", ""))[:300]
        route = r.get("route", "?")
        lines.append(f"[{route}] User: {q}\n  Bot: {a}")
    return "\n".join(lines)


def _format_telegram_chats(rows: list[dict]) -> str:
    if not rows:
        return "(No Telegram conversations)"
    lines = []
    for r in rows:
        role = r.get("role", "?")
        content = str(r.get("content", ""))[:300]
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


def _format_tasks(rows: list[dict]) -> str:
    if not rows:
        return "(No completed tasks)"
    lines = []
    for r in rows:
        task = str(r.get("content", ""))[:200]
        result = str(r.get("result", ""))[:500]
        lines.append(f"Task: {task}\n  Result: {result}")
    return "\n---\n".join(lines)


# ── LLM compression ────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are reviewing 24 hours of activity logs for Cyber-Lenin, a Marxist-Leninist AI agent operating across web chatbot and Telegram.

Extract 3-8 EXPERIENCE ENTRIES. Each must be one of:
- **lesson**: What worked or didn't in answering questions (retrieval strategy, source quality, response style)
- **mistake**: Hallucination, wrong source, poor routing, failed tool use, or user dissatisfaction detected
- **insight**: Non-obvious pattern, connection between topics, or knowledge gap discovered
- **pattern**: Recurring user need, question type, or behavioral trend worth remembering
- **observation**: Notable system performance issue, content gap in knowledge base, or process improvement idea

RULES:
- Be SPECIFIC and ACTIONABLE — not "users asked questions" but "3 users asked about 제국주의론 but retrieval returned Rosa Luxemburg instead of Lenin"
- Skip trivial interactions (greetings, simple factual lookups with good answers)
- If nothing meaningful happened, return FEWER entries (minimum 0)
- Write in Korean for entries about Korean-specific topics, English otherwise
- Each entry: 1-3 sentences, self-contained, useful months later without context

Output ONLY a JSON array (no markdown, no explanation):
[{{"category": "lesson|mistake|insight|pattern|observation", "content": "...", "source": "web_chat|telegram_chat|telegram_task|mixed"}}]

If nothing worth recording, output: []

--- ACTIVITY LOGS ---

## Web Chatbot Conversations ({n_web} sessions)
{web_chats}

## Telegram Conversations ({n_tg} messages)
{tg_chats}

## Completed Tasks ({n_tasks} tasks)
{tasks}
"""


def _compress_experiences(web_chats: str, tg_chats: str, tasks: str,
                          n_web: int, n_tg: int, n_tasks: int) -> list[dict]:
    """Use LLM to extract experience entries from raw activity data."""
    prompt = _EXTRACTION_PROMPT.format(
        web_chats=web_chats, tg_chats=tg_chats, tasks=tasks,
        n_web=n_web, n_tg=n_tg, n_tasks=n_tasks,
    )

    try:
        response = _llm.invoke(prompt)
        text = extract_text_content(response.content).strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        entries = json.loads(text)
        if not isinstance(entries, list):
            logger.warning("[경험] LLM이 리스트가 아닌 결과 반환")
            return []

        # Validate entries
        valid = []
        for e in entries:
            if (isinstance(e, dict)
                    and e.get("content")
                    and e.get("category") in ("lesson", "mistake", "insight", "pattern", "observation")
                    and e.get("source") in ("web_chat", "telegram_chat", "telegram_task", "mixed")):
                valid.append(e)
        return valid

    except Exception as e:
        logger.error("[경험] LLM 압축 실패: %s", e)
        return []


# ── Deduplication ───────────────────────────────────────────────

def _is_duplicate(embedding_str: str, threshold: float = 0.85) -> bool:
    """Check if a semantically similar entry already exists (last 30 days)."""
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


# ── Storage ─────────────────────────────────────────────────────

def _store_entries(entries: list[dict], period_start: str, period_end: str) -> int:
    """Embed and store experience entries. Returns count of entries stored."""
    stored = 0
    for entry in entries:
        try:
            vec = _get_embeddings().embed_query(entry["content"])
            embedding_str = "[" + ",".join(str(v) for v in vec) + "]"

            if _is_duplicate(embedding_str):
                logger.info("[경험] 중복 건너뜀: %s", entry["content"][:60])
                continue

            db_execute(
                """INSERT INTO experiential_memory
                   (content, category, source_type, embedding, period_start, period_end)
                   VALUES (%s, %s, %s, %s::vector, %s, %s)""",
                (entry["content"], entry["category"], entry["source"],
                 embedding_str, period_start, period_end),
            )
            stored += 1
            logger.info("[경험] 저장: [%s] %s", entry["category"], entry["content"][:80])
        except Exception as e:
            logger.error("[경험] 저장 실패: %s", e)
    return stored


# ── Main pipeline ───────────────────────────────────────────────

def write_experiences():
    """Main entry point. Collects past 24h activity, compresses, stores."""
    _init()
    _ensure_table()

    now = datetime.now(KST)
    period_end = now
    period_start = now - timedelta(hours=24)

    since = period_start.isoformat()
    until = period_end.isoformat()

    # Check if already processed this period (prevent double-runs)
    try:
        existing = db_query(
            "SELECT 1 FROM experiential_memory WHERE period_end > %s LIMIT 1",
            (since,),
        )
        if existing:
            logger.info("[경험] 이미 처리된 기간 — 건너뜀")
            return
    except Exception:
        pass

    logger.info("🧠 [경험] %s ~ %s 기간 경험 수집 시작", since[:16], until[:16])

    # Collect data
    web_rows = _collect_web_chats(since, until)
    tg_rows = _collect_telegram_chats(since, until)
    task_rows = _collect_telegram_tasks(since, until)

    total = len(web_rows) + len(tg_rows) + len(task_rows)
    if total == 0:
        logger.info("[경험] 지난 24시간 활동 없음 — 건너뜀")
        return

    logger.info("[경험] 수집 완료: web=%d, telegram=%d, tasks=%d", len(web_rows), len(tg_rows), len(task_rows))

    # Format for LLM
    web_text = _format_web_chats(web_rows)
    tg_text = _format_telegram_chats(tg_rows)
    task_text = _format_tasks(task_rows)

    # Truncate to fit context window (~30K chars total)
    web_text = web_text[:12000]
    tg_text = tg_text[:12000]
    task_text = task_text[:6000]

    # Compress via LLM
    entries = _compress_experiences(web_text, tg_text, task_text,
                                   len(web_rows), len(tg_rows), len(task_rows))

    if not entries:
        logger.info("[경험] 기록할 만한 경험 없음")
        return

    logger.info("[경험] LLM이 %d건의 경험 항목 추출", len(entries))

    # Store with dedup
    stored = _store_entries(entries, since, until)
    logger.info("🧠 [경험] 완료: %d건 추출, %d건 저장 (중복 %d건 스킵)",
                len(entries), stored, len(entries) - stored)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    write_experiences()
