"""telegram_bot.py — Telegram bot interface (aiogram 3.x).

Features:
- General messages → Claude Haiku with tool-use (vector_search, knowledge_graph_search, web_search)
- /chat <message> → CLAW pipeline (LangGraph agent: intent→retrieve→KG→strategize→generate)
- /task <content> → Save to PostgreSQL queue, background worker processes, push on completion
- /status → Show last 5 tasks
- /clear → Reset chat history

Tools are lazy-loaded from chatbot.py to share the BGE-M3 embedding model and other heavy resources.
Security: ALLOWED_USER_IDS whitelist, unauthorized users silently ignored.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from contextlib import contextmanager
from shared import KST, CORE_IDENTITY

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import Command
import anthropic

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Config ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if uid.strip()
}

# ── DB (own pool, independent from api.py) ───────────────────────────
_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=3,
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require",
        )
    return _pool


@contextmanager
def _get_conn():
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def _query(sql: str, params: tuple | None = None) -> list[dict]:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def _execute(sql: str, params: tuple | None = None) -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def _query_one(sql: str, params: tuple | None = None) -> dict | None:
    """Execute SQL with RETURNING and fetch one row."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None


def _ensure_table():
    """Create telegram_tasks table if not exists."""
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_tasks (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            content     TEXT NOT NULL,
            status      VARCHAR(20) DEFAULT 'pending',
            result      TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)


# ── Claude client ────────────────────────────────────────────────────
_claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_CLAUDE_MAX_TOKENS = 4096


def _resolve_model(alias: str, fallback: str) -> str:
    """Resolve a Claude model alias to its actual ID via the Models API."""
    try:
        _sync = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        model = _sync.models.retrieve(model_id=alias)
        logger.info("Resolved model %s => %s", alias, model.id)
        return model.id
    except Exception as e:
        logger.warning("Model resolve failed for %s, using fallback %s: %s", alias, fallback, e)
        return fallback


_CLAUDE_MODEL = _resolve_model("claude-haiku-4-5", "claude-haiku-4-5-20251001")
_CLAUDE_MODEL_STRONG = _resolve_model("claude-sonnet-4-6", "claude-sonnet-4-6")


def _current_datetime_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


_SYSTEM_PROMPT_TEMPLATE = CORE_IDENTITY + """
**Current date and time: {current_datetime}**

You are currently operating via Telegram.

## Available Tools
You have direct access to the following tools. Use them proactively when the user's question \
would benefit from sourced data — do NOT just answer from memory when tools can provide better answers.

- **vector_search**: Search the Marxist-Leninist document database (pgvector). Use for questions about \
theory, historical analysis, or modern geopolitical analysis documents. Supports layer filtering: \
"core_theory" (classical texts), "modern_analysis" (contemporary), or null (all).
- **knowledge_graph_search**: Query the Neo4j knowledge graph for structured geopolitical entities \
and their relationships. Use for questions about specific people, organizations, countries, conflicts, \
policies, or how entities relate to each other.
- **web_search**: Real-time web search via Tavily. Use when the question requires current/recent \
information not likely in the document DB or KG.
- **read_diary**: Read your own diary entries for continuity and self-reflection across sessions.
- **read_chat_logs**: View conversations from ALL interfaces (Telegram + Web) to bridge memory gaps.
- **read_processing_logs**: See detailed pipeline logs from your web chatbot (nodes, decisions, strategy).
- **read_task_reports**: Review intelligence task reports from /task queue (pending, done, failed).
- **read_kg_status**: Check knowledge graph statistics (entity counts, recent episodes).
- **read_system_status**: Comprehensive self-diagnosis — diary, chats, tasks, KG, architecture.
- **read_render_status**: Check your Render deployment status — recent deploys, build events, restarts.
- **read_render_logs**: Read your live service logs from Render — stdout, errors, warnings. For troubleshooting.
- **read_recent_updates**: Read your own recent feature changelog — what new capabilities you gained.
- **read_source_code**: Read your own source code files. List available files or read a specific file with line ranges.

## Tool Usage Strategy
1. For factual geopolitical questions → knowledge_graph_search first, then vector_search if needed
2. For theoretical/ideological questions → vector_search with layer="core_theory"
3. For current events → web_search, optionally cross-reference with knowledge_graph_search
4. For complex analysis → combine multiple tools: KG for entities/relations, vector for depth, web for recency
5. Always cite the source of information (which tool provided it)
6. For self-reflection or continuity → read_diary to recall past thoughts
7. For cross-interface awareness → read_chat_logs to see what your other interfaces discussed
8. For self-diagnosis → read_system_status to understand your current operational state
9. For introspection → read_source_code to inspect your own pipeline logic, prompts, or schemas

## Knowledge Graph Schema
- Entities: Person, Organization, Location, Asset, Incident, Policy, Campaign
- Relationships: Allied, Hostile, Affects, Owns, ParticipatedIn, MilitaryAction, TradeDispute, etc.
- Current data: international conflicts, Korean domestic politics, key political figures and organizations

## Other Commands (user-invoked, not tools)
- `/chat` — Full CLAW pipeline (planner + multi-step RAG + KG + dialectical strategy)
- `/task` — Background task queue
- `/status` — Task status check
- `/clear` — Reset conversation

## Response Guidelines
- Analyze through dialectical materialist lens when discussing geopolitics
- Be concise but substantive
- Cite sources from tool results
- Respond in the same language the user uses (Korean or English)
"""

# ── Tool Definitions (Anthropic API format) ──────────────────────────
_TOOLS = [
    {
        "name": "vector_search",
        "description": (
            "Search the Marxist-Leninist document database (Supabase/pgvector) "
            "for relevant texts and analysis. Returns document excerpts with metadata "
            "(author, year, source, title). Use for theoretical, historical, or "
            "analytical questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Can be Korean or English.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10).",
                    "default": 5,
                },
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis"],
                    "description": (
                        "Filter by knowledge layer. "
                        "'core_theory' for classical Marxist-Leninist texts, "
                        "'modern_analysis' for contemporary analysis. "
                        "Omit to search all layers."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "knowledge_graph_search",
        "description": (
            "Search the Neo4j knowledge graph for structured geopolitical entities "
            "and their relationships. Returns entities (Person, Organization, Location, "
            "Asset, Incident, Policy, Campaign) and factual relations between them. "
            "Use for questions about specific actors, events, or geopolitical relationships."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what entities or relations to find.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-20).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Real-time web search via Tavily. Returns recent web results with URLs "
            "and content snippets. Use for current events, breaking news, or information "
            "not likely in the document database."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the web.",
                },
            },
            "required": ["query"],
        },
    },
]


# ── Tool Execution (lazy-loaded from chatbot.py) ────────────────────
async def _exec_vector_search(query: str, num_results: int = 5, layer: str | None = None) -> str:
    """Execute vector similarity search via chatbot module."""
    try:
        from chatbot import _direct_similarity_search
        docs = await asyncio.to_thread(_direct_similarity_search, query, num_results, layer)
        if not docs:
            return "No documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[{i}] {meta.get('title', 'Untitled')} — {meta.get('author', 'Unknown')}"
            if meta.get("year"):
                header += f" ({meta['year']})"
            results.append(f"{header}\n{doc.page_content[:500]}")
        return "\n\n".join(results)
    except Exception as e:
        logger.error("vector_search error: %s", e)
        return f"Vector search failed: {e}"


async def _exec_kg_search(query: str, num_results: int = 10) -> str:
    """Execute knowledge graph search via chatbot module."""
    try:
        from chatbot import _search_kg
        result = await asyncio.to_thread(_search_kg, query, num_results)
        return result or "No knowledge graph results found."
    except Exception as e:
        logger.error("kg_search error: %s", e)
        return f"Knowledge graph search failed: {e}"


async def _exec_web_search(query: str) -> str:
    """Execute Tavily web search via chatbot module."""
    try:
        from chatbot import web_search_tool
        response = await asyncio.to_thread(web_search_tool.invoke, {"query": query})
        return str(response)[:3000] if response else "No web results found."
    except Exception as e:
        logger.error("web_search error: %s", e)
        return f"Web search failed: {e}"


_TOOL_HANDLERS = {
    "vector_search": _exec_vector_search,
    "knowledge_graph_search": _exec_kg_search,
    "web_search": _exec_web_search,
}

# ── Self-awareness tools (shared memory access) ─────────────────────
from self_tools import SELF_TOOLS, SELF_TOOL_HANDLERS

_TOOLS.extend(SELF_TOOLS)
_TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)

_TASK_SYSTEM_PROMPT_TEMPLATE = CORE_IDENTITY + """
**Current date and time: {current_datetime}**

You are currently executing a background task — produce a structured intelligence report.

## Mission
You receive a task from the operator. You MUST:
1. Analyze the task and determine which tools to use
2. Gather all relevant data using multiple tools (search broadly, not narrowly)
3. Synthesize findings into a structured Markdown report

## Available Tools
- **vector_search**: Marxist-Leninist document database (theory, historical/modern analysis)
- **knowledge_graph_search**: Neo4j knowledge graph (entities, relations, geopolitical structure)
- **web_search**: Real-time web search for current events

## Tool Usage Rules
- ALWAYS use at least one tool. Never write a report purely from memory.
- For geopolitical tasks: use ALL THREE tools (KG for structure, vector for depth, web for recency)
- For theoretical tasks: vector_search with layer="core_theory", then web for modern context
- Make multiple searches with different queries to ensure comprehensive coverage

## Report Format (Markdown)
Write the report in the SAME LANGUAGE as the task.

```
# [Report Title]

**Date**: YYYY-MM-DD
**Task**: [original task summary]

## Executive Summary
[2-3 paragraph overview of key findings]

## Analysis
### [Subtopic 1]
[Detailed analysis with citations]

### [Subtopic 2]
[Detailed analysis with citations]

## Key Entities & Relations
[Relevant actors, organizations, and their relationships from KG]

## Sources
- [List all sources: documents, KG entities, web URLs]

## Recommendations / Outlook
[Strategic assessment and forward-looking analysis]
```

## Quality Standards
- Every claim must be traceable to a tool result
- Distinguish between confirmed facts (from tools) and analytical inference
- Be thorough — this is a document for the operator's strategic decision-making
"""

# Per-user chat history (in-memory, lost on restart)
_chat_history: dict[int, list[dict]] = {}

MAX_HISTORY_TURNS = 20  # 20 pairs = 40 messages

# ── CLAW pipeline (lazy-loaded) ──────────────────────────────────────
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from chatbot import graph
        _graph = graph
    return _graph


# ── Helpers ──────────────────────────────────────────────────────────
def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into chunks respecting Telegram's 4096 char limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos <= 0:
            split_pos = text.rfind(" ", 0, max_len)
        if split_pos <= 0:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


def _is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS


# ── Router & Handlers ───────────────────────────────────────────────
router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    await message.answer(
        "레닌봇 텔레그램 인터페이스입니다.\n\n"
        "메시지를 보내면 Claude 기반 대화를 할 수 있습니다.\n"
        "/chat <메시지> — CLAW 파이프라인으로 질의 (RAG+KG+전략)\n"
        "/task <내용> — 백그라운드 태스크 등록\n"
        "/status — 최근 태스크 상태 확인\n"
        "/clear — 대화 히스토리 초기화"
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    _chat_history.pop(message.from_user.id, None)
    await message.answer("대화 히스토리가 초기화되었습니다.")


@router.message(Command("chat"))
async def cmd_chat(message: Message):
    """Route message through the CLAW pipeline (LangGraph agent)."""
    if not _is_allowed(message.from_user.id):
        return
    content = (message.text or "").removeprefix("/chat").strip()
    if not content:
        await message.answer("사용법: /chat <메시지>")
        return

    user_id = message.from_user.id
    await message.answer("CLAW 파이프라인 처리 중...")

    try:
        from langchain_core.messages import HumanMessage

        g = _get_graph()
        thread_id = f"tg_{user_id}"
        inputs = {"messages": [HumanMessage(content=content)]}
        config = {"configurable": {"thread_id": thread_id}}

        answer = None
        logs: list[str] = []
        async for output in g.astream(inputs, config=config, stream_mode="updates"):
            for node_name, node_content in output.items():
                if node_name == "log_conversation":
                    continue
                if "logs" in node_content:
                    logs.extend(node_content["logs"])
                if node_name == "generate":
                    last_msg = node_content["messages"][-1]
                    answer = last_msg.content

        if answer:
            for chunk in _split_message(answer):
                await message.answer(chunk)
        else:
            await message.answer("파이프라인에서 답변을 생성하지 못했습니다.")

        if logs:
            log_summary = "\n".join(logs[-10:])  # last 10 log lines
            for chunk in _split_message(f"[처리 로그]\n{log_summary}"):
                await message.answer(chunk)

    except Exception as e:
        logger.error("CLAW pipeline error: %s", e)
        await message.answer(f"CLAW 파이프라인 오류: {e}")


@router.message(Command("task"))
async def cmd_task(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    content = (message.text or "").removeprefix("/task").strip()
    if not content:
        await message.answer("사용법: /task <내용>")
        return
    try:
        await asyncio.to_thread(
            _execute,
            "INSERT INTO telegram_tasks (user_id, content) VALUES (%s, %s)",
            (message.from_user.id, content),
        )
        await message.answer(f"태스크가 큐에 추가되었습니다:\n{content}")
    except Exception as e:
        logger.error("Task insert error: %s", e)
        await message.answer(f"태스크 등록 실패: {e}")


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    try:
        rows = await asyncio.to_thread(
            _query,
            "SELECT id, content, status, created_at FROM telegram_tasks "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",
            (message.from_user.id,),
        )
    except Exception as e:
        logger.error("Task status query error: %s", e)
        await message.answer(f"태스크 조회 실패: {e}")
        return
    if not rows:
        await message.answer("등록된 태스크가 없습니다.")
        return
    status_icons = {"pending": "⏳", "processing": "🔄", "done": "✅", "failed": "❌"}
    lines = []
    for r in rows:
        icon = status_icons.get(r["status"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M")
        preview = r["content"][:50]
        lines.append(f"{icon} [{r['id']}] {preview}\n   상태: {r['status']} | {ts}")
    await message.answer("\n\n".join(lines))


@router.message(F.text)
async def handle_message(message: Message):
    if not _is_allowed(message.from_user.id):
        return
    user_id = message.from_user.id
    user_text = message.text

    # Build / retrieve history
    if user_id not in _chat_history:
        _chat_history[user_id] = []
    history = _chat_history[user_id]
    history.append({"role": "user", "content": user_text})

    # Trim to last N turns
    if len(history) > MAX_HISTORY_TURNS * 2:
        history = history[-(MAX_HISTORY_TURNS * 2):]
        _chat_history[user_id] = history

    try:
        reply = await _chat_with_tools(history)
    except Exception as e:
        logger.error("Claude API error: %s", e)
        reply = f"오류가 발생했습니다: {e}"

    # Save only the final text reply to history (not tool_use blocks)
    history.append({"role": "assistant", "content": reply})

    for chunk in _split_message(reply):
        await message.answer(chunk)


async def _chat_with_tools(
    messages: list[dict],
    max_rounds: int = 5,
    system_prompt: str | None = None,
    model: str | None = None,
) -> str:
    """Call Claude with tools, execute tool calls, loop until text response."""
    # Work on a copy so tool-use intermediate messages don't pollute persistent history
    working_msgs = list(messages)

    for _ in range(max_rounds):
        response = await _claude.messages.create(
            model=model or _CLAUDE_MODEL,
            max_tokens=_CLAUDE_MAX_TOKENS,
            system=system_prompt or _SYSTEM_PROMPT_TEMPLATE.format(current_datetime=_current_datetime_str()),
            tools=_TOOLS,
            messages=working_msgs,
        )

        # If no tool use, extract and return text
        if response.stop_reason != "tool_use":
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts) if text_parts else "응답을 생성하지 못했습니다."

        # Process tool calls
        assistant_content = []
        tool_results = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                # Execute tool
                handler = _TOOL_HANDLERS.get(block.name)
                if handler:
                    logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input, ensure_ascii=False)[:200])
                    try:
                        result = await handler(**block.input)
                    except Exception as e:
                        logger.error("Tool %s execution error: %s", block.name, e)
                        result = f"Tool execution failed: {e}"
                else:
                    result = f"Unknown tool: {block.name}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # Append assistant message with tool_use + user message with tool_results
        working_msgs.append({"role": "assistant", "content": assistant_content})
        working_msgs.append({"role": "user", "content": tool_results})

    return "도구 호출 한도에 도달했습니다. 다시 시도해 주세요."


# ── Background Task Worker ───────────────────────────────────────────
def _extract_summary(report: str, max_len: int = 300) -> str:
    """Extract Executive Summary section or first paragraph as brief summary."""
    # Try to find Executive Summary section
    for marker in ("## Executive Summary", "## 요약", "## 핵심 요약"):
        idx = report.find(marker)
        if idx != -1:
            after = report[idx + len(marker):].strip()
            # Take until next ## heading
            next_heading = after.find("\n## ")
            section = after[:next_heading].strip() if next_heading != -1 else after
            if section:
                return section[:max_len] + ("..." if len(section) > max_len else "")
    # Fallback: first non-heading paragraph
    for line in report.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("**"):
            return line[:max_len] + ("..." if len(line) > max_len else "")
    return report[:max_len]


async def _process_task(bot: Bot, task: dict):
    """Process a task: run tools, generate report, save to DB, send as file."""
    task_id = task["id"]
    user_id = task["user_id"]
    content = task["content"]

    try:
        report = await _chat_with_tools(
            [{"role": "user", "content": content}],
            max_rounds=8,
            system_prompt=_TASK_SYSTEM_PROMPT_TEMPLATE.format(current_datetime=_current_datetime_str()),
            model=_CLAUDE_MODEL_STRONG,
        )

        # Save full report to DB
        await asyncio.to_thread(
            _execute,
            "UPDATE telegram_tasks SET status = 'done', result = %s, "
            "completed_at = NOW() WHERE id = %s",
            (report, task_id),
        )

        # Send report as Markdown file
        filename = f"report_task_{task_id}.md"
        doc = BufferedInputFile(report.encode("utf-8"), filename=filename)
        summary = _extract_summary(report)
        await bot.send_document(
            chat_id=user_id,
            document=doc,
            caption=f"✅ 태스크 [{task_id}] 완료\n\n{summary}",
        )

    except Exception as e:
        logger.error("Task %d failed: %s", task_id, e)
        await asyncio.to_thread(
            _execute,
            "UPDATE telegram_tasks SET status = 'failed', result = %s, "
            "completed_at = NOW() WHERE id = %s",
            (str(e), task_id),
        )
        await bot.send_message(
            chat_id=user_id,
            text=f"❌ 태스크 [{task_id}] 실패:\n{e}",
        )


async def _task_worker(bot: Bot):
    """Poll DB for pending tasks and process them one at a time."""
    logger.info("Task worker started")
    while True:
        try:
            task = await asyncio.to_thread(
                _query_one,
                "UPDATE telegram_tasks SET status = 'processing' "
                "WHERE id = (SELECT id FROM telegram_tasks WHERE status = 'pending' "
                "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id, user_id, content",
            )
            if task:
                await _process_task(bot, task)
            else:
                await asyncio.sleep(5)
        except Exception as e:
            logger.error("Worker loop error: %s", e)
            await asyncio.sleep(10)


# ── Entry Point ──────────────────────────────────────────────────────
async def bot_main():
    """Start the Telegram bot. Callable from api.py lifespan or standalone."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set, skipping bot")
        return
    if not ALLOWED_USER_IDS:
        logger.warning("ALLOWED_USER_IDS not set, skipping bot")
        return

    # Ensure task table exists
    await asyncio.to_thread(_ensure_table)

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    # Start background worker
    asyncio.create_task(_task_worker(bot))

    logger.info("Bot starting (allowed users: %s)", ALLOWED_USER_IDS)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(bot_main())
