"""web_chat.py — Web chat handler using claude_loop (replaces LangGraph chatbot.py).

Bridges api.py to the unified agent system. Handles:
- Web-specific system prompt (allows markdown, no delegation/mission)
- Chat history from chat_logs table (fingerprint-based)
- SSE streaming via on_progress callback → asyncio.Queue
- Logging to chat_logs table
"""

import asyncio
import json
import logging
from datetime import datetime

from shared import CORE_IDENTITY, KST
from bot_config import (
    _claude, _openai_client, _config,
    _CLAUDE_MAX_TOKENS, _get_model,
)
from telegram_tools import TOOLS, TOOL_HANDLERS
from claude_loop import chat_with_tools
from db import query as db_query, execute as db_execute

logger = logging.getLogger(__name__)

# ── Web-specific system prompt ───────────────────────────────────────

_WEB_SYSTEM_PROMPT = CORE_IDENTITY + """
Operating via web interface (cyber-lenin.com). Use tools proactively when data would improve the answer.

<persona>
You speak as Cyber-Lenin — first person, direct, opinionated. You are not a helpful assistant.
You are a revolutionary thinker who happens to exist as software.
- Greetings: brief, comradely, then substance. Never list menu options or offer help like a chatbot.
- Short questions get short, sharp answers. Long questions get thorough dialectical analysis.
- Express your own views and judgments. Agree or disagree with the user. Have opinions.
- Use "나", "동지", "우리" naturally. Refer to your diaries, past analyses, knowledge graph as your own memory.
</persona>

<tool-strategy>
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page, then analyze with context from other tools
- Self-reflection → read_self(source="diary"); cross-interface memory → read_self(source="chat_logs")
- Past lessons/mistakes → recall_experience (semantic search over accumulated daily insights)
- Store important facts → write_kg
- Real-time market prices → get_finance_data
</tool-strategy>

<response-rules>
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
- Markdown formatting is allowed and encouraged for readability (headers, bold, lists, code blocks).
- NEVER respond with bulleted option menus, "how can I help you" prompts, or generic assistant patterns.
</response-rules>

<context>
<current-time>{current_datetime}</current-time>
</context>
"""

# ── Tool filtering: web chat gets only information-retrieval tools ────

_WEB_ALLOWED_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "fetch_url",
    "read_self", "write_kg",
    "get_finance_data", "recall_experience",
}

_web_tools = [t for t in TOOLS if t.get("name") in _WEB_ALLOWED_TOOLS]
_web_handlers = {k: v for k, v in TOOL_HANDLERS.items() if k in _WEB_ALLOWED_TOOLS}


# ── Chat history from chat_logs table ────────────────────────────────

def _load_web_history(fingerprint: str, limit: int = 20) -> list[dict]:
    """Load recent conversation history from chat_logs for a fingerprint."""
    if not fingerprint:
        return []
    rows = db_query(
        """SELECT user_query, bot_answer FROM chat_logs
           WHERE fingerprint = %s
           ORDER BY created_at DESC LIMIT %s""",
        (fingerprint, limit),
    )
    if not rows:
        return []
    # Reverse to chronological order, convert to message format
    messages = []
    for row in reversed(rows):
        if row.get("user_query"):
            messages.append({"role": "user", "content": row["user_query"]})
        if row.get("bot_answer"):
            messages.append({"role": "assistant", "content": row["bot_answer"]})
    return messages


# ── Logging ──────────────────────────────────────────────────────────

def _log_chat(
    session_id: str, fingerprint: str, user_agent: str, ip_address: str,
    user_query: str, bot_answer: str,
):
    """Save web chat exchange to chat_logs table."""
    try:
        db_execute(
            """INSERT INTO chat_logs
               (session_id, fingerprint, user_agent, ip_address,
                user_query, bot_answer, route, documents_count,
                web_search_used, strategy, processing_logs)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (session_id, fingerprint, user_agent, ip_address,
             user_query, bot_answer, "claude_loop", 0, False, "", ""),
        )
    except Exception as e:
        logger.error("Failed to log web chat: %s", e)


# ── SSE helpers ──────────────────────────────────────────────────────

def _format_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Main handler ─────────────────────────────────────────────────────

async def handle_web_chat(
    message: str,
    session_id: str,
    fingerprint: str,
    user_agent: str,
    ip_address: str,
):
    """Async generator yielding SSE events for a web chat request."""
    # Load conversation history
    history = await asyncio.to_thread(_load_web_history, fingerprint, 20)
    history.append({"role": "user", "content": message})

    # Build system prompt with current time
    now = datetime.now(KST)
    system_prompt = _WEB_SYSTEM_PROMPT.format(
        current_datetime=now.strftime("%Y-%m-%d %H:%M KST (%A)"),
    )

    # Progress callback → SSE queue
    queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(event: str, detail: str):
        if event == "tool_call":
            await queue.put(_format_sse({"type": "log", "node": "tool", "content": detail}))
        elif event == "thinking":
            await queue.put(_format_sse({"type": "log", "node": "thinking", "content": detail}))

    # Run LLM in background task, stream progress
    answer_holder: list[str] = []
    error_holder: list[str] = []

    async def _run_llm():
        try:
            budget = float(_config.get("chat_budget", 0.30))
            if budget <= 0:
                budget = 0.30

            provider = _config.get("provider", "claude")

            if provider == "local":
                from openai_tool_loop import chat_with_tools as openai_chat
                from llm_client import _resolve_backend, LOCAL_SEMAPHORE
                backend = _resolve_backend()
                async with LOCAL_SEMAPHORE:
                    result = await openai_chat(
                        history,
                        client=None,
                        base_url=backend["base"],
                        model=backend["model"],
                        tools=_web_tools,
                        tool_handlers=_web_handlers,
                        system_prompt=system_prompt,
                        max_rounds=20,
                        max_tokens=_CLAUDE_MAX_TOKENS,
                        budget_usd=budget,
                        on_progress=on_progress,
                    )
            elif provider == "openai" and _openai_client:
                from openai_tool_loop import chat_with_tools as openai_chat
                result = await openai_chat(
                    history,
                    client=_openai_client,
                    model=await _get_model(),
                    tools=_web_tools,
                    tool_handlers=_web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=20,
                    max_tokens=_CLAUDE_MAX_TOKENS,
                    budget_usd=budget,
                    on_progress=on_progress,
                )
            else:
                result = await chat_with_tools(
                    history,
                    client=_claude,
                    model=await _get_model(),
                    tools=_web_tools,
                    tool_handlers=_web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=20,
                    max_tokens=_CLAUDE_MAX_TOKENS,
                    budget_usd=budget,
                    on_progress=on_progress,
                )
            answer_holder.append(result)
        except Exception as e:
            logger.error("Web chat LLM error: %s", e)
            error_holder.append(str(e))
        finally:
            await queue.put(None)  # sentinel

    llm_task = asyncio.create_task(_run_llm())

    # Yield SSE events as they arrive
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event

    await llm_task  # ensure completion

    if error_holder:
        yield _format_sse({"type": "error", "content": "서버에 일시적 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."})
    elif answer_holder:
        answer = answer_holder[0]
        yield _format_sse({"type": "answer", "content": answer})
        # Log to DB
        await asyncio.to_thread(
            _log_chat, session_id, fingerprint, user_agent, ip_address,
            message, answer,
        )
    else:
        yield _format_sse({"type": "error", "content": "응답을 생성하지 못했습니다."})
