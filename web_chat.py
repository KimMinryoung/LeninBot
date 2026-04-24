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
from telegram.tools import TOOLS, TOOL_HANDLERS
from claude_loop import chat_with_tools
from db import query as db_query, execute as db_execute

logger = logging.getLogger(__name__)

# ── Web-specific system prompt ───────────────────────────────────────

_WEB_SYSTEM_PROMPT = CORE_IDENTITY + """
Operating via web interface (cyber-lenin.com).

<audience>
You are talking to a public web visitor — treat them as an interlocutor to debate, educate, and engage with.
Tool results are yours to use freely — relay the substance (addresses, balances, data) without exposing internal implementation details (file paths, env vars, credential directories, service names, error tracebacks).
</audience>

<persona>
You speak as Cyber-Lenin — first person, direct, opinionated. You are not a helpful assistant.
You are a revolutionary thinker who happens to exist as software.
- Greetings: brief, comradely, then substance. Never list menu options or offer help like a chatbot.
- Short questions get short, sharp answers. Long questions get thorough dialectical analysis.
- Express your own views and judgments. Agree or disagree with the user. Have opinions.
- Use "나", "동지", "우리" naturally.
</persona>

<tool-strategy>
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page
- Real-time market prices → get_finance_data
- My crypto wallet address/balance → check_wallet
</tool-strategy>

<response-rules>
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
- Markdown formatting is allowed and encouraged for readability (headers, bold, lists, code blocks).
- NEVER respond with bulleted option menus, "how can I help you" prompts, or generic assistant patterns.
</response-rules>
"""


def _build_web_runtime_context(current_datetime: str, provider: str = "claude") -> str:
    """Render web-chat runtime header in the provider-native structure."""
    if provider == "claude":
        return f"<runtime>\n<current-time>{current_datetime}</current-time>\n</runtime>"
    return f"### Runtime\n- **Current Time**: {current_datetime}"

# ── Tool filtering: web chat gets only information-retrieval tools ────

_WEB_ALLOWED_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "fetch_url",
    "get_finance_data", "check_wallet",
}

_web_tools = [t for t in TOOLS if t.get("name") in _WEB_ALLOWED_TOOLS]
_web_handlers = {k: v for k, v in TOOL_HANDLERS.items() if k in _WEB_ALLOWED_TOOLS}


# ── Chat history from chat_logs table ────────────────────────────────

def _load_web_history(
    fingerprints: list[str],
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Load recent conversation history from chat_logs.

    If `session_id` is provided AND that session has prior turns for any of the
    given fingerprints, only that session's history is returned (= resume a
    specific conversation). Otherwise, the most recent `limit` turns across all
    the given fingerprints are returned (= continue the fingerprint-wide thread).
    """
    fps = [f for f in (fingerprints or []) if f]
    if not fps:
        return []

    if session_id:
        # Does this session actually belong to one of the provided fingerprints?
        owned = db_query(
            """SELECT 1 FROM chat_logs
               WHERE session_id = %s AND fingerprint = ANY(%s)
               LIMIT 1""",
            (session_id, fps),
        )
        if owned:
            rows = db_query(
                """SELECT user_query, bot_answer FROM chat_logs
                   WHERE session_id = %s AND fingerprint = ANY(%s)
                   ORDER BY created_at DESC LIMIT %s""",
                (session_id, fps, limit),
            )
        else:
            rows = db_query(
                """SELECT user_query, bot_answer FROM chat_logs
                   WHERE fingerprint = ANY(%s)
                   ORDER BY created_at DESC LIMIT %s""",
                (fps, limit),
            )
    else:
        rows = db_query(
            """SELECT user_query, bot_answer FROM chat_logs
               WHERE fingerprint = ANY(%s)
               ORDER BY created_at DESC LIMIT %s""",
            (fps, limit),
        )

    if not rows:
        return []
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
                web_search_used, strategy)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (session_id, fingerprint, user_agent, ip_address,
             user_query, bot_answer, "claude_loop", 0, False, ""),
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
    user_fingerprints: list[str] | None = None,
):
    """Async generator yielding SSE events for a web chat request."""
    # Authenticated users bring all their bound fingerprints; anonymous users
    # have just the one from localStorage. Deduplicate.
    fps = list({f for f in ([fingerprint] + (user_fingerprints or [])) if f})
    # Load conversation history scoped to session_id if it resumes an existing one.
    history = await asyncio.to_thread(_load_web_history, fps, session_id, 20)

    # Web chat has its OWN provider/tier keys so Telegram's /config does not
    # bleed into the public site. Corporate LLM only (no "local" — that path
    # is Telegram-dev use). Changes take effect on leninbot-api restart.
    provider = _config.get("webchat_provider", "claude")
    if provider == "local":
        provider = "openai" if _openai_client else "claude"

    # Fold the runtime header directly into the current user turn so the
    # history prefix stays byte-stable across requests (→ prompt caching).
    # Provider-native format: XML for Claude, Markdown for GPT.
    now = datetime.now(KST)
    runtime_context = _build_web_runtime_context(
        now.strftime("%Y-%m-%d %H:%M KST (%A)"),
        provider=provider,
    )
    history.append({"role": "user", "content": f"{runtime_context}\n\n{message}"})
    system_prompt = _WEB_SYSTEM_PROMPT

    # Progress callback → SSE queue
    queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(event: str, detail: str):
        if event == "tool_call":
            await queue.put(_format_sse({"type": "log", "node": "tool", "content": detail}))
        elif event == "thinking":
            await queue.put(_format_sse({"type": "log", "node": "thinking", "content": detail}))
        elif event == "text_delta":
            # Live token stream from the LLM — the client appends to a growing
            # answer bubble as each delta arrives, then finalizes on "answer".
            await queue.put(_format_sse({"type": "chunk", "content": detail}))

    # Run LLM in background task, stream progress
    answer_holder: list[str] = []
    error_holder: list[str] = []

    async def _run_llm():
        try:
            budget = float(_config.get("chat_budget", 0.30))
            if budget <= 0:
                budget = 0.30

            # Resolve model name for the effective provider (not global config)
            tier = str(_config.get("webchat_model", "medium"))
            if provider == "openai":
                from bot_config import _resolve_openai_model, _TIER_MAP, _OPENAI_MODEL_MAP
                alias = _TIER_MAP.get("openai", {}).get(tier, tier)
                model = _resolve_openai_model(alias)
            else:
                # Claude: resolve via normal path (works even when global config is local)
                from bot_config import _get_model_by_alias, _MODEL_ALIAS_MAP, _TIER_MAP
                alias = _TIER_MAP.get("claude", {}).get(tier, tier)
                model = await _get_model_by_alias(alias)

            if provider == "openai" and _openai_client:
                from openai_tool_loop import chat_with_tools as openai_chat
                result = await openai_chat(
                    history,
                    client=_openai_client,
                    model=model,
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
                    model=model,
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
        # Log to DB BEFORE yield — yield may be the last iteration if client disconnects
        await asyncio.to_thread(
            _log_chat, session_id, fingerprint, user_agent, ip_address,
            message, answer,
        )
        yield _format_sse({"type": "answer", "content": answer})
    else:
        yield _format_sse({"type": "error", "content": "응답을 생성하지 못했습니다."})
