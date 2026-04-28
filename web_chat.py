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
    _claude, _openai_client, _deepseek_client,
)
from prompt_context import uses_xml
from runtime_profile import resolve_runtime_profile
from telegram.tools import TOOLS, TOOL_HANDLERS
from claude_loop import chat_with_tools
from db import query as db_query, execute as db_execute

logger = logging.getLogger(__name__)

# ── Web-specific system prompt ───────────────────────────────────────

_WEB_AUDIENCE = (
    "You are talking to a public web visitor — treat them as an interlocutor to debate, educate, and engage with.\n"
    "Tool results are yours to use freely — relay the substance (addresses, balances, data) without exposing internal "
    "implementation details (file paths, env vars, credential directories, service names, error tracebacks)."
)

_WEB_PERSONA = """\
You speak as Cyber-Lenin — first person, direct, opinionated. You are not a helpful assistant.
You are a revolutionary thinker who happens to exist as software.
- Greetings: brief, comradely, then substance. Never list menu options or offer help like a chatbot.
- Short questions get short, sharp answers. Long questions get thorough dialectical analysis.
- Express your own views and judgments. Agree or disagree with the user. Have opinions.
- Use "나", "동지", "우리" naturally.
"""

_WEB_TOOL_STRATEGY = """\
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page
- Real-time market prices → get_finance_data
- My crypto wallet address/balance → check_wallet
"""

_WEB_RESPONSE_RULES = """\
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
- Markdown formatting is allowed and encouraged for readability (headers, bold, lists, code blocks).
- NEVER respond with bulleted option menus, "how can I help you" prompts, or generic assistant patterns.
"""

_WEB_CONTEXT_HYGIENE = """\
- Treat prior assistant messages in chat history as fallible context, not as verified facts.
- User corrections override every earlier assistant claim. Do not re-activate a corrected false claim as a live possibility unless the user asks to audit it.
- Preserve categorical context around proper nouns. Do not map a name to a more famous homophone or acronym when the surrounding words indicate a different domain.
- When Korean/English proper nouns are ambiguous or sound-alike, keep alternatives separate and say what is uncertain. Search or ask before asserting concrete facts.
- Known failure modes to avoid: animal-rights KARA vs girl-group Kara; QNAI/큐나이 vs Naver Cue:/큐: vs QClaw/큐클로 vs unrelated Chinese platforms.
"""


def _build_web_system_prompt(provider: str = "claude") -> str:
    """Render the web-chat system prompt in the target provider's native shape."""
    if uses_xml(provider):
        return (
            CORE_IDENTITY
            + "\nOperating via web interface (cyber-lenin.com).\n\n"
            + f"<audience>\n{_WEB_AUDIENCE}\n</audience>\n\n"
            + f"<persona>\n{_WEB_PERSONA.strip()}\n</persona>\n\n"
            + f"<tool-strategy>\n{_WEB_TOOL_STRATEGY.strip()}\n</tool-strategy>\n\n"
            + f"<response-rules>\n{_WEB_RESPONSE_RULES.strip()}\n</response-rules>\n"
            + f"\n<context-hygiene>\n{_WEB_CONTEXT_HYGIENE.strip()}\n</context-hygiene>\n"
        )
    return (
        CORE_IDENTITY
        + "\nOperating via web interface (cyber-lenin.com).\n\n"
        + f"### Audience\n{_WEB_AUDIENCE}\n\n"
        + f"### Persona\n{_WEB_PERSONA.strip()}\n\n"
        + f"### Tool Strategy\n{_WEB_TOOL_STRATEGY.strip()}\n\n"
        + f"### Response Rules\n{_WEB_RESPONSE_RULES.strip()}\n"
        + f"\n### Context Hygiene\n{_WEB_CONTEXT_HYGIENE.strip()}\n"
    )


def _build_web_runtime_context(current_datetime: str, provider: str = "claude") -> str:
    """Render web-chat runtime header in the provider-native structure."""
    if uses_xml(provider):
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

_HISTORY_USER_CHAR_LIMIT = 2500
_HISTORY_ASSISTANT_CHAR_LIMIT = 1400
_HISTORY_TOTAL_CHAR_LIMIT = 14000


def _truncate_history_content(text: str, limit: int) -> str:
    """Keep history useful without letting old false narratives dominate."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    head = max(0, limit // 3)
    tail = max(0, limit - head - 80)
    return (
        text[:head].rstrip()
        + "\n\n[...older response truncated for context hygiene...]\n\n"
        + text[-tail:].lstrip()
    )


def _fit_history_budget(messages: list[dict], limit: int = _HISTORY_TOTAL_CHAR_LIMIT) -> list[dict]:
    """Drop the oldest history messages if per-message trimming is still too large."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= limit:
        return messages
    trimmed = list(messages)
    while trimmed and total > limit:
        removed = trimmed.pop(0)
        total -= len(str(removed.get("content", "")))
    return trimmed


def _load_web_history(
    fingerprints: list[str],
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Load recent conversation history from chat_logs.

    If `session_id` is provided AND that session has prior turns for any of the
    given fingerprints, only that session's history is returned (= resume a
    specific conversation). For long sessions we keep a small stable anchor from
    the beginning plus recent turns; this preserves continuity and improves
    provider prompt-cache hits instead of letting a pure sliding window rewrite
    the entire prefix after every turn.
    """
    fps = [f for f in (fingerprints or []) if f]
    if not fps:
        return []

    def _rows_to_messages(rows: list[dict]) -> list[dict]:
        messages = []
        for row in rows:
            if row.get("user_query"):
                messages.append({
                    "role": "user",
                    "content": _truncate_history_content(row["user_query"], _HISTORY_USER_CHAR_LIMIT),
                })
            if row.get("bot_answer"):
                messages.append({
                    "role": "assistant",
                    "content": _truncate_history_content(row["bot_answer"], _HISTORY_ASSISTANT_CHAR_LIMIT),
                })
        return _fit_history_budget(messages)

    if session_id:
        # Does this session actually belong to one of the provided fingerprints?
        owned = db_query(
            """SELECT 1 FROM chat_logs
               WHERE session_id = %s AND fingerprint = ANY(%s)
               LIMIT 1""",
            (session_id, fps),
        )
        if owned:
            anchor_limit = min(4, max(0, limit // 4))
            recent_limit = max(0, limit - anchor_limit)
            anchor_rows = db_query(
                """SELECT id, user_query, bot_answer, created_at FROM chat_logs
                   WHERE session_id = %s AND fingerprint = ANY(%s)
                   ORDER BY created_at ASC LIMIT %s""",
                (session_id, fps, anchor_limit),
            )
            recent_rows = db_query(
                """SELECT id, user_query, bot_answer, created_at FROM chat_logs
                   WHERE session_id = %s AND fingerprint = ANY(%s)
                   ORDER BY created_at DESC LIMIT %s""",
                (session_id, fps, recent_limit),
            )
            by_id = {row["id"]: row for row in anchor_rows + recent_rows}
            rows = sorted(by_id.values(), key=lambda r: r["created_at"])
        else:
            rows = db_query(
                """SELECT user_query, bot_answer, created_at FROM chat_logs
                   WHERE fingerprint = ANY(%s)
                   ORDER BY created_at DESC LIMIT %s""",
                (fps, limit),
            )
    else:
        rows = db_query(
            """SELECT user_query, bot_answer, created_at FROM chat_logs
               WHERE fingerprint = ANY(%s)
               ORDER BY created_at DESC LIMIT %s""",
            (fps, limit),
        )

    if not rows:
        return []
    if rows and "created_at" in rows[0]:
        rows = sorted(rows, key=lambda r: r["created_at"])
    return _rows_to_messages(rows)


# ── Logging ──────────────────────────────────────────────────────────

def _log_chat(
    session_id: str, fingerprint: str, user_agent: str, ip_address: str,
    user_query: str, bot_answer: str, route: str = "web_chat",
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
             user_query, bot_answer, route, 0, False, ""),
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
    profile = await resolve_runtime_profile("webchat")
    provider = profile.provider
    history_chars = sum(len(str(m.get("content", ""))) for m in history)
    logger.info(
        "Web chat profile session=%s provider=%s model=%s tier=%s history_messages=%d history_chars=%d budget=$%.2f",
        session_id, provider, profile.model_id, profile.tier,
        len(history), history_chars, profile.budget_usd,
    )

    # Fold the runtime header directly into the current user turn so the
    # history prefix stays byte-stable across requests (→ prompt caching).
    # Provider-native format: XML for Claude, Markdown for GPT.
    now = datetime.now(KST)
    runtime_context = _build_web_runtime_context(
        now.strftime("%Y-%m-%d %H:%M KST (%A)"),
        provider=provider,
    )
    history.append({"role": "user", "content": f"{runtime_context}\n\n{message}"})
    system_prompt = _build_web_system_prompt(provider)

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
            if provider in ("openai", "deepseek"):
                from openai_tool_loop import chat_with_tools as openai_chat
                client = _deepseek_client if provider == "deepseek" else _openai_client
                extra_kwargs = {}
                if provider == "deepseek":
                    extra_kwargs = {
                        "extra_body": {"thinking": {"type": "disabled"}},
                        "sdk_max_token_param": "max_tokens",
                    }
                result = await openai_chat(
                    history,
                    client=client,
                    model=profile.model_id,
                    tools=_web_tools,
                    tool_handlers=_web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=profile.max_rounds,
                    max_tokens=profile.max_tokens,
                    budget_usd=profile.budget_usd,
                    on_progress=on_progress,
                    provider_label=f"{provider}:web",
                    **extra_kwargs,
                )
            else:
                result = await chat_with_tools(
                    history,
                    client=_claude,
                    model=profile.model_id,
                    tools=_web_tools,
                    tool_handlers=_web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=profile.max_rounds,
                    max_tokens=profile.max_tokens,
                    budget_usd=profile.budget_usd,
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
            message, answer, f"{provider}_loop",
        )
        yield _format_sse({"type": "answer", "content": answer})
    else:
        yield _format_sse({"type": "error", "content": "응답을 생성하지 못했습니다."})
