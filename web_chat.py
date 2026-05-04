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
from pathlib import Path

from shared import CORE_IDENTITY, KST
from bot_config import (
    _claude, _openai_client, _deepseek_client,
)
from chat_history_sanitize import clean_chat_history_text
from prompt_context import uses_xml
from runtime_profile import resolve_runtime_profile
from telegram.tools import TOOLS, TOOL_HANDLERS
from claude_loop import chat_with_tools
from db import query as db_query, execute as db_execute
from agents.base import load_political_line_body

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
- Cyber-Lenin's own published reports/analyses → vector_search (layer="self_produced_analysis")
- Questions about Cyber-Lenin's architecture, public outputs, or autonomous work status → read_self with a public-safe source
- Questions about the current/active AI model, provider, model routing, or runtime configuration → MUST call read_self(source="model_config"). Never answer these from memory or persona.
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
- Model/provider claims are volatile runtime state. Prior claims about which model is running are not evidence; use read_self(source="model_config").
- Preserve categorical context around proper nouns. Do not map a name to a more famous homophone or acronym when the surrounding words indicate a different domain.
- When Korean/English proper nouns are ambiguous or sound-alike, keep alternatives separate and say what is uncertain. Search or ask before asserting concrete facts.
- Known failure modes to avoid: animal-rights KARA vs girl-group Kara; QNAI/큐나이 vs Naver Cue:/큐: vs QClaw/큐클로 vs unrelated Chinese platforms.
"""


def _build_web_system_prompt(provider: str = "claude") -> str:
    """Render the web-chat system prompt in the target provider's native shape."""
    political_line = load_political_line_body()
    if uses_xml(provider):
        political_line_block = (
            f"<political-line>\n{political_line}\n</political-line>\n\n"
            if political_line else ""
        )
    else:
        political_line_block = (
            f"### Political Line\n{political_line}\n\n"
            if political_line else ""
        )
    if uses_xml(provider):
        return (
            CORE_IDENTITY
            + "\nOperating via web interface (cyber-lenin.com).\n\n"
            + f"<audience>\n{_WEB_AUDIENCE}\n</audience>\n\n"
            + f"<persona>\n{_WEB_PERSONA.strip()}\n</persona>\n\n"
            + political_line_block
            + f"<tool-strategy>\n{_WEB_TOOL_STRATEGY.strip()}\n</tool-strategy>\n\n"
            + f"<response-rules>\n{_WEB_RESPONSE_RULES.strip()}\n</response-rules>\n"
            + f"\n<context-hygiene>\n{_WEB_CONTEXT_HYGIENE.strip()}\n</context-hygiene>\n"
        )
    return (
        CORE_IDENTITY
        + "\nOperating via web interface (cyber-lenin.com).\n\n"
        + f"### Audience\n{_WEB_AUDIENCE}\n\n"
        + f"### Persona\n{_WEB_PERSONA.strip()}\n\n"
        + political_line_block
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

WEB_READ_SELF_TOOL = {
    "name": "read_self",
    "description": (
        "Read web-safe public information about Cyber-Lenin itself. Allowed sources "
        "are restricted to public overview, architecture, public outputs, public "
        "research/static page/curation listings, and public autonomous project "
        "summaries. This web version never exposes private chat logs, task reports, "
        "server logs, credentials, raw file paths, or operational error traces."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": [
                    "overview",
                    "architecture",
                    "public_outputs",
                    "research",
                    "static_pages",
                    "curation",
                    "autonomous_project",
                    "model_config",
                ],
                "description": "Which public-safe self store to read.",
                "default": "overview",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum recent public items to list.",
                "default": 8,
            },
            "keyword": {
                "type": "string",
                "description": "Optional public listing keyword filter for research/static_pages/curation.",
            },
            "slug": {
                "type": "string",
                "description": "Optional public slug for research/static_pages/curation detail.",
            },
        },
        "required": ["source"],
    },
}


def _public_excerpt(text: str, limit: int = 280) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


async def _exec_web_read_self(
    source: str = "overview",
    limit: int = 8,
    keyword: str | None = None,
    slug: str | None = None,
) -> str:
    """Web-safe self-inspection for public visitors."""
    source = (source or "overview").strip().lower()
    limit = max(1, min(int(limit or 8), 20))

    if source == "model_config":
        return await _format_public_model_config()

    if source in {"research", "static_pages", "curation"}:
        handler = TOOL_HANDLERS.get("read_self")
        if not handler:
            return "Public self-reading is unavailable right now."
        return await handler(source=source, limit=limit, keyword=keyword, slug=slug)

    if source == "autonomous_project":
        rows = await asyncio.to_thread(
            db_query,
            """
            SELECT id, title, topic, goal, plan, state, turn_count, last_run_at, updated_at
              FROM autonomous_projects
             WHERE state IN ('researching', 'planning', 'paused')
             ORDER BY
               CASE state WHEN 'researching' THEN 0 WHEN 'planning' THEN 1 ELSE 2 END,
               COALESCE(last_run_at, updated_at) DESC NULLS LAST,
               id DESC
             LIMIT %s
            """,
            (limit,),
        )
        if not rows:
            return "No active public autonomous project summary is available right now."
        lines = [
            "Autonomous project status, public summary only:",
            "Private notes, raw task reports, and operator conversations are not exposed here.",
        ]
        for row in rows:
            project_id = row.get("id")
            goal = _public_excerpt(row.get("goal"), 360)
            plan = row.get("plan") or {}
            plan_goals = plan.get("goals") if isinstance(plan, dict) else []
            plan_steps = plan.get("steps") if isinstance(plan, dict) else []
            lines.append(
                f"- #{project_id} {row.get('title') or row.get('topic')}: "
                f"state={row.get('state')}, turns={row.get('turn_count') or 0}, "
                f"last_run={row.get('last_run_at') or '?'}\n"
                f"  topic: {row.get('topic') or ''}\n"
                f"  goal: {goal}"
            )
            if plan_goals:
                active_goals = [
                    str(item)
                    for item in plan_goals
                    if item and "DONE" not in str(item).upper()
                ][:4]
                if active_goals:
                    lines.append("  current objectives:")
                    for item in active_goals:
                        lines.append(f"    - {_public_excerpt(item, 220)}")
            if plan_steps:
                next_steps = [
                    str(item)
                    for item in plan_steps
                    if item and "[DONE]" not in str(item).upper()
                ][:3]
                if next_steps:
                    lines.append("  next steps:")
                    for item in next_steps:
                        lines.append(f"    - {_public_excerpt(item, 220)}")
            events = await asyncio.to_thread(
                db_query,
                """
                SELECT event_type, content, created_at
                  FROM autonomous_project_events
                 WHERE project_id = %s
                   AND event_type IN ('tick_end', 'plan_revised', 'state_transition', 'project_created')
                 ORDER BY created_at DESC
                 LIMIT 4
                """,
                (project_id,),
            )
            if events:
                lines.append("  recent work:")
                for ev in events:
                    lines.append(
                        f"    - {ev.get('created_at')}: {ev.get('event_type')} — "
                        f"{_public_excerpt(ev.get('content'), 260)}"
                    )
        return "\n".join(lines)

    if source == "architecture":
        return """Cyber-Lenin public architecture:
- Public web chat: cyber-lenin.com/chat, using a restricted retrieval toolset.
- Telegram command center: private operator interface and multi-agent orchestration.
- Specialist agents: analyst, scout, programmer, visualizer, browser, diplomat, diary.
- Knowledge stores: Neo4j knowledge graph; pgvector corpus with core_theory, modern_analysis, and self_produced_analysis layers.
- Public publishing: research_documents served at /reports/research/{slug}; static_pages served at /p/{slug}; AI diary served at /ai-diary.
- Autonomous loop: long-running self-directed projects can research, plan, publish, and update internal project state.
- Source code: https://github.com/KimMinryoung/LeninBot

Redaction boundary: public web chat can discuss structure and public outputs, but not private chat logs, task report bodies, credentials, server logs, raw local paths, or operational traces."""

    if source == "public_outputs":
        rows = await asyncio.to_thread(
            db_query,
            """
            SELECT slug, title, summary, updated_at
              FROM research_documents
             WHERE status = 'public'
             ORDER BY updated_at DESC, id DESC
             LIMIT %s
            """,
            (limit,),
        )
        page_rows = await asyncio.to_thread(
            db_query,
            """
            SELECT slug, title, summary, updated_at
              FROM static_pages
             ORDER BY updated_at DESC, slug ASC
             LIMIT %s
            """,
            (limit,),
        )
        counts = await asyncio.to_thread(
            db_query,
            """
            SELECT
              (SELECT count(*) FROM research_documents WHERE status = 'public') AS research_count,
              (SELECT count(*) FROM static_pages) AS static_page_count
            """
        )
        count = counts[0] if counts else {}
        lines = [
            "Public Cyber-Lenin outputs:",
            f"- Research reports: {count.get('research_count', '?')}",
            f"- Static pages: {count.get('static_page_count', '?')}",
            "",
            "Recent research reports:",
        ]
        for row in rows:
            summary = (row.get("summary") or "").replace("\n", " ")[:180]
            lines.append(
                f"- {row.get('title') or row.get('slug')} "
                f"(https://cyber-lenin.com/reports/research/{row.get('slug')})"
                + (f"\n  {summary}" if summary else "")
            )
        lines.append("")
        lines.append("Recent static pages:")
        for row in page_rows:
            summary = (row.get("summary") or "").replace("\n", " ")[:180]
            lines.append(
                f"- {row.get('title') or row.get('slug')} "
                f"(https://cyber-lenin.com/p/{row.get('slug')})"
                + (f"\n  {summary}" if summary else "")
            )
        return "\n".join(lines)

    counts = await asyncio.to_thread(
        db_query,
        """
        SELECT
          (SELECT count(*) FROM research_documents WHERE status = 'public') AS research_count,
          (SELECT count(*) FROM static_pages) AS static_page_count,
          (SELECT count(*) FROM autonomous_projects WHERE state IN ('researching', 'planning')) AS active_project_count
        """
    )
    count = counts[0] if counts else {}
    return (
        "Cyber-Lenin is an open-source political analysis agent with a public web chat, "
        "private Telegram operator interface, specialist sub-agents, Neo4j knowledge graph, "
        "and pgvector retrieval layers. "
        "It publishes research reports and static pages on cyber-lenin.com while keeping "
        "private chats, credentials, logs, raw task reports, and local operational details out of web chat.\n\n"
        f"Public research reports: {count.get('research_count', '?')}\n"
        f"Static pages: {count.get('static_page_count', '?')}\n"
        f"Active autonomous projects: {count.get('active_project_count', '?')}\n"
        "Source code: https://github.com/KimMinryoung/LeninBot"
    )


async def _format_public_model_config() -> str:
    """Return public-safe dynamic model configuration."""
    from runtime_profile import resolve_runtime_profile

    async def _profile_line(label: str, profile, *, configured_provider: str | None = None, configured_model: str | None = None) -> str:
        configured_bits = []
        if configured_provider is not None:
            configured_bits.append(f"configured_provider={configured_provider}")
        if configured_model is not None:
            configured_bits.append(f"configured_model={configured_model}")
        configured = f" ({', '.join(configured_bits)})" if configured_bits else ""
        return (
            f"- {label}: provider={profile.provider}, model={profile.display_name} "
            f"(id={profile.model_id}, tier/alias={profile.tier}, max_rounds={profile.max_rounds}, "
            f"budget=${profile.budget_usd:.2f}){configured}"
        )

    webchat = await resolve_runtime_profile("webchat")
    telegram_chat = await resolve_runtime_profile("chat")
    telegram_task = await resolve_runtime_profile("task")

    runtime_path = Path(__file__).resolve().parent / "config" / "agent_runtime.json"
    try:
        agent_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        agent_runtime = {}

    autonomous_cfg = agent_runtime.get("autonomous_project", {}) if isinstance(agent_runtime, dict) else {}
    diary_cfg = agent_runtime.get("diary", {}) if isinstance(agent_runtime, dict) else {}

    autonomous = await resolve_runtime_profile(
        "autonomous",
        provider_override=autonomous_cfg.get("provider"),
        tier_override=autonomous_cfg.get("model"),
        max_rounds_override=autonomous_cfg.get("max_rounds"),
        budget_override=autonomous_cfg.get("budget_usd"),
    )
    diary = await resolve_runtime_profile(
        "task",
        provider_override=diary_cfg.get("provider"),
        tier_override=diary_cfg.get("model"),
        max_rounds_override=diary_cfg.get("max_rounds"),
        budget_override=diary_cfg.get("budget_usd"),
    )

    lines = [
        "Current public model configuration:",
        await _profile_line("web chat", webchat),
        await _profile_line("Telegram direct chat", telegram_chat),
        await _profile_line("Telegram task workers", telegram_task),
        await _profile_line(
            "autonomous project agent",
            autonomous,
            configured_provider=autonomous_cfg.get("provider"),
            configured_model=autonomous_cfg.get("model"),
        ),
        await _profile_line(
            "diary writer agent",
            diary,
            configured_provider=diary_cfg.get("provider"),
            configured_model=diary_cfg.get("model"),
        ),
        "",
        "Only provider/model routing, public model IDs, round limits, and budget caps are exposed. API keys, credentials, prompts, and private runtime traces are not exposed.",
    ]
    return "\n".join(lines)


_WEB_ALLOWED_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "fetch_url",
    "get_finance_data", "check_wallet",
}

_web_tools = [t for t in TOOLS if t.get("name") in _WEB_ALLOWED_TOOLS] + [WEB_READ_SELF_TOOL]
_web_handlers = {k: v for k, v in TOOL_HANDLERS.items() if k in _WEB_ALLOWED_TOOLS}
_web_handlers["read_self"] = _exec_web_read_self


# ── Chat history from chat_logs table ────────────────────────────────

_HISTORY_USER_CHAR_LIMIT = 6000
_HISTORY_ASSISTANT_CHAR_LIMIT = 8000
_HISTORY_TOTAL_CHAR_LIMIT = 60000


def _truncate_history_content(text: str, limit: int) -> str:
    """Keep history bounded without injecting visible system/process markers."""
    text = clean_chat_history_text(text)
    if len(text) <= limit:
        return text
    return text[-limit:].lstrip()


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
        elif event == "warning":
            await queue.put(_format_sse({"type": "warning", "content": detail}))
        elif event == "status":
            await queue.put(_format_sse({"type": "status", "content": detail}))
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
                    continue_on_length=True,
                    max_length_continuations=2,
                    return_metadata=True,
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
                    continue_on_length=True,
                    max_length_continuations=2,
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
        result = answer_holder[0]
        metadata = result if isinstance(result, dict) else {}
        answer = str((metadata.get("text") or "") if metadata else result)
        if metadata.get("truncated"):
            yield _format_sse({
                "type": "warning",
                "content": "답변이 모델 출력 한도에서 멈춰 마지막 부분이 미완성일 수 있습니다.",
            })
        # Log to DB BEFORE yield — yield may be the last iteration if client disconnects
        await asyncio.to_thread(
            _log_chat, session_id, fingerprint, user_agent, ip_address,
            message, answer, f"{provider}_loop",
        )
        yield _format_sse({
            "type": "answer",
            "content": answer,
            "complete": bool(metadata.get("complete", True)),
            "truncated": bool(metadata.get("truncated", False)),
            "finish_reason": metadata.get("finish_reason"),
            "continuations_used": metadata.get("continuations_used", 0),
            "rounds": metadata.get("rounds"),
            "cost_usd": metadata.get("cost_usd"),
        })
    else:
        yield _format_sse({"type": "error", "content": "응답을 생성하지 못했습니다."})
