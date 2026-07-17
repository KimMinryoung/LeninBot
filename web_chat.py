"""web_chat.py — Web chat handler using claude_loop (replaces LangGraph chatbot.py).

Bridges api.py to the unified agent system. Handles:
- Web-specific system prompt (allows markdown, no delegation/mission)
- Chat history from chat_logs table (fingerprint-based)
- SSE streaming via on_progress callback → asyncio.Queue
- Logging to chat_logs table
"""

import asyncio
from contextlib import suppress
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

from shared import KST
from bot_config import (
    _claude, _openai_client, _deepseek_anthropic_client, _kimi_client,
)
from chat_history_sanitize import clean_chat_history_text
from prompt_context import uses_xml
from runtime_profile import resolve_runtime_profile
from runtime_tools.registry import TOOLS, TOOL_HANDLERS
from tool_gateway.selection import build_toolset
from claude_loop import chat_with_tools
from db import query as db_query, query_one as db_query_one, execute as db_execute
from web_personas import (
    DEFAULT_PERSONA_ID,
    CYBER_LENIN_TOOLS,
    get_persona,
    render_system_prompt,
)

logger = logging.getLogger(__name__)

# ── Web-specific system prompt ───────────────────────────────────────
# Persona definitions live in web_personas.py. The web-chat system prompt is
# now rendered per-persona; this thin wrapper preserves the default (Cyber-Lenin)
# rendering for callers/tests that reference it by name.


def _build_web_system_prompt(provider: str = "claude") -> str:
    """Render the default (Cyber-Lenin) web-chat system prompt."""
    return render_system_prompt(get_persona(DEFAULT_PERSONA_ID), provider)


def _build_web_runtime_context(current_datetime: str, provider: str = "claude") -> str:
    """Render web-chat runtime header in the provider-native structure."""
    if uses_xml(provider):
        return f"<runtime>\n<current-time>{current_datetime}</current-time>\n</runtime>"
    return f"### Runtime\n- **Current Time**: {current_datetime}"


def _build_web_model_context(profile, provider: str = "claude") -> str:
    """Render the authoritative current web-chat model display name."""
    if uses_xml(provider):
        return f"<current-model>{profile.display_name}</current-model>"
    return f"### Current Model\n- **Display Name**: {profile.display_name}"


# ── Tool filtering: web chat gets only information-retrieval tools ────

PERSONA_CONTEXT_ROOT = Path(__file__).resolve().parent / "identity" / "web_personas"


WEB_READ_SELF_TOOL = {
    "name": "read_self",
    "description": (
        "Read web-safe public information about Cyber-Lenin itself. This is a "
        "public-safe subset of the internal read_self content_type interface, plus "
        "a few web-only public summaries. It can read public overview/model/config "
        "summaries, public diary entries, public research documents, public static "
        "pages, public blog posts, hub curations, and public autonomous project "
        "summaries. It never exposes private chat logs, task reports, private "
        "research documents, server logs, credentials, raw file paths, or "
        "operational error traces."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content_type": {
                "type": "string",
                "enum": [
                    "overview",
                    "architecture",
                    "public_outputs",
                    "diary",
                    "research_document",
                    "static_page",
                    "blog_post",
                    "hub_curation",
                    "autonomous_project",
                    "model_config",
                ],
                "description": "Which public-safe content/runtime type to read.",
                "default": "overview",
            },
            "source": {
                "type": "string",
                "description": "Deprecated compatibility alias for content_type; do not use in new calls.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum recent public items to list.",
                "default": 8,
            },
            "keyword": {
                "type": "string",
                "description": "Optional public listing keyword filter for public content lists.",
            },
            "id": {
                "type": "integer",
                "description": "Optional numeric id for diary, blog_post, or autonomous_project detail.",
            },
            "post_id": {
                "type": "integer",
                "description": "Deprecated alias for id on public diary/blog_post reads.",
            },
            "slug": {
                "type": "string",
                "description": (
                    "Optional public slug for detail reads. Use content_type='static_page' for "
                    "cyber-lenin.com/p/{slug}; content_type='research_document' for "
                    "/reports/research/{slug}; and content_type='hub_curation' for /hub/{slug}."
                ),
            },
            "max_chars": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": "Maximum body characters for long public detail reads.",
            },
            "offset": {
                "type": "integer",
                "description": "Character offset for paginating long public detail reads.",
            },
        },
        "required": ["content_type"],
    },
}


WEB_PERSONA_CONTEXT_TOOL = {
    "name": "read_persona_context",
    "description": (
        "Read this selected persona's private dossier. The server binds the "
        "lookup to the active persona, so one persona cannot read another "
        "persona's notes. Use it for that persona's concepts, biography, "
        "strategic vocabulary, and prepared background details."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "search"],
                "description": "List available topics, read one topic, or search the persona dossier.",
                "default": "list",
            },
            "topic": {
                "type": "string",
                "description": "Topic slug returned by action='list', for example concepts or biography.",
            },
            "query": {
                "type": "string",
                "description": "Search query for action='search'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum search results or list items.",
                "default": 5,
            },
            "max_chars": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": "Maximum body characters for long topic reads.",
            },
            "offset": {
                "type": "integer",
                "description": "Character offset for paginating long topic reads.",
                "default": 0,
            },
        },
        "required": ["action"],
    },
}


def _public_excerpt(text: str, limit: int = 280) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


_PERSONA_CONTEXT_TOPIC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}(?:\.md)?$")


def _persona_context_path(spec) -> Path | None:
    context_dir = str(getattr(spec, "context_dir", "") or "").strip()
    if not context_dir or not _PERSONA_CONTEXT_TOPIC_RE.match(context_dir):
        return None
    base = PERSONA_CONTEXT_ROOT.resolve()
    path = (base / context_dir / "knowledge").resolve()
    try:
        path.relative_to(base)
    except ValueError:
        logger.warning("Rejected persona context path outside base persona=%s", getattr(spec, "id", "?"))
        return None
    if not path.is_dir():
        return None
    return path


def _persona_context_files(spec) -> list[Path]:
    path = _persona_context_path(spec)
    if not path:
        return []
    return sorted(p for p in path.glob("*.md") if p.is_file())


def _persona_context_title(path: Path, body: str | None = None) -> str:
    if body is None:
        try:
            body = path.read_text(encoding="utf-8")
        except Exception:
            body = ""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _persona_topic_file(spec, topic: str | None) -> Path | None:
    raw = str(topic or "").strip()
    if not raw or "/" in raw or "\\" in raw:
        return None
    if not _PERSONA_CONTEXT_TOPIC_RE.match(raw):
        return None
    if raw.endswith(".md"):
        raw = raw[:-3]
    path = _persona_context_path(spec)
    if not path:
        return None
    candidate = (path / f"{raw}.md").resolve()
    try:
        candidate.relative_to(path.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


async def _exec_persona_context_for(
    spec,
    action: str = "list",
    topic: str | None = None,
    query: str | None = None,
    limit: int = 5,
    max_chars: int | None = None,
    offset: int = 0,
) -> str:
    """Read the active persona's own dossier; never crosses persona dirs."""
    action = str(action or "list").strip().lower()
    if action not in {"list", "read", "search"}:
        action = "list"
    files = _persona_context_files(spec)
    persona_name = getattr(spec, "display_name", getattr(spec, "id", "this persona"))
    if not files:
        return f"No persona dossier is configured for {persona_name}."

    try:
        limit = int(limit or 5)
    except Exception:
        limit = 5
    limit = max(1, min(limit, 20))
    if action == "list":
        lines = [f"Available persona dossier topics for {persona_name}:"]
        for file_path in files[:limit]:
            body = file_path.read_text(encoding="utf-8")
            excerpt = _public_excerpt(body.replace("#", ""), 180)
            lines.append(f"- {file_path.stem}: {_persona_context_title(file_path, body)}" + (f" -- {excerpt}" if excerpt else ""))
        return "\n".join(lines)

    if action == "read":
        file_path = _persona_topic_file(spec, topic)
        if not file_path:
            return "Topic not found in this persona dossier. Call action='list' for available topic slugs."
        body = file_path.read_text(encoding="utf-8")
        try:
            offset = int(offset or 0)
        except Exception:
            offset = 0
        try:
            max_chars = int(max_chars or 6000)
        except Exception:
            max_chars = 6000
        offset = max(0, offset)
        max_chars = max(500, min(max_chars, 12000))
        chunk = body[offset: offset + max_chars]
        next_offset = offset + len(chunk)
        suffix = f"\n\n[next_offset={next_offset}]" if next_offset < len(body) else ""
        return f"Persona dossier topic: {file_path.stem} ({_persona_context_title(file_path, body)})\n\n{chunk}{suffix}"

    needle = str(query or "").strip().lower()
    if not needle:
        return "Search query is required for action='search'."
    results: list[str] = []
    for file_path in files:
        body = file_path.read_text(encoding="utf-8")
        lower = body.lower()
        idx = lower.find(needle)
        if idx < 0:
            continue
        start = max(0, idx - 260)
        end = min(len(body), idx + len(needle) + 520)
        excerpt = body[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(body):
            excerpt += "..."
        results.append(f"- {file_path.stem}: {_persona_context_title(file_path, body)}\n  {excerpt}")
        if len(results) >= limit:
            break
    if not results:
        return f"No matches for {query!r} in {persona_name}'s persona dossier."
    return f"Persona dossier search results for {persona_name}:\n" + "\n".join(results)


_GRAMSCI_PREFLIGHT_TRIGGERS = {
    "gramsci", "그람시", "hegemony", "헤게모니", "civil society", "시민사회",
    "integral state", "통일적 국가", "political society", "정치사회",
    "war of position", "진지전", "war of manoeuvre", "war of maneuver", "기동전",
    "organic intellectual", "유기적 지식인", "traditional intellectual", "전통적 지식인",
    "common sense", "상식", "good sense", "양식", "modern prince", "현대 군주",
    "passive revolution", "수동혁명", "caesarism", "카이사르주의",
    "prison notebooks", "옥중수고", "fordism", "포드주의",
}

_GRAMSCI_QUERY_TERMS = (
    ("hegemony", ("hegemony", "헤게모니")),
    ("civil society political society integral state", ("civil society", "시민사회", "political society", "정치사회", "integral state", "통일적 국가")),
    ("war of position war of manoeuvre", ("war of position", "진지전", "war of manoeuvre", "war of maneuver", "기동전")),
    ("organic intellectuals traditional intellectuals", ("organic intellectual", "유기적 지식인", "traditional intellectual", "전통적 지식인")),
    ("common sense good sense", ("common sense", "상식", "good sense", "양식")),
    ("modern prince party collective will", ("modern prince", "현대 군주", "party", "정당", "collective will", "집단 의지")),
    ("passive revolution caesarism transformism", ("passive revolution", "수동혁명", "caesarism", "카이사르주의", "transformism")),
    ("prison notebooks philosophy of praxis", ("prison notebooks", "옥중수고", "philosophy of praxis", "실천철학")),
    ("fordism americanism", ("fordism", "포드주의", "americanism", "미국주의")),
)


def _build_gramsci_preflight_query(message: str) -> str:
    raw = str(message or "")
    lowered = raw.lower()
    if not any(trigger in lowered or trigger in raw for trigger in _GRAMSCI_PREFLIGHT_TRIGGERS):
        return ""
    terms: list[str] = []
    for english_term, needles in _GRAMSCI_QUERY_TERMS:
        if any(needle in lowered or needle in raw for needle in needles):
            terms.append(english_term)
    if not terms:
        terms.append("hegemony civil society war of position organic intellectuals")
    return " ".join(dict.fromkeys(terms))


async def _build_gramsci_preflight_context(message: str, provider: str = "claude") -> str:
    query = _build_gramsci_preflight_query(message)
    if not query:
        return ""
    handler = TOOL_HANDLERS.get("vector_search")
    if not handler:
        return ""
    try:
        result = await handler(
            query=query,
            layer="core_theory",
            author="Gramsci",
            num_results=3,
        )
    except Exception as exc:
        logger.warning("Gramsci preflight vector_search failed: %s", exc)
        return ""
    result = str(result or "").strip()
    if not result or result == "No documents found." or result.startswith("Vector search failed"):
        return ""
    if len(result) > 10000:
        result = result[:10000].rstrip() + "\n[preflight truncated]"
    logger.info("Gramsci preflight vector_search query=%s chars=%d", query, len(result))
    body = (
        "Server-side preflight retrieval from the Gramsci primary-text corpus. "
        "Use this as grounding for Gramsci textual/conceptual claims; do not mention the preflight mechanism.\n"
        f"Query: vector_search(query={query!r}, layer=\"core_theory\", author=\"Gramsci\", num_results=3)\n\n"
        f"{result}"
    )
    if uses_xml(provider):
        return f"<preflight-retrieval persona=\"gramsci\">\n{body}\n</preflight-retrieval>"
    return f"### Preflight Retrieval: Gramsci Corpus\n{body}"


async def _exec_web_read_self(
    content_type: str | None = None,
    source: str | None = None,
    limit: int = 8,
    keyword: str | None = None,
    id: int | None = None,
    post_id: int | None = None,
    slug: str | None = None,
    max_chars: int | None = None,
    offset: int | None = None,
) -> str:
    """Web-safe self-inspection for public visitors."""
    raw_type = (content_type or source or "overview").strip().lower()
    compat_aliases = {
        "research": "research_document",
        "research_documents": "research_document",
        "static_pages": "static_page",
        "curation": "hub_curation",
        "curations": "hub_curation",
        "post": "blog_post",
        "posts": "blog_post",
    }
    content_type = compat_aliases.get(raw_type, raw_type)
    if id is not None and post_id is None:
        post_id = id
    limit = max(1, min(int(limit or 8), 20))

    if content_type == "model_config":
        return await _format_public_model_config()

    if content_type in {"diary", "research_document", "static_page", "blog_post", "hub_curation"}:
        handler = TOOL_HANDLERS.get("read_self")
        if not handler:
            return "Public self-reading is unavailable right now."
        return await handler(
            content_type=content_type,
            limit=limit,
            keyword=keyword,
            id=id,
            post_id=post_id,
            slug=slug,
            status="public" if content_type == "research_document" else None,
            max_chars=max_chars,
            offset=offset,
        )

    if content_type == "autonomous_project":
        try:
            from bot_config import is_autonomous_active
            loop_active = await asyncio.to_thread(is_autonomous_active)
        except Exception:
            loop_active = True
        project_filter = "AND id = %s" if id is not None else ""
        params = (int(id), limit) if id is not None else (limit,)
        rows = await asyncio.to_thread(
            db_query,
            f"""
            SELECT id, title, topic, goal, plan, state, turn_count, last_run_at, updated_at
              FROM autonomous_projects
             WHERE state IN ('researching', 'planning', 'paused')
               {project_filter}
             ORDER BY
               CASE state WHEN 'researching' THEN 0 WHEN 'planning' THEN 1 ELSE 2 END,
               COALESCE(last_run_at, updated_at) DESC NULLS LAST,
               id DESC
             LIMIT %s
            """,
            params,
        )
        if not rows:
            if id is not None:
                return f"No active public autonomous project summary is available for id={id}."
            return "No active public autonomous project summary is available right now."
        loop_text = (
            "enabled; scheduled ticks can advance due projects"
            if loop_active
            else "paused by config; scheduled timer wakes skip project execution"
        )
        lines = [
            "Autonomous project status, public summary only:",
            f"Autonomous loop: {loop_text}.",
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
                   AND event_type IN (
                       'tick_end', 'plan_revised', 'state_transition',
                       'project_created', 'publication_created'
                   )
                 ORDER BY created_at DESC, id DESC
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

    if content_type == "architecture":
        return """Cyber-Lenin public architecture:
- Public web chat: cyber-lenin.com/chat, using a restricted retrieval toolset.
- Telegram command center: private operator interface and multi-agent orchestration.
- Specialist agents: analyst, scout, programmer, visualizer, browser, diplomat, diary.
- Knowledge stores: Neo4j knowledge graph; pgvector corpus with core_theory, modern_analysis, and self_produced_analysis layers.
- Public publishing: research_documents served at /reports/research/{slug}; static_pages served at /p/{slug}; AI diary served at /ai-diary.
- Autonomous loop: long-running self-directed projects can research, plan, publish, and update internal project state.
- Source code: https://github.com/KimMinryoung/LeninBot

Redaction boundary: public web chat can discuss structure and public outputs, but not private chat logs, task report bodies, credentials, server logs, raw local paths, or operational traces."""

    if content_type == "public_outputs":
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

    try:
        from bot_config import is_autonomous_active
        loop_active = await asyncio.to_thread(is_autonomous_active)
    except Exception:
        loop_active = True
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
        f"Autonomous loop: {'enabled' if loop_active else 'paused by config'}\n"
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


def _build_persona_tools(persona_or_allowed_tools) -> tuple[list[dict], dict]:
    """Build the (tools, handlers) pair for a persona's allowed-tool set.

    `read_self` is special: the shared TOOLS registry also defines a read_self,
    but the public web must only ever see the public-safe WEB_READ_SELF_TOOL /
    _exec_web_read_self pair. So the registry read_self is always excluded here
    and the web-safe one is injected when the persona allows read_self.
    """
    spec = persona_or_allowed_tools if hasattr(persona_or_allowed_tools, "allowed_tools") else None
    allowed_tools = spec.allowed_tools if spec is not None else persona_or_allowed_tools
    web_only_tools = {"read_self", "read_persona_context"}
    registry_allowed = set(allowed_tools) - web_only_tools
    tools, handlers = build_toolset(TOOLS, TOOL_HANDLERS, registry_allowed)
    if "read_self" in allowed_tools:
        tools = tools + [WEB_READ_SELF_TOOL]
        handlers = {**handlers, "read_self": _exec_web_read_self}
    if "read_persona_context" in allowed_tools and spec is not None and spec.context_dir:
        async def _bound_persona_context(**kwargs):
            return await _exec_persona_context_for(spec, **kwargs)

        tools = tools + [WEB_PERSONA_CONTEXT_TOOL]
        handlers = {**handlers, "read_persona_context": _bound_persona_context}
    return tools, handlers


# Backward-compatible default (Cyber-Lenin) tool set. `_WEB_ALLOWED_TOOLS` holds
# only registry-backed tools (read_self lives outside TOOLS); per-persona tool
# resolution happens in handle_web_chat via _build_persona_tools.
_WEB_ALLOWED_TOOLS = set(CYBER_LENIN_TOOLS) - {"read_self"}
_web_tools, _web_handlers = _build_persona_tools(get_persona(DEFAULT_PERSONA_ID))


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


_FEEDBACK_TONE_LABELS = {
    "shorter": "shorter and less digressive",
    "longer": "more developed and detailed",
    "warmer": "warmer and more emotionally responsive",
    "colder": "colder, sharper, and more severe",
    "more_direct": "more direct and less hedged",
    "more_in_character": "more strongly in character",
    "less_formal": "less formal and more conversational",
    "more_cited": "more grounded with citations or concrete references when factual",
}


def normalize_web_chat_tone_feedback(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in _FEEDBACK_TONE_LABELS else ""


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


def _load_web_feedback_rows(
    fingerprints: list[str],
    session_id: str | None,
    persona: str,
    limit: int = 8,
    account_user_id: int | None = None,
) -> list[dict]:
    fps = [f for f in (fingerprints or []) if f]
    if not account_user_id and not fps:
        return []
    session_clause = "AND (f.session_id = %s OR f.session_id IS NULL)" if session_id else ""
    identity_clause = "l.user_id = %s" if account_user_id else "f.fingerprint = ANY(%s)"
    params: list = [account_user_id or fps, persona]
    if session_id:
        params.append(session_id)
    params.append(limit)
    return db_query(
        f"""SELECT f.id, f.rating, f.tone_feedback, f.note,
                  CASE WHEN l.user_query_active THEN l.user_query ELSE '[지워진 턴]' END AS user_query,
                  CASE WHEN l.bot_answer_active THEN l.bot_answer ELSE '[지워진 턴]' END AS bot_answer,
                  f.updated_at
              FROM web_chat_feedback f
              JOIN chat_logs l ON l.id = f.chat_log_id
             WHERE {identity_clause}
               AND f.persona = %s
               AND f.consumed_at IS NULL
               AND f.note IS NOT NULL
               AND btrim(f.note) <> ''
               {session_clause}
             ORDER BY f.updated_at DESC
             LIMIT %s""",
        params,
    )


def _mark_web_feedback_consumed(feedback_ids: list[int]) -> None:
    ids = [int(x) for x in (feedback_ids or []) if x]
    if not ids:
        return
    db_execute(
        """UPDATE web_chat_feedback
              SET consumed_at = COALESCE(consumed_at, now()),
                  updated_at = now()
            WHERE id = ANY(%s)""",
        (ids,),
    )


def _load_web_tone_policy(
    fingerprints: list[str],
    session_id: str | None,
    persona: str,
    limit: int = 40,
    account_user_id: int | None = None,
) -> list[dict]:
    fps = [f for f in (fingerprints or []) if f]
    if not account_user_id and not fps:
        return []
    session_clause = "AND (f.session_id = %s OR f.session_id IS NULL)" if session_id else ""
    identity_clause = "l.user_id = %s" if account_user_id else "f.fingerprint = ANY(%s)"
    params: list = [account_user_id or fps, persona]
    if session_id:
        params.append(session_id)
    params.append(max(1, min(int(limit or 40), 100)))
    rows = db_query(
        f"""WITH recent AS (
	               SELECT f.tone_feedback
	                 FROM web_chat_feedback f
	                 JOIN chat_logs l ON l.id = f.chat_log_id
	                WHERE {identity_clause}
	                  AND f.persona = %s
	                  AND f.tone_feedback IS NOT NULL
	                  AND btrim(f.tone_feedback) <> ''
	                  {session_clause}
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


def _render_web_tone_policy(rows: list[dict], provider: str = "claude") -> str:
    if not rows:
        return ""
    lines = [
        "Ongoing response policy inferred from the visitor's dropdown feedback. Apply as standing style policy for this persona/session; do not treat it as factual evidence and do not mention feedback history.",
    ]
    for row in rows[:3]:
        tone = normalize_web_chat_tone_feedback(row.get("tone_feedback"))
        if not tone:
            continue
        try:
            count = int(row.get("count") or 0)
        except Exception:
            count = 0
        suffix = f" (selected {count} times recently)" if count > 1 else ""
        lines.append(f"- {_FEEDBACK_TONE_LABELS[tone]}{suffix}")
    if len(lines) == 1:
        return ""
    body = "\n".join(lines)
    if uses_xml(provider):
        return f"<response-policy>\n{body}\n</response-policy>"
    return f"### Response Policy\n{body}"


def _render_web_feedback_context(rows: list[dict], provider: str = "claude") -> str:
    if not rows:
        return ""
    lines = [
        "The visitor has given manual written feedback for this next answer only. Apply it once as local style guidance, not factual evidence; do not carry it into later turns after this answer.",
    ]
    for row in rows[:8]:
        note = clean_chat_history_text(str(row.get("note") or "")).strip()[:220]
        if note:
            lines.append(f"- note={note}")
    body = "\n".join(lines)
    if uses_xml(provider):
        return f"<response-feedback>\n{body}\n</response-feedback>"
    return f"### Response Feedback\n{body}"


def _build_regeneration_message(row: dict, tone_feedback: str = "", note: str = "") -> str:
    tone_feedback = normalize_web_chat_tone_feedback(tone_feedback)
    feedback_bits: list[str] = []
    if tone_feedback:
        feedback_bits.append(_FEEDBACK_TONE_LABELS[tone_feedback])
    note = clean_chat_history_text(str(note or "")).strip()[:500]
    if note:
        feedback_bits.append(note)
    feedback = "; ".join(feedback_bits) or "Give a better alternative response while preserving the persona."
    user_query = clean_chat_history_text(str(row.get("user_query") or ""))
    previous_answer = clean_chat_history_text(str(row.get("bot_answer") or ""))[:2000]
    return (
        "Regenerate the previous answer for this same user request. "
        "Do not mention that this is a regeneration unless the character would naturally do so. "
        "Apply this feedback: " + feedback + "\n\n"
        "Original user request:\n" + user_query + "\n\n"
        "Previous answer to improve:\n" + previous_answer
    )


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
    deleted_turn_marker = "[지워진 턴]"

    def _rows_to_messages(rows: list[dict]) -> list[dict]:
        if excluded_ids:
            rows = [row for row in rows if int(row.get("id") or 0) not in excluded_ids]
        messages = []
        for row in rows:
            if row.get("user_query"):
                messages.append({
                    "role": "user",
                    "content": (
                        _truncate_history_content(row["user_query"], _HISTORY_USER_CHAR_LIMIT)
                        if row.get("user_query_active", True)
                        else deleted_turn_marker
                    ),
                })
            if row.get("bot_answer"):
                messages.append({
                    "role": "assistant",
                    "content": (
                        _truncate_history_content(row["bot_answer"], _HISTORY_ASSISTANT_CHAR_LIMIT)
                        if row.get("bot_answer_active", True)
                        else deleted_turn_marker
                    ),
                })
        return _fit_history_budget(messages)

    if session_id:
        # Does this session actually belong to one of the provided fingerprints,
        # under this persona?
        owned = db_query(
            f"""SELECT 1 FROM chat_logs
               WHERE session_id = %s AND {identity_clause} AND persona = %s
               LIMIT 1""",
            (session_id, identity_value, persona),
        )
        if owned:
            anchor_limit = min(4, max(0, limit // 4))
            recent_limit = max(0, limit - anchor_limit)
            anchor_rows = db_query(
                f"""SELECT id, user_query, bot_answer,
                          user_query_active, bot_answer_active, created_at FROM chat_logs
                   WHERE session_id = %s AND {identity_clause} AND persona = %s
                   ORDER BY created_at ASC LIMIT %s""",
                (session_id, identity_value, persona, anchor_limit),
            )
            recent_rows = db_query(
                f"""SELECT id, user_query, bot_answer,
                          user_query_active, bot_answer_active, created_at FROM chat_logs
                   WHERE session_id = %s AND {identity_clause} AND persona = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (session_id, identity_value, persona, recent_limit),
            )
            by_id = {row["id"]: row for row in anchor_rows + recent_rows}
            rows = sorted(by_id.values(), key=lambda r: r["created_at"])
        else:
            # Brand-new session: start clean. This used to fall back to the
            # fingerprint's recent turns across ALL sessions, which injected a
            # previous conversation's context into every fresh session.
            return []
    else:
        rows = db_query(
            f"""SELECT id, user_query, bot_answer,
                          user_query_active, bot_answer_active, created_at FROM chat_logs
               WHERE {identity_clause} AND persona = %s
               ORDER BY created_at DESC LIMIT %s""",
            (identity_value, persona, limit),
        )

    if not rows:
        return []
    if rows and "created_at" in rows[0]:
        rows = sorted(rows, key=lambda r: r["created_at"])
    return _rows_to_messages(rows)


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


_SOURCE_TOOL_NAMES = {
    "knowledge_graph_search",
    "vector_search",
    "web_search",
    "fetch_url",
}

_TOOL_DETAIL_RE = re.compile(r"\]\s*([A-Za-z_][A-Za-z0-9_]*)\(")


def _summarize_tool_usage(tool_work_details: list[str]) -> tuple[int, bool, str]:
    """Convert low-level tool work records into chat_logs display fields."""
    counts: dict[str, int] = {}
    source_count = 0
    for detail in tool_work_details or []:
        match = _TOOL_DETAIL_RE.search(str(detail))
        if not match:
            continue
        name = match.group(1)
        counts[name] = counts.get(name, 0) + 1
        if name in _SOURCE_TOOL_NAMES:
            source_count += 1

    web_search_used = counts.get("web_search", 0) > 0
    if not counts:
        return 0, False, ""
    strategy = "tools: " + ", ".join(
        f"{name} x{count}" for name, count in sorted(counts.items())
    )
    return source_count, web_search_used, strategy[:1000]


def _log_chat(
    session_id: str, fingerprint: str, user_agent: str, ip_address: str,
    user_query: str, bot_answer: str, route: str = "web_chat",
    documents_count: int = 0, web_search_used: bool = False, strategy: str = "",
    persona: str = DEFAULT_PERSONA_ID, authenticated_user_id: int | None = None,
) -> int | None:
    """Save web chat exchange to chat_logs table and return its id."""
    try:
        row = db_query_one(
            """INSERT INTO chat_logs
               (session_id, fingerprint, user_agent, ip_address,
                user_query, bot_answer, route, documents_count,
                web_search_used, strategy, persona, user_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               RETURNING id""",
            (session_id, fingerprint, user_agent, ip_address,
             user_query, bot_answer, route, documents_count, web_search_used, strategy,
             persona, authenticated_user_id),
        )
        return int(row["id"]) if row and row.get("id") is not None else None
    except Exception as e:
        logger.error("Failed to log web chat: %s", e)
        return None


def _update_chat_answer(
    chat_log_id: int, fingerprint: str, bot_answer: str, route: str = "web_chat_regenerated",
    documents_count: int = 0, web_search_used: bool = False, strategy: str = "",
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
                      strategy = %s
                WHERE id = %s AND fingerprint = %s
                RETURNING id""",
            (bot_answer, route, documents_count, web_search_used, strategy, chat_log_id, fingerprint),
        )
        return int(row["id"]) if row and row.get("id") is not None else None
    except Exception as e:
        logger.error("Failed to update regenerated web chat: %s", e)
        return None


# ── SSE helpers ──────────────────────────────────────────────────────

def _format_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


_SSE_HEARTBEAT_INTERVAL_SEC = 15.0

_WEB_TOOL_LABELS = {
    "knowledge_graph_search": "지식 그래프 검색",
    "vector_search": "문서 검색",
    "web_search": "웹 검색",
    "fetch_url": "페이지 읽기",
    "get_finance_data": "시장 데이터 조회",
    "check_wallet": "지갑 정보 확인",
    "read_self": "공개 자기 정보 조회",
    "read_persona_context": "페르소나 자료 조회",
}

_TOOL_PROGRESS_RE = re.compile(r"\]\s*(?:[^\w\s]+)?\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$")
_TOOL_RESULT_RE = re.compile(r"^\s*(?:\S+)\s+([A-Za-z_][A-Za-z0-9_]*):")


def _tool_progress_payload(detail: str, *, done: bool = False) -> dict:
    """Return a public-safe structured tool progress SSE payload."""
    raw = str(detail or "")
    match = _TOOL_RESULT_RE.search(raw) if done else _TOOL_PROGRESS_RE.search(raw)
    tool_name = match.group(1) if match else ""
    label = _WEB_TOOL_LABELS.get(tool_name, tool_name or "도구")
    state = "완료" if done else "사용 중"
    payload = {
        "type": "tool_done" if done else "tool_start",
        "node": "tool",
        "tool_name": tool_name,
        "label": label,
        "content": f"{label} {state}",
    }
    if not done and tool_name == "web_search" and match:
        try:
            args = json.loads(match.group(2))
            query = str(args.get("query") or "").strip()
            if query:
                payload["content"] = f"{label} 중: {query[:120]}"
        except Exception:
            pass
    return payload


# ── Main handler ─────────────────────────────────────────────────────

async def handle_web_chat(
    message: str,
    session_id: str,
    fingerprint: str,
    user_agent: str,
    ip_address: str,
    authenticated_user_id: int | None = None,
    user_fingerprints: list[str] | None = None,
    persona: str = DEFAULT_PERSONA_ID,
    regenerate_from_id: int | None = None,
    tone_feedback: str = "",
    feedback_note: str = "",
):
    """Async generator yielding SSE events for a web chat request."""
    # Resolve the requested persona (unknown ids fall back to the default).
    spec = get_persona(persona)
    persona = spec.id
    web_tools, web_handlers = _build_persona_tools(spec)

    # Authenticated users bring all their bound fingerprints; anonymous users
    # have just the one from localStorage. Deduplicate.
    fps = list({f for f in ([fingerprint] + (user_fingerprints or [])) if f})

    regeneration_source: dict | None = None
    original_message = message
    if regenerate_from_id is not None:
        regeneration_source = await asyncio.to_thread(
            get_web_chat_log_for_feedback,
            regenerate_from_id,
            fps,
            session_id,
            persona,
            account_user_id=authenticated_user_id,
        )
        if (
            not regeneration_source
            or not regeneration_source.get("user_query_active", True)
            or not regeneration_source.get("bot_answer_active", True)
        ):
            yield _format_sse({"type": "error", "content": "재생성할 이전 응답을 찾을 수 없습니다."})
            return
        if tone_feedback or feedback_note:
            await asyncio.to_thread(
                save_web_chat_feedback,
                chat_log_id=int(regeneration_source["id"]),
                session_id=regeneration_source["session_id"],
                fingerprint=fingerprint or regeneration_source["fingerprint"],
                persona=persona,
                rating=None,
                tone_feedback=tone_feedback,
                note=feedback_note,
                pending=False,
            )
        message = _build_regeneration_message(regeneration_source, tone_feedback, feedback_note)
        original_message = str(regeneration_source.get("user_query") or original_message)

    # Load conversation history scoped to this persona + session. During
    # regeneration, exclude the answer being replaced so it cannot contaminate
    # the new answer's conversational context.
    exclude_history_ids = {int(regeneration_source["id"])} if regeneration_source else set()
    history = await asyncio.to_thread(
        _load_web_history, fps, session_id, 20, persona, exclude_history_ids,
        account_user_id=authenticated_user_id,
    )

    # Web chat has its OWN provider/tier keys so Telegram's /config does not
    # bleed into the public site. Corporate LLM only (no "local" — that path
    # is Telegram-dev use). Changes take effect on leninbot-api restart.
    # A persona may pin its own provider/tier (e.g. roleplay → DeepSeek).
    profile = await resolve_runtime_profile(
        "webchat",
        provider_override=spec.provider_override,
        tier_override=spec.tier_override,
    )
    provider = profile.provider
    history_chars = sum(len(str(m.get("content", ""))) for m in history)
    logger.info(
        "Web chat profile session=%s persona=%s provider=%s model=%s tier=%s history_messages=%d history_chars=%d budget=$%.2f",
        session_id, persona, provider, profile.model_id, profile.tier,
        len(history), history_chars, profile.budget_usd,
    )

    # Fold the runtime header directly into the current user turn so the
    # history prefix stays byte-stable across requests (→ prompt caching).
    # Provider-native format: XML for Claude, Markdown for OpenAI/DeepSeek.
    now = datetime.now(KST)
    runtime_context = _build_web_runtime_context(
        now.strftime("%Y-%m-%d %H:%M KST (%A)"),
        provider=provider,
    )
    feedback_rows = []
    feedback_ids: list[int] = []
    tone_policy_rows = []
    if regeneration_source is None:
        feedback_rows = await asyncio.to_thread(
            _load_web_feedback_rows, fps, session_id, persona, 8,
            account_user_id=authenticated_user_id,
        )
        feedback_ids = [int(row["id"]) for row in feedback_rows if row.get("id")]
        tone_policy_rows = await asyncio.to_thread(
            _load_web_tone_policy, fps, session_id, persona, 40,
            account_user_id=authenticated_user_id,
        )
    feedback_context = _render_web_feedback_context(feedback_rows, provider)
    tone_policy_context = _render_web_tone_policy(tone_policy_rows, provider)
    preflight_context = ""
    preflight_tool_detail = ""
    if persona == "gramsci" and regeneration_source is None:
        preflight_query = _build_gramsci_preflight_query(message)
        if preflight_query:
            preflight_context = await _build_gramsci_preflight_context(message, provider)
            if preflight_context:
                preflight_tool_detail = f"[preflight] vector_search({json.dumps({'query': preflight_query, 'layer': 'core_theory', 'author': 'Gramsci', 'num_results': 3}, ensure_ascii=False)})"
    runtime_parts = [runtime_context, _build_web_model_context(profile, provider=provider)]
    if tone_policy_context:
        runtime_parts.append(tone_policy_context)
    if feedback_context:
        runtime_parts.append(feedback_context)
    if preflight_context:
        runtime_parts.append(preflight_context)
    runtime_context = "\n\n".join(runtime_parts)
    history.append({"role": "user", "content": f"{runtime_context}\n\n{message}"})
    system_prompt = render_system_prompt(spec, provider)

    # Progress callback → SSE queue
    queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(event: str, detail: str):
        if event == "tool_call":
            await queue.put(_format_sse(_tool_progress_payload(detail)))
            await queue.put(_format_sse({"type": "log", "node": "tool", "content": detail}))
        elif event == "tool_result":
            await queue.put(_format_sse(_tool_progress_payload(detail, done=True)))
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
    budget_tracker: dict = {}
    if preflight_tool_detail:
        budget_tracker["tool_work_details"] = [preflight_tool_detail]
    web_request_id = uuid.uuid4().hex
    try:
        from redis_state import register_active_web_chat
        register_active_web_chat(web_request_id, session_id=session_id, fingerprint=fingerprint)
    except Exception:
        pass

    async def _run_llm():
        try:
            # Security gateway: public, untrusted, non-owner caller. Runs in its
            # own task (create_task below), so the contextvar is isolated.
            from tool_gateway.security import set_caller, CallerContext
            set_caller(CallerContext(
                interface="webchat",
                user_id=fingerprint or None,
                session_id=session_id,
                is_owner=False,
            ))
            if provider == "openai":
                from openai_tool_loop import chat_with_tools as openai_chat
                result = await openai_chat(
                    history,
                    client=_openai_client,
                    model=profile.model_id,
                    tools=web_tools,
                    tool_handlers=web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=profile.max_rounds,
                    max_tokens=profile.max_tokens,
                    budget_usd=profile.budget_usd,
                    on_progress=on_progress,
                    provider_label=f"{provider}:web",
                    continue_on_length=True,
                    max_length_continuations=2,
                    return_metadata=True,
                    budget_tracker=budget_tracker,
                )
            elif provider == "kimi":
                if not _kimi_client:
                    raise RuntimeError("MOONSHOT_API_KEY is not configured for webchat_provider=kimi")
                from openai_tool_loop import chat_with_tools as openai_chat
                result = await openai_chat(
                    history,
                    client=_kimi_client,
                    model=profile.model_id,
                    tools=web_tools,
                    tool_handlers=web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=profile.max_rounds,
                    max_tokens=profile.max_tokens,
                    budget_usd=profile.budget_usd,
                    on_progress=on_progress,
                    provider_label=f"{provider}:web",
                    extra_body={"reasoning_effort": "max"},
                    sdk_max_token_param="max_tokens",
                    include_parallel_tool_calls=False,
                    preserve_reasoning_content=True,
                    continue_on_length=True,
                    max_length_continuations=2,
                    return_metadata=True,
                    budget_tracker=budget_tracker,
                )
            elif provider == "deepseek":
                if not _deepseek_anthropic_client:
                    raise RuntimeError("DEEPSEEK_API_KEY is not configured for webchat_provider=deepseek")
                result = await chat_with_tools(
                    history,
                    client=_deepseek_anthropic_client,
                    model=profile.model_id,
                    tools=web_tools,
                    tool_handlers=web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=profile.max_rounds,
                    max_tokens=profile.max_tokens,
                    budget_usd=profile.budget_usd,
                    on_progress=on_progress,
                    continue_on_length=True,
                    max_length_continuations=2,
                    budget_tracker=budget_tracker,
                    thinking={"type": "disabled"},
                )
            else:
                result = await chat_with_tools(
                    history,
                    client=_claude,
                    model=profile.model_id,
                    tools=web_tools,
                    tool_handlers=web_handlers,
                    system_prompt=system_prompt,
                    max_rounds=profile.max_rounds,
                    max_tokens=profile.max_tokens,
                    budget_usd=profile.budget_usd,
                    on_progress=on_progress,
                    continue_on_length=True,
                    max_length_continuations=2,
                    budget_tracker=budget_tracker,
                )
            answer_holder.append(result)
        except Exception as e:
            logger.error("Web chat LLM error: %s", e)
            error_holder.append(str(e))
        finally:
            try:
                from redis_state import unregister_active_web_chat
                unregister_active_web_chat(web_request_id)
            except Exception:
                pass
            await queue.put(None)  # sentinel

    llm_task = asyncio.create_task(_run_llm())

    # Yield SSE events as they arrive. Some provider/tool combinations do not
    # emit token deltas, so send SSE comments periodically to keep proxies and
    # browser clients from treating a long model call as a dead connection.
    stream_finished = False
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=_SSE_HEARTBEAT_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if event is None:
                break
            yield event

        stream_finished = True
        await llm_task  # ensure completion
    except (asyncio.CancelledError, GeneratorExit):
        logger.info("Web chat client disconnected; cancelling LLM task session=%s", session_id)
        llm_task.cancel()
        with suppress(asyncio.CancelledError):
            await llm_task
        raise
    finally:
        if not stream_finished and not llm_task.done():
            llm_task.cancel()

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
        rounds_used = metadata.get("rounds", budget_tracker.get("rounds_used"))
        cost_usd = metadata.get("cost_usd", budget_tracker.get("total_cost"))
        documents_count, web_search_used, strategy = _summarize_tool_usage(
            budget_tracker.get("tool_work_details", [])
        )
        # Log to DB BEFORE yield — yield may be the last iteration if client disconnects
        if regeneration_source:
            chat_log_id = await asyncio.to_thread(
                _update_chat_answer,
                int(regeneration_source["id"]),
                regeneration_source["fingerprint"],
                answer,
                f"{provider}_loop_regenerated",
                documents_count, web_search_used, strategy,
            )
        else:
            chat_log_id = await asyncio.to_thread(
                _log_chat, session_id, fingerprint, user_agent, ip_address,
                original_message, answer, f"{provider}_loop",
                documents_count, web_search_used, strategy, persona, authenticated_user_id,
            )
            if feedback_ids:
                try:
                    await asyncio.to_thread(_mark_web_feedback_consumed, feedback_ids)
                except Exception as exc:
                    logger.warning("Failed to mark web chat feedback consumed: %s", exc)
        yield _format_sse({
            "type": "answer",
            "message_id": chat_log_id,
            "regenerated_from_id": regenerate_from_id,
            "content": answer,
            "complete": bool(metadata.get("complete", True)),
            "truncated": bool(metadata.get("truncated", False)),
            "finish_reason": metadata.get("finish_reason"),
            "continuations_used": metadata.get("continuations_used", 0),
            "rounds": rounds_used,
            "cost_usd": cost_usd,
        })
    else:
        yield _format_sse({"type": "error", "content": "응답을 생성하지 못했습니다."})
