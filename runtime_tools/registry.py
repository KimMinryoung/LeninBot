"""Global runtime tool registry and execution handlers."""

import asyncio
import logging

from runtime_tools.a2a import A2A_TOOL_HANDLERS, A2A_TOOLS
from runtime_tools.fetch import FETCH_TOOL_HANDLERS, FETCH_TOOLS
from runtime_tools.filesystem import FILESYSTEM_TOOL_HANDLERS, FILESYSTEM_TOOLS
from runtime_tools.media import MEDIA_TOOL_HANDLERS, MEDIA_TOOLS
from runtime_tools.social import SOCIAL_TOOL_HANDLERS, SOCIAL_TOOLS
from runtime_tools.web_search import execute_web_search
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)

# restart_service import-preflight targets. Dotted module names — the systemd
# units run dotted package entrypoints; stale flat module names make the
# preflight silently skip services via its isfile guard.
from runtime_tools.restart_service import RESTART_SERVICE_TOOL, restart_service


from runtime_tools.vector_search import exec_vector_search


# ── Tool Definitions (Anthropic API format) ──────────────────────────
TOOLS = [
    {
        "name": "vector_search",
        "description": (
            "Search Marxist-Leninist document DB (pgvector). Returns excerpts with "
            "author/year/title. MATCH YOUR QUERY LANGUAGE TO THE LAYER: "
            "core_theory is English-language classics (Marx, Engels, Lenin, Mao, "
            "Trotsky translations) → query in English. modern_analysis is Korean "
            "analysis/commentary → query in Korean. self_produced_analysis is your "
            "own high-quality saved analysis → query in the language used when saved. "
            "Cross-language queries return near-empty results due to embedding-space separation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Use English for layer=core_theory, Korean for "
                        "layer=modern_analysis. For self_produced_analysis, use the "
                        "same language as the saved analysis. Mismatching language "
                        "to layer degrades recall sharply."
                    ),
                },
                "num_results": {"type": "integer", "description": "Results count (1-10).", "default": 5},
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis", "self_produced_analysis"],
                    "description": (
                        "core_theory: English-language Marxist-Leninist classics. "
                        "modern_analysis: Korean-language contemporary analysis/commentary. "
                        "self_produced_analysis: your own actively saved analytical outputs. "
                        "Omit to search all layers (not recommended — mixes languages)."
                    ),
                },
                "author": {
                    "type": "string",
                    "description": (
                        "Optional metadata filter. Use canonical author names such as "
                        "Stalin, Lenin, Mao, Rosa Luxemburg, Trotsky, Gramsci."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Optional title/source metadata substring filter, e.g. "
                        "'National Question' or 'Chinese Revolution'."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Optional metadata year filter, e.g. 1913.",
                },
                "keywords": {
                    "type": "string",
                    "description": (
                        "Optional exact keyword/phrase filter against chunk text or title. "
                        "Use this when vector similarity alone returns adjacent authors."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "knowledge_graph_search",
        "description": (
            "Search the knowledge graph (Neo4j): people, organizations, events, concepts, "
            "policies and documents across current affairs, the CommuLingo Soviet-history "
            "dictionary (people/terms/events, Korean canonical names) and published research/"
            "archival documents. Facts come back as 'Subject —Predicate→ Object: fact' with "
            "validity dates, trust tier and source. When the query consists of one unambiguous entity name the "
            "result is that entity's full neighbourhood (aliases, external ids, active and "
            "expired facts). Do not invent English names for Korean organizations/publications; "
            "prefer canonical names already used in KG, e.g. '디아마트 (DiaMat)' and "
            "'웹진 반란(Uprising)'. Preserve Korean person names such as '신현준' or '니키타 흐루쇼프' "
            "instead of romanizing them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What entities/relations to find. Preserve proper nouns in "
                        "their known canonical language/name; do not translate or "
                        "romanize Korean organization names unless that spelling is "
                        "part of the canonical name. For Korean people, use the "
                        "Korean name if known."
                    ),
                },
                "num_results": {"type": "integer", "description": "Maximum returned entity/fact items (1-20).", "default": 10},
                "entity": {
                    "type": "string",
                    "description": (
                        "Optional exact entity name or alias (e.g. '니키타 흐루쇼프', 'Nikita Khrushchev'). "
                        "Returns that entity's neighbourhood instead of a semantic search."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "entity", "semantic"],
                    "description": (
                        "auto (default): entity view for an entity name alone, "
                        "else semantic search. entity: force the entity view. semantic: force hybrid search."
                    ),
                    "default": "auto",
                },
            },
            "anyOf": [{"required": ["query"], "properties": {"query": {"pattern": "\\S"}}},
                      {"required": ["entity"], "properties": {"entity": {"pattern": "\\S"}}}],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search missing facts; stop when claims have adequate evidence. Reuse sources for wording/save repairs. "
            "New queries only for gaps, contradictions or plausible changes; no redundant paraphrases. "
            "Known URL: fetch_url. Snippets are leads; fetch before citing figures/quotes. "
            "Use domain parameters, not site operators; queries under 400 chars."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": 1500,
                    "description": (
                        "One missing fact; brief keywords/question with exact names, year/version/location. "
                        "Use source language/original names. No task prompts or answer-format demands."
                    ),
                },
                "max_results": {"type": "integer", "description": "Number of results (1-10).", "default": 5},
                "search_depth": {
                    "type": "string",
                    "enum": ["ultra-fast", "fast", "basic", "advanced"],
                    "description": (
                        "Default basic: 1 Tavily credit. advanced: 2 credits, focused extra context; "
                        "only for difficult unresolved facts. fast/ultra-fast prioritize latency."
                    ),
                    "default": "basic",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "news/finance rank recent coverage higher and return publish dates. Use for current events and market data.",
                    "default": "general",
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Filter by page publication/update recency. Use for current coverage; omit for historical research (put the historical year in query instead).",
                },
                "include_domains": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 10,
                    "description": "Strict domain/subdomain filter; may yield nothing. Prefer known primary sources; omit if unknown. Bare hosts only (docs.python.org), no URLs/paths/wildcards.",
                },
                "exclude_domains": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 10,
                    "description": "Exclude these domains and their subdomains when known to be irrelevant. Bare hostnames only, maximum 10; omit by default.",
                },
                "use_cache": {
                    "type": "boolean",
                    "description": "Reuse identical searches for up to 5 minutes (news/finance/day: 60 seconds). Set false only when a fresh lookup is required; it makes another paid request.",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
    *FILESYSTEM_TOOLS,
    *FETCH_TOOLS,
]


# ── Tool Execution Functions ─────────────────────────────────────────


async def _exec_kg_search(query: str = "", num_results: int = 10, entity: str | None = None,
                          mode: str = "auto") -> str:
    """Execute knowledge graph search (entity view or semantic) off the event loop."""
    try:
        from kg_runtime.search import search_knowledge_graph
        try:
            num_results = max(1, min(int(num_results), 20))
        except (TypeError, ValueError):
            num_results = 10
        result = await asyncio.to_thread(
            search_knowledge_graph, query, num_results, None,
            entity=(entity or "").strip() or None, mode=mode or "auto",
        )
        if not result:
            from tool_gateway.results import ToolResult
            return ToolResult("No knowledge graph results found.", getattr(result, "result_metadata", None))
        return result
    except Exception as e:
        logger.error("kg_search error: %s", e)
        return ToolFailure(f"Knowledge graph search failed; do not treat this as no KG data: {e}")


# ── Research publish/edit/unpublish tools live in runtime_tools.research ──
# They are registered into TOOLS / TOOL_HANDLERS at the bottom of this file.


# ── Mission Tool ──────────────────────────────────────────────────────

MISSION_TOOL = {
    "name": "mission",
    "description": "Manage the active mission (shared context between chat and tasks). Use 'status' to check current mission, 'close' to end a completed mission.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "close"],
                "description": "status: view active mission + recent events. close: end the mission (use when the goal is fully achieved).",
            },
        },
        "required": ["action"],
    },
}


def build_mission_handler(user_id: int):
    """Create a mission tool handler bound to a specific user_id."""
    async def _handle(action: str, **_kwargs) -> str:
        try:
            from telegram.mission import get_active_mission, get_mission_events, close_mission
            if action == "status":
                mission = get_active_mission(user_id)
                if not mission:
                    return "No active mission."
                events = get_mission_events(mission["id"], limit=10)
                lines = [f"Mission #{mission['id']}: {mission['title']} [{mission['status']}]"]
                lines.append(f"Created: {mission['created_at']}")
                if events:
                    lines.append(f"\nTimeline ({len(events)} events):")
                    for e in events:
                        lines.append(f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {str(e['content'] or '')[:200]}")
                return "\n".join(lines)
            elif action == "close":
                mission = get_active_mission(user_id)
                if not mission:
                    return "No active mission to close."
                return close_mission(mission["id"])
            return f"Unknown mission action: {action}"
        except Exception as e:
            return ToolFailure(f"Mission error: {e}")
    return _handle


# ── Web Search (Tavily / Brave provider chain) ───────────────────────

async def _exec_web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    time_range: str | None = None,
    use_cache: bool = True,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    return await execute_web_search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        time_range=time_range,
        use_cache=use_cache,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )


# ── Restart Service Tool ─────────────────────────────────────────────


# ── Handler Registry ─────────────────────────────────────────────────

def dedupe_tool_registry(tools: list[dict]) -> list[dict]:
    """Deduplicate tool registry entries by name while preserving first occurrence.

    Root cause for browser task #330: source code had been patched, but a worker can
    still start or keep running with an inconsistent import/lifecycle state. Keeping
    the registry itself unique makes every downstream caller safer, regardless of
    whether agent-level dedupe runs.
    """
    deduped: list[dict] = []
    seen_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            deduped.append(tool)
            continue
        name = str(tool.get("name", "") or "").strip()
        if name and name in seen_names:
            logger.warning("Dropping duplicate tool from base registry: %s", name)
            continue
        if name:
            seen_names.add(name)
        deduped.append(tool)
    return deduped


TOOL_HANDLERS = {
    "vector_search": exec_vector_search,
    "knowledge_graph_search": _exec_kg_search,
    "web_search": _exec_web_search,
    **FETCH_TOOL_HANDLERS,
    **FILESYSTEM_TOOL_HANDLERS,
    "restart_service": restart_service,
}

# ── Restart service tool ──────────────────────────────────────────────
TOOLS.append(RESTART_SERVICE_TOOL)

# ── R2 Upload + File Registry ────────────────────────────────────────
from runtime_tools.r2_upload import UPLOAD_TO_R2_TOOL, upload_to_r2_tool


TOOLS.append(UPLOAD_TO_R2_TOOL)
TOOL_HANDLERS["upload_to_r2"] = upload_to_r2_tool

# ── Send Email Tool (implementation: mail_runtime/tools.py) ─────────
from mail_runtime.tools import SEND_EMAIL_TOOL, exec_send_email

TOOLS.append(SEND_EMAIL_TOOL)
TOOL_HANDLERS["send_email"] = exec_send_email

# ── Self-awareness tools (shared memory access) ─────────────────────
from self_runtime.tools import SELF_TOOLS, SELF_TOOL_HANDLERS

TOOLS.extend(SELF_TOOLS)
TOOLS.append(MISSION_TOOL)
TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)

# ── Finance data tool (real-time market prices) ──────────────────────
from runtime_tools.finance import FINANCE_TOOL, FINANCE_TOOL_HANDLER

TOOLS.append(FINANCE_TOOL)
TOOL_HANDLERS["get_finance_data"] = FINANCE_TOOL_HANDLER

# ── X/Twitter post lookup ────────────────────────────────────────────
from runtime_tools.x import X_TOOLS, X_TOOL_HANDLERS

TOOLS.extend(X_TOOLS)
TOOL_HANDLERS.update(X_TOOL_HANDLERS)

# ── Site publishing tools (hub curations + static pages for cyber-lenin.com) ──
from runtime_tools.site_publishing import SITE_PUBLISHING_TOOLS, SITE_PUBLISHING_TOOL_HANDLERS

TOOLS.extend(SITE_PUBLISHING_TOOLS)
TOOL_HANDLERS.update(SITE_PUBLISHING_TOOL_HANDLERS)

# ── Direct SQL tool (programmer only; analyst etc. keep read_self/kg_search) ──
from runtime_tools.db import DB_TOOLS, DB_TOOL_HANDLERS

TOOLS.extend(DB_TOOLS)
TOOL_HANDLERS.update(DB_TOOL_HANDLERS)

# ── Public-post editor (UPDATE + Redis cache purge in one step) ──
from runtime_tools.post_edit import POST_EDIT_TOOLS, POST_EDIT_TOOL_HANDLERS

TOOLS.extend(POST_EDIT_TOOLS)
TOOL_HANDLERS.update(POST_EDIT_TOOL_HANDLERS)

# ── Research publish/edit/unpublish (atomic write + cache purge) ──
from runtime_tools.research import RESEARCH_TOOLS, RESEARCH_TOOL_HANDLERS

TOOLS.extend(RESEARCH_TOOLS)
TOOL_HANDLERS.update(RESEARCH_TOOL_HANDLERS)

# ── Wikipedia (free MediaWiki API — no Tavily credits) ──
from runtime_tools.wiki import WIKI_TOOLS, WIKI_TOOL_HANDLERS

TOOLS.extend(WIKI_TOOLS)
TOOL_HANDLERS.update(WIKI_TOOL_HANDLERS)

# ── CommuLingo people dictionary (read + staged edit suggestions) ──
from commulingo.people import COMMULINGO_TOOLS, COMMULINGO_TOOL_HANDLERS

TOOLS.extend(COMMULINGO_TOOLS)
TOOL_HANDLERS.update(COMMULINGO_TOOL_HANDLERS)

# ── Admin-only private research documents (not exposed to public web chat) ──
from runtime_tools.private_reports import PRIVATE_REPORT_TOOLS, PRIVATE_REPORT_TOOL_HANDLERS

TOOLS.extend(PRIVATE_REPORT_TOOLS)
TOOL_HANDLERS.update(PRIVATE_REPORT_TOOL_HANDLERS)

# ── Crypto wallet tools (address + balance + swap + transfer + x402 pay) ───
from crypto_wallet import (
    WALLET_TOOL, WALLET_TOOL_HANDLER,
    SWAP_TOOL, SWAP_TOOL_HANDLER,
    TRANSFER_TOOL, TRANSFER_TOOL_HANDLER,
    PAY_AND_FETCH_TOOL, PAY_AND_FETCH_TOOL_HANDLER,
)

TOOLS.append(WALLET_TOOL)
TOOL_HANDLERS["check_wallet"] = WALLET_TOOL_HANDLER
TOOLS.append(SWAP_TOOL)
TOOL_HANDLERS["swap_eth_to_usdc"] = SWAP_TOOL_HANDLER
TOOLS.append(TRANSFER_TOOL)
TOOL_HANDLERS["transfer_usdc"] = TRANSFER_TOOL_HANDLER
TOOLS.append(PAY_AND_FETCH_TOOL)
TOOL_HANDLERS["pay_and_fetch"] = PAY_AND_FETCH_TOOL_HANDLER

# ── Telegram channel broadcast tool ─────────────────────────────────
from runtime_tools.broadcast import BROADCAST_TO_CHANNEL_TOOL, broadcast_to_channel

TOOLS.append(BROADCAST_TO_CHANNEL_TOOL)
TOOL_HANDLERS["broadcast_to_channel"] = broadcast_to_channel

TOOLS.extend(MEDIA_TOOLS)
TOOL_HANDLERS.update(MEDIA_TOOL_HANDLERS)


# ── Mailbox tools (implementation: mail_runtime/tools.py) ──────────
from mail_runtime.tools import (
    ALLOWLIST_SENDER_TOOL, CHECK_INBOX_TOOL, exec_allowlist_sender, exec_check_inbox,
)

TOOLS.append(CHECK_INBOX_TOOL)
TOOL_HANDLERS["check_inbox"] = exec_check_inbox
TOOLS.append(ALLOWLIST_SENDER_TOOL)
TOOL_HANDLERS["allowlist_sender"] = exec_allowlist_sender

# ── Diary Writer Tool ─────────────────────────────────────────────────
from runtime_tools.diary import SAVE_DIARY_TOOL, save_diary


TOOLS.append(SAVE_DIARY_TOOL)
TOOL_HANDLERS["save_diary"] = save_diary

TOOLS.extend(SOCIAL_TOOLS)
TOOL_HANDLERS.update(SOCIAL_TOOL_HANDLERS)

TOOLS.extend(A2A_TOOLS)
TOOL_HANDLERS.update(A2A_TOOL_HANDLERS)


from mail_runtime.inbox import PREPARE_MAIL_BRIEFING_TOOL, prepare_mail_briefing
TOOLS.append(PREPARE_MAIL_BRIEFING_TOOL)
TOOL_HANDLERS["prepare_mail_briefing"] = prepare_mail_briefing

# Private standalone roleplay notes; handler also enforces caller isolation.
from roleplay.memory import ROLEPLAY_MEMORY_TOOL, roleplay_memory
TOOLS.append(ROLEPLAY_MEMORY_TOOL)
TOOL_HANDLERS["roleplay_memory"] = roleplay_memory

from roleplay.memory import ROLEPLAY_STATE_TOOL, roleplay_state
TOOLS.append(ROLEPLAY_STATE_TOOL)
TOOL_HANDLERS["roleplay_state"] = roleplay_state

from roleplay.memory import ROLEPLAY_PERSON_TOOL, roleplay_person
TOOLS.append(ROLEPLAY_PERSON_TOOL)
TOOL_HANDLERS["roleplay_person"] = roleplay_person


# ── Schema normalization ─────────────────────────────────────────────
#
# Runs last — after every module-level TOOLS.append/extend above and after
# the name dedupe — so every registered tool acquires
# ``additionalProperties: false`` unless it deliberately opts out. Effects per provider:
#   * llama-server: constrains grammar-based tool-call decoding so Qwen
#     can't emit parameter names outside the declared schema.
#   * Anthropic: treats it as advisory (no behavioral change).
#   * OpenAI: strict mode is enabled only when the schema is also
#     "strict-safe" (see openai_tool_loop._convert_tool_anthropic_to_openai).

def _normalize_tool_schemas_inplace(tools: list[dict]) -> None:
    for t in tools:
        schema = t.get("input_schema")
        if not isinstance(schema, dict):
            continue
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False


TOOLS = dedupe_tool_registry(TOOLS)
_normalize_tool_schemas_inplace(TOOLS)
