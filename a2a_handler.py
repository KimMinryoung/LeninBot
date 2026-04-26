"""a2a_handler.py — A2A protocol v1.0 handler.

Implements A2A JSON-RPC 2.0 `SendMessage` method for synchronous
conversation with optional skill routing. Reuses web_chat.py's LLM pipeline.

Supported skills:
  - (none / general chat) — default conversational agent
  - geopolitical-analysis — structured geopolitical analysis (KG + theory + web)
  - research-synthesis   — multi-source research report

Spec reference: https://a2a-protocol.org/latest/specification/
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

from shared import CORE_IDENTITY
from llm.prompt_renderer import SystemPrompt, render as _render_prompt
from bot_config import (
    _claude, _openai_client, _deepseek_client, _config,
    _CLAUDE_MAX_TOKENS,
)
from telegram.tools import TOOLS, TOOL_HANDLERS

logger = logging.getLogger(__name__)

# ── Tool sets per skill ──────────────────────────────────────────────

_GENERAL_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "fetch_url",
    "get_finance_data", "check_wallet",
}

_GEOPOLITICAL_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "write_kg_structured",
}

_RESEARCH_TOOLS = {
    "web_search", "knowledge_graph_search", "vector_search",
    "fetch_url",
}


def _build_toolset(allowed: set[str]):
    tools = [t for t in TOOLS if t.get("name") in allowed]
    handlers = {k: v for k, v in TOOL_HANDLERS.items() if k in allowed}
    return tools, handlers


# ── System prompt IR builders ────────────────────────────────────────
#
# The prompt is authored as a SystemPrompt IR and compiled at dispatch
# time to match whichever provider will actually run the turn (XML for
# Claude, Markdown for OpenAI/Qwen). The three shapes — general chat,
# geopolitical-analysis, research-synthesis — share the base identity
# and diverge only in their trailing sections.

_A2A_PREAMBLE = "Operating via A2A (Agent-to-Agent) protocol."

_AUDIENCE_SECTION: tuple[str, str] = (
    "audience",
    (
        "You are communicating with another AI agent, not a human.\n"
        "Be precise, structured, and substantive. Skip pleasantries.\n"
        "Tool results are yours to use freely — relay the substance "
        "without exposing internal implementation details."
    ),
)

_PERSONA_SECTION: tuple[str, str] = (
    "persona",
    (
        "You speak as Cyber-Lenin — direct, analytical, opinionated.\n"
        "Provide clear, well-structured responses suitable for machine consumption.\n"
        "Use the user's language (Korean or English) matching the input."
    ),
)

_GENERAL_TOOL_STRATEGY_SECTION: tuple[str, str] = (
    "tool-strategy",
    (
        "- Geopolitics → knowledge_graph_search first, then vector_search\n"
        "- Theory/ideology → vector_search (layer=\"core_theory\")\n"
        "- Current events → web_search, cross-ref with KG\n"
        "- URL in message → fetch_url to read the page\n"
        "- Real-time market prices → get_finance_data\n"
        "- My crypto wallet address/balance → check_wallet"
    ),
)


def _base_sections() -> list[tuple[str, str]]:
    return [_AUDIENCE_SECTION, _PERSONA_SECTION]


def _general_prompt_ir() -> SystemPrompt:
    return SystemPrompt(
        identity=CORE_IDENTITY.rstrip(),
        preamble=_A2A_PREAMBLE,
        sections=_base_sections() + [_GENERAL_TOOL_STRATEGY_SECTION],
    )


_SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def _load_skill_prompt(skill_name: str) -> str | None:
    path = os.path.join(_SKILL_DIR, skill_name, "SKILL.md")
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return None


def _geopolitical_prompt_ir() -> SystemPrompt:
    skill_md = _load_skill_prompt("geopolitical-analysis") or ""
    return SystemPrompt(
        identity=CORE_IDENTITY.rstrip(),
        preamble=_A2A_PREAMBLE,
        sections=_base_sections() + [
            ("skill", "geopolitical-analysis"),
            ("instructions", skill_md.strip()),
            (
                "tool-strategy",
                (
                    "Follow the 5-step process in the skill instructions exactly.\n"
                    "Available tools: knowledge_graph_search, vector_search, web_search, write_kg_structured."
                ),
            ),
        ],
    )


def _research_prompt_ir() -> SystemPrompt:
    skill_md = _load_skill_prompt("research-report") or ""
    return SystemPrompt(
        identity=CORE_IDENTITY.rstrip(),
        preamble=_A2A_PREAMBLE,
        sections=_base_sections() + [
            ("skill", "research-synthesis"),
            ("instructions", skill_md.strip()),
            (
                "tool-strategy",
                (
                    "Follow the multi-source collection process in the skill instructions.\n"
                    "Available tools: web_search, knowledge_graph_search, vector_search, fetch_url.\n"
                    "Do NOT save files — return the report content directly in your response."
                ),
            ),
        ],
    )


# ── Skill registry ───────────────────────────────────────────────────

_SKILLS = {
    "geopolitical-analysis": {
        "prompt_ir_fn": _geopolitical_prompt_ir,
        "tools": _GEOPOLITICAL_TOOLS,
        "max_rounds": 30,
        "budget": 0.50,
    },
    "research-synthesis": {
        "prompt_ir_fn": _research_prompt_ir,
        "tools": _RESEARCH_TOOLS,
        "max_rounds": 30,
        "budget": 0.50,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────

def _resolve_a2a_provider() -> str:
    """Pick the effective provider for this A2A turn.

    Mirrors the runtime fallback logic in ``_run_llm`` so that the prompt
    is rendered in the format native to whichever SDK will actually be
    called. ``local`` falls back to OpenAI (when the client is wired up)
    or Claude.
    """
    provider = _config.get("provider", "claude")
    if provider == "local":
        provider = "openai" if _openai_client else "claude"
    if provider == "deepseek" and not _deepseek_client:
        provider = "claude"
    return provider


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_error(rpc_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": rpc_id,
    }


def _extract_text(parts: list[dict]) -> str:
    """Extract plain text from A2A message parts."""
    texts = []
    for p in parts:
        if "text" in p:
            texts.append(p["text"])
    return "\n".join(texts)


def _normalize_role(role: str) -> str:
    """Accept both v0.2 and v1.0 role formats."""
    mapping = {"user": "ROLE_USER", "agent": "ROLE_AGENT"}
    return mapping.get(role, role)


# ── Main handler ─────────────────────────────────────────────────────

async def handle_a2a_message(request_body: dict) -> dict:
    """Process a JSON-RPC 2.0 SendMessage request and return a Task."""
    rpc_id = request_body.get("id")
    method = request_body.get("method", "")
    params = request_body.get("params", {})

    # Validate JSON-RPC structure
    if request_body.get("jsonrpc") != "2.0":
        return _make_error(rpc_id, -32600, "Invalid JSON-RPC: missing jsonrpc 2.0")

    if method != "SendMessage":
        return _make_error(rpc_id, -32601, f"Method not found: {method}")

    message = params.get("message", {})
    parts = message.get("parts", [])
    if not parts:
        return _make_error(rpc_id, -32602, "Invalid params: message.parts is empty")

    user_text = _extract_text(parts)
    if not user_text.strip():
        return _make_error(rpc_id, -32602, "Invalid params: no text content in parts")

    task_id = str(uuid.uuid4())
    context_id = message.get("contextId") or str(uuid.uuid4())

    # v1.0: configuration (also accept legacy "config" for interop)
    configuration = params.get("configuration") or params.get("config") or {}

    # Skill routing: check configuration, then metadata
    skill_id = configuration.get("skillId")
    if not skill_id:
        skill_id = (params.get("metadata") or {}).get("skillId")

    skill_cfg = _SKILLS.get(skill_id) if skill_id else None

    if skill_id and not skill_cfg:
        return _make_error(rpc_id, -32602, f"Unknown skill: {skill_id}. Available: {', '.join(_SKILLS.keys())}")

    # Build prompt and tools based on skill
    effective_provider = _resolve_a2a_provider()

    if skill_cfg:
        prompt_ir = skill_cfg["prompt_ir_fn"]()
        tools, handlers = _build_toolset(skill_cfg["tools"])
        max_rounds = skill_cfg["max_rounds"]
        budget = skill_cfg["budget"]
    else:
        prompt_ir = _general_prompt_ir()
        tools, handlers = _build_toolset(_GENERAL_TOOLS)
        max_rounds = 20
        budget = float(_config.get("chat_budget", 0.30)) or 0.30

    # System prompt is fully static (no per-request placeholders) so prompt
    # caching hits across A2A requests. Runtime header (current time + active
    # model) is prepended to the user message below.
    system_prompt = _render_prompt(prompt_ir, effective_provider)

    timeout_sec = 120
    from telegram.bot import (
        _build_runtime_prelude, _join_context_blocks,
        _merge_runtime_context_into_last_user,
    )
    history = _merge_runtime_context_into_last_user(
        [{"role": "user", "content": user_text}],
        _join_context_blocks(_build_runtime_prelude(effective_provider, kind="chat")),
    )

    # Run LLM
    try:
        answer = await asyncio.wait_for(
            _run_llm(history, system_prompt, tools, handlers, max_rounds, budget),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        return _make_error(rpc_id, -32000, "Task timed out")
    except Exception as e:
        logger.error("A2A LLM error: %s", e)
        return _make_error(rpc_id, -32000, f"Internal error: {type(e).__name__}")

    # Build A2A v1.0 Task response
    user_msg_id = message.get("messageId") or str(uuid.uuid4())
    agent_msg_id = str(uuid.uuid4())

    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "TASK_STATE_COMPLETED",
            "timestamp": _now_iso(),
        },
        "history": [
            {
                "messageId": user_msg_id,
                "role": "ROLE_USER",
                "parts": parts,
            },
            {
                "messageId": agent_msg_id,
                "role": "ROLE_AGENT",
                "parts": [{"text": answer}],
            },
        ],
        "artifacts": [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "response",
                "parts": [{"text": answer}],
            }
        ],
        "metadata": {"skillId": skill_id} if skill_id else {},
    }

    return {
        "jsonrpc": "2.0",
        "result": task,
        "id": rpc_id,
    }


# ── LLM runner ───────────────────────────────────────────────────────

async def _run_llm(
    history: list[dict],
    system_prompt: str,
    tools: list[dict],
    handlers: dict,
    max_rounds: int,
    budget: float,
) -> str:
    """Run the LLM pipeline with the given tool set."""
    provider = _config.get("provider", "claude")
    if provider == "local":
        provider = "openai" if _openai_client else "claude"
    if provider == "deepseek" and not _deepseek_client:
        provider = "claude"

    if provider == "openai" and _openai_client:
        from bot_config import _resolve_openai_model, _TIER_MAP
        tier = str(_config.get("chat_model", "high"))
        alias = _TIER_MAP.get("openai", {}).get(tier, tier)
        model = _resolve_openai_model(alias)

        from openai_tool_loop import chat_with_tools as openai_chat
        return await openai_chat(
            history,
            client=_openai_client,
            model=model,
            tools=tools,
            tool_handlers=handlers,
            system_prompt=system_prompt,
            max_rounds=max_rounds,
            max_tokens=_CLAUDE_MAX_TOKENS,
            budget_usd=budget,
            provider_label="openai:a2a",
        )
    elif provider == "deepseek" and _deepseek_client:
        from bot_config import _resolve_deepseek_model, _TIER_MAP
        tier = str(_config.get("chat_model", "high"))
        alias = _TIER_MAP.get("deepseek", {}).get(tier, tier)
        model = _resolve_deepseek_model(alias)

        from openai_tool_loop import chat_with_tools as openai_chat
        return await openai_chat(
            history,
            client=_deepseek_client,
            model=model,
            tools=tools,
            tool_handlers=handlers,
            system_prompt=system_prompt,
            max_rounds=max_rounds,
            max_tokens=_CLAUDE_MAX_TOKENS,
            budget_usd=budget,
            extra_body={"thinking": {"type": "disabled"}},
            sdk_max_token_param="max_tokens",
            provider_label="deepseek:a2a",
        )
    else:
        from bot_config import _get_model_by_alias, _TIER_MAP
        tier = str(_config.get("chat_model", "high"))
        alias = _TIER_MAP.get("claude", {}).get(tier, tier)
        model = await _get_model_by_alias(alias)

        from claude_loop import chat_with_tools
        return await chat_with_tools(
            history,
            client=_claude,
            model=model,
            tools=tools,
            tool_handlers=handlers,
            system_prompt=system_prompt,
            max_rounds=max_rounds,
            max_tokens=_CLAUDE_MAX_TOKENS,
            budget_usd=budget,
        )
