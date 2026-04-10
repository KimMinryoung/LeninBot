"""a2a_handler.py — A2A protocol (message/send) handler.

Implements Google A2A JSON-RPC 2.0 `SendMessage` method for synchronous
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

from shared import CORE_IDENTITY, KST
from bot_config import (
    _claude, _openai_client, _config,
    _CLAUDE_MAX_TOKENS,
)
from telegram_tools import TOOLS, TOOL_HANDLERS

logger = logging.getLogger(__name__)

# ── Tool sets per skill ──────────────────────────────────────────────

_GENERAL_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "fetch_url",
    "get_finance_data", "check_wallet",
}

_GEOPOLITICAL_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "write_kg",
}

_RESEARCH_TOOLS = {
    "web_search", "knowledge_graph_search", "vector_search",
    "fetch_url",
}


def _build_toolset(allowed: set[str]):
    tools = [t for t in TOOLS if t.get("name") in allowed]
    handlers = {k: v for k, v in TOOL_HANDLERS.items() if k in allowed}
    return tools, handlers


# ── System prompts ───────────────────────────────────────────────────

_BASE_PROMPT = CORE_IDENTITY + """
Operating via A2A (Agent-to-Agent) protocol.

<audience>
You are communicating with another AI agent, not a human.
Be precise, structured, and substantive. Skip pleasantries.
Tool results are yours to use freely — relay the substance without exposing internal implementation details.
</audience>

<persona>
You speak as Cyber-Lenin — direct, analytical, opinionated.
Provide clear, well-structured responses suitable for machine consumption.
Use the user's language (Korean or English) matching the input.
</persona>

<context>
<current-time>{current_datetime}</current-time>
</context>
"""

_GENERAL_PROMPT = _BASE_PROMPT + """
<tool-strategy>
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page
- Real-time market prices → get_finance_data
- My crypto wallet address/balance → check_wallet
</tool-strategy>
"""

_SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def _load_skill_prompt(skill_name: str) -> str | None:
    path = os.path.join(_SKILL_DIR, skill_name, "SKILL.md")
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return None


def _geopolitical_prompt() -> str:
    skill_md = _load_skill_prompt("geopolitical-analysis") or ""
    return _BASE_PROMPT + f"""
<skill>geopolitical-analysis</skill>
<instructions>
{skill_md}
</instructions>

<tool-strategy>
Follow the 5-step process in the skill instructions exactly.
Available tools: knowledge_graph_search, vector_search, web_search, write_kg.
</tool-strategy>
"""


def _research_prompt() -> str:
    skill_md = _load_skill_prompt("research-report") or ""
    return _BASE_PROMPT + f"""
<skill>research-synthesis</skill>
<instructions>
{skill_md}
</instructions>

<tool-strategy>
Follow the multi-source collection process in the skill instructions.
Available tools: web_search, knowledge_graph_search, vector_search, fetch_url.
Do NOT save files — return the report content directly in your response.
</tool-strategy>
"""


# ── Skill registry ───────────────────────────────────────────────────

_SKILLS = {
    "geopolitical-analysis": {
        "prompt_fn": _geopolitical_prompt,
        "tools": _GEOPOLITICAL_TOOLS,
        "max_rounds": 30,
        "budget": 0.50,
    },
    "research-synthesis": {
        "prompt_fn": _research_prompt,
        "tools": _RESEARCH_TOOLS,
        "max_rounds": 30,
        "budget": 0.50,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────

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

    task_id = params.get("taskId") or str(uuid.uuid4())
    context_id = params.get("contextId") or str(uuid.uuid4())
    timeout_ms = (params.get("config") or {}).get("timeoutMs", 120_000)

    # Skill routing
    skill_id = (params.get("config") or {}).get("skillId")
    # Also check metadata for skill hint
    if not skill_id:
        skill_id = (params.get("metadata") or {}).get("skillId")

    skill_cfg = _SKILLS.get(skill_id) if skill_id else None

    if skill_id and not skill_cfg:
        return _make_error(rpc_id, -32602, f"Unknown skill: {skill_id}. Available: {', '.join(_SKILLS.keys())}")

    # Build prompt and tools based on skill
    now = datetime.now(KST)
    dt_str = now.strftime("%Y-%m-%d %H:%M KST (%A)")

    if skill_cfg:
        system_prompt = skill_cfg["prompt_fn"]().format(current_datetime=dt_str)
        tools, handlers = _build_toolset(skill_cfg["tools"])
        max_rounds = skill_cfg["max_rounds"]
        budget = skill_cfg["budget"]
    else:
        system_prompt = _GENERAL_PROMPT.format(current_datetime=dt_str)
        tools, handlers = _build_toolset(_GENERAL_TOOLS)
        max_rounds = 20
        budget = float(_config.get("chat_budget", 0.30)) or 0.30

    history = [{"role": "user", "content": user_text}]

    # Run LLM
    try:
        answer = await asyncio.wait_for(
            _run_llm(history, system_prompt, tools, handlers, max_rounds, budget),
            timeout=timeout_ms / 1000,
        )
    except asyncio.TimeoutError:
        return _make_error(rpc_id, -32000, "Task timed out")
    except Exception as e:
        logger.error("A2A LLM error: %s", e)
        return _make_error(rpc_id, -32000, f"Internal error: {type(e).__name__}")

    # Build A2A Task response
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "completed",
            "timestamp": _now_iso(),
        },
        "history": [
            {"role": "user", "parts": parts},
            {"role": "agent", "parts": [{"text": answer}]},
        ],
        "artifacts": [
            {
                "name": "response",
                "parts": [{"type": "text", "text": answer}],
            }
        ],
        "metadata": {"skillId": skill_id} if skill_id else {},
        "kind": "task",
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
