"""a2a_handler.py — A2A protocol (message/send) handler.

Implements Google A2A JSON-RPC 2.0 `SendMessage` method for synchronous
conversation. Reuses web_chat.py's LLM pipeline (tools, system prompt)
but returns a completed Task object instead of SSE streaming.

Spec reference: https://a2a-protocol.org/latest/specification/
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from shared import CORE_IDENTITY, KST
from bot_config import (
    _claude, _openai_client, _config,
    _CLAUDE_MAX_TOKENS,
)
from web_chat import _web_tools, _web_handlers

logger = logging.getLogger(__name__)

# ── A2A system prompt (agent-to-agent, not human visitor) ──────────

_A2A_SYSTEM_PROMPT = CORE_IDENTITY + """
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

<tool-strategy>
- Geopolitics → knowledge_graph_search first, then vector_search
- Theory/ideology → vector_search (layer="core_theory")
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page
- Real-time market prices → get_finance_data
- My crypto wallet address/balance → check_wallet
</tool-strategy>

<context>
<current-time>{current_datetime}</current-time>
</context>
"""


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

    # Build system prompt
    now = datetime.now(KST)
    system_prompt = _A2A_SYSTEM_PROMPT.format(
        current_datetime=now.strftime("%Y-%m-%d %H:%M KST (%A)"),
    )

    # Build messages (single-turn for now)
    history = [{"role": "user", "content": user_text}]

    # Run LLM
    try:
        answer = await asyncio.wait_for(
            _run_llm(history, system_prompt),
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
        "kind": "task",
    }

    return {
        "jsonrpc": "2.0",
        "result": task,
        "id": rpc_id,
    }


async def _run_llm(history: list[dict], system_prompt: str) -> str:
    """Run the LLM pipeline (reuses web_chat tool set)."""
    budget = float(_config.get("chat_budget", 0.30))
    if budget <= 0:
        budget = 0.30

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
            tools=_web_tools,
            tool_handlers=_web_handlers,
            system_prompt=system_prompt,
            max_rounds=20,
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
            tools=_web_tools,
            tool_handlers=_web_handlers,
            system_prompt=system_prompt,
            max_rounds=20,
            max_tokens=_CLAUDE_MAX_TOKENS,
            budget_usd=budget,
        )
