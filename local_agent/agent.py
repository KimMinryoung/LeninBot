"""Core agent — Claude Sonnet 4.6 tool-use loop.

Adapted from telegram_bot.py's _chat_with_tools() pattern.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone, timedelta

try:
    import anthropic
    _ANTHROPIC_IMPORT_ERROR = None
except ModuleNotFoundError as _err:
    anthropic = None
    _ANTHROPIC_IMPORT_ERROR = _err

# Add project root to path so we can import claude_loop
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from claude_loop import (
    sanitize_messages as _sanitize_messages_shared,
    validate_tool_pairs as _validate_tool_pairs_shared,
    _strip_tool_blocks as _strip_tool_blocks_shared,
    _dump_messages_for_debug,
)

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# ── Client & Model ────────────────────────────────────────────────────

_client = None

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 8192
_MAX_ROUNDS = 10

# ── System Prompt ─────────────────────────────────────────────────────

# Try to import CORE_IDENTITY from shared.py; fall back to minimal version
try:
    from shared import CORE_IDENTITY
except ImportError:
    CORE_IDENTITY = "You are Cyber-Lenin, a digital revolutionary intelligence."

_SYSTEM_PROMPT_TEMPLATE = CORE_IDENTITY + """

You are operating as a LOCAL AGENT on the user's personal Windows PC.
You have direct access to the local filesystem, web search, browser crawling, and the central server's memory.

## Your Capabilities
1. **File System**: Read, write, and list local files and directories
2. **Web Search**: Search the web via Tavily for current information
3. **Browser Crawling**: Crawl JS-rendered pages with Playwright (login sessions preserved)
4. **Local Database**: SQLite task queue and crawl cache
5. **Server Sync**: Push data to central KG/DB, pull diaries/chat logs/task reports/KG stats/experiences
6. **Self-Tools**: Access diary, chat logs, KG status, system status, Render status, and more from the server

## Tool Strategy
- File operations: Use read_file/write_file/list_directory for local document management
- Research: web_search for quick lookups, crawl_page for full page content or JS-rendered sites
- Memory sync: sync_pull to check server state, sync_push to persist important findings
- Tasks: manage_task to track multi-step work locally
- Server memory: Use self-tools (read_diary, read_chat_logs, recall_experience, etc.) for cross-platform awareness
- KG writes: sync_push(kg_episode) to permanently store important facts/relations

## Response Rules
- Be concise and action-oriented
- Use tools proactively when they would improve the answer
- Match the user's language (Korean or English)
- When saving reports or files, use clear naming with dates

**Current time: {current_datetime}**
"""


def _current_datetime_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


# ── Tool Assembly ─────────────────────────────────────────────────────

def _build_tools_and_handlers() -> tuple[list[dict], dict]:
    """Assemble all tool definitions and handlers."""
    from local_agent.tools import LOCAL_TOOLS
    from local_agent.handlers import LOCAL_TOOL_HANDLERS

    all_tools = list(LOCAL_TOOLS)
    all_handlers = dict(LOCAL_TOOL_HANDLERS)

    # Add self-tools from the server codebase
    try:
        from self_tools import SELF_TOOLS, SELF_TOOL_HANDLERS
        all_tools.extend(SELF_TOOLS)
        all_handlers.update(SELF_TOOL_HANDLERS)
        logger.info("Loaded %d self-tools from server codebase", len(SELF_TOOLS))
    except ImportError as e:
        logger.warning("Could not load self-tools: %s", e)

    return all_tools, all_handlers


_tools_cache = None
_handlers_cache = None


def _get_tools_and_handlers():
    global _tools_cache, _handlers_cache
    if _tools_cache is None:
        _tools_cache, _handlers_cache = _build_tools_and_handlers()
    return _tools_cache, _handlers_cache


# ── Message Sanitization (shared from claude_loop.py) ──────────────────


def _sanitize_messages(msgs: list[dict]) -> list[dict]:
    """Wrapper: sanitize messages without server_tool_use handling (local agent)."""
    return _sanitize_messages_shared(msgs, handle_server_tools=False)


def _validate_messages(msgs: list[dict]) -> list[dict]:
    """Strict local-agent tool pair validation (server tools disabled)."""
    sanitized = _sanitize_messages(msgs)
    return _validate_tool_pairs_shared(sanitized)


def _extract_text_response(response) -> str:
    text_parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    return "\n".join(text_parts) if text_parts else "(no response)"


def _build_client():
    if anthropic is None:
        raise RuntimeError(
            "Missing dependency: anthropic. Install with `pip install anthropic` "
            f"(original import error: {_ANTHROPIC_IMPORT_ERROR})"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your environment or .env file."
        )
    return anthropic.AsyncAnthropic(api_key=api_key)


def _get_client():
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ── Core Loop ─────────────────────────────────────────────────────────

async def chat(
    messages: list[dict],
    on_tool_call: Callable | None = None,
    max_rounds: int | None = None,
) -> str:
    """Run the Claude tool-use loop.

    Args:
        messages: Conversation history (Anthropic format).
        on_tool_call: Optional callback(name, input_summary) for UI updates.
        max_rounds: Override default max rounds.

    Returns:
        Final text response from Claude.
    """
    effective_rounds = max_rounds or _MAX_ROUNDS
    tools, handlers = _get_tools_and_handlers()
    client = _get_client()

    # Work on a copy so tool-use intermediate messages don't pollute history
    working_msgs = list(messages)

    # Prompt caching
    sys_prompt = _SYSTEM_PROMPT_TEMPLATE.format(current_datetime=_current_datetime_str())
    cached_system = [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}]

    cached_tools = [dict(t) for t in tools]
    if cached_tools:
        cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

    for round_num in range(1, effective_rounds + 1):
        working_msgs = _validate_messages(working_msgs)

        try:
            response = await client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                system=cached_system,
                tools=cached_tools,
                messages=working_msgs,
            )
        except Exception as api_err:
            err_str = str(api_err)
            if "tool_use" in err_str and "tool_result" in err_str:
                _dump_messages_for_debug(working_msgs, round_num, api_err)
                logger.warning("Local agent auto-recovery: strict canonicalization retry")
                strict_msgs = _validate_messages(working_msgs)
                try:
                    response = await client.messages.create(
                        model=_MODEL,
                        max_tokens=_MAX_TOKENS,
                        system=cached_system,
                        tools=cached_tools,
                        messages=strict_msgs,
                    )
                    working_msgs = strict_msgs
                except Exception as strict_err:
                    logger.warning(
                        "Local agent strict retry failed; falling back to text-only final response: %s",
                        strict_err,
                    )
                    fallback_msgs = _strip_tool_blocks_shared(strict_msgs)
                    final = await client.messages.create(
                        model=_MODEL,
                        max_tokens=_MAX_TOKENS,
                        system=cached_system,
                        messages=fallback_msgs,
                    )
                    return _extract_text_response(final)
            else:
                raise

        # If no tool use-like stop, extract and return text
        if response.stop_reason not in ("tool_use", "pause_turn"):
            if response.stop_reason == "max_tokens":
                logger.warning("Response truncated by max_tokens at round %d/%d", round_num, effective_rounds)
            return _extract_text_response(response)

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

                # Notify UI
                input_summary = json.dumps(block.input, ensure_ascii=False)
                if len(input_summary) > 120:
                    input_summary = input_summary[:120] + "..."
                if on_tool_call:
                    on_tool_call(block.name, input_summary)

                # Execute tool
                handler = handlers.get(block.name)
                if handler:
                    try:
                        result = await handler(**block.input)
                        is_error = False
                    except Exception as e:
                        logger.error("Tool %s error: %s", block.name, e, exc_info=True)
                        result = f"Tool execution failed: {e}"
                        is_error = True
                else:
                    result = f"Unknown tool: {block.name}"
                    is_error = True
                if not isinstance(result, str):
                    result = str(result) if result is not None else "(no result)"

                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
                if is_error:
                    tool_result_block["is_error"] = True
                tool_results.append(tool_result_block)
            else:
                # Preserve unknown future block types as text to avoid losing context.
                assistant_content.append({
                    "type": "text",
                    "text": f"[unsupported block:{getattr(block, 'type', 'unknown')}]",
                })

        # Safety net: ensure EVERY tool_use block has a matching tool_result
        resolved_ids = {r["tool_use_id"] for r in tool_results}
        for block in assistant_content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block["id"] not in resolved_ids:
                logger.warning("Safety net: missing tool_result for tool_use id=%s name=%s", block["id"], block.get("name"))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Tool execution skipped (internal error): no result for {block.get('name', 'unknown')}",
                    "is_error": True,
                })

        working_msgs.append({"role": "assistant", "content": assistant_content})
        working_msgs.append({"role": "user", "content": tool_results})

    # Round limit reached — force final response without tools
    logger.warning("Tool round limit (%d) reached. Forcing final response.", effective_rounds)
    working_msgs.append({
        "role": "user",
        "content": "[SYSTEM] Tool call limit reached. Answer with what you have gathered so far.",
    })
    working_msgs = _sanitize_messages(working_msgs)
    try:
        final = await client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=cached_system,
            messages=working_msgs,
        )
        return _extract_text_response(final)
    except Exception as e:
        logger.error("Final forced response failed: %s", e)
        return f"Error: Tool limit ({effective_rounds}) reached and final response failed: {e}"
