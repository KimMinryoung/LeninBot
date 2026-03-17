"""Core agent — Claude Sonnet 4.6 tool-use loop.

Adapted from telegram_bot.py's _chat_with_tools() pattern.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone, timedelta

import anthropic

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# ── Client & Model ────────────────────────────────────────────────────

_client = anthropic.AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

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

    # Work on a copy so tool-use intermediate messages don't pollute history
    working_msgs = list(messages)

    # Prompt caching
    sys_prompt = _SYSTEM_PROMPT_TEMPLATE.format(current_datetime=_current_datetime_str())
    cached_system = [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}]

    cached_tools = [dict(t) for t in tools]
    if cached_tools:
        cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

    for round_num in range(1, effective_rounds + 1):
        response = await _client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=cached_system,
            tools=cached_tools,
            messages=working_msgs,
        )

        # If no tool use, extract and return text
        if response.stop_reason != "tool_use":
            if response.stop_reason == "max_tokens":
                logger.warning("Response truncated by max_tokens at round %d/%d", round_num, effective_rounds)
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts) if text_parts else "(no response)"

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
                    except Exception as e:
                        logger.error("Tool %s error: %s", block.name, e, exc_info=True)
                        result = f"Tool execution failed: {e}"
                else:
                    result = f"Unknown tool: {block.name}"

                # Truncate very large results to avoid context overflow
                if isinstance(result, str) and len(result) > 30000:
                    result = result[:30000] + f"\n\n... [truncated, total {len(result)} chars]"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        working_msgs.append({"role": "assistant", "content": assistant_content})
        working_msgs.append({"role": "user", "content": tool_results})

    # Round limit reached — force final response without tools
    logger.warning("Tool round limit (%d) reached. Forcing final response.", effective_rounds)
    working_msgs.append({
        "role": "user",
        "content": "[SYSTEM] Tool call limit reached. Answer with what you have gathered so far.",
    })
    try:
        final = await _client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=cached_system,
            messages=working_msgs,
        )
        text_parts = [b.text for b in final.content if b.type == "text"]
        return "\n".join(text_parts) if text_parts else "(no response)"
    except Exception as e:
        logger.error("Final forced response failed: %s", e)
        return f"Error: Tool limit ({effective_rounds}) reached and final response failed: {e}"
