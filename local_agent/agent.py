"""Core agent — delegates to claude_loop.chat_with_tools() for robustness.

Budget tracking, multi-layer error recovery, and safety nets are all
inherited from the shared loop used by telegram_bot.
"""

from __future__ import annotations

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

# Add project root to path so we can import claude_loop / shared
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from claude_loop import chat_with_tools

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# ── Client & Model ────────────────────────────────────────────────────

_client = None

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 8192
_MAX_ROUNDS = 15
_BUDGET_USD = 0.50  # local agent default (higher than telegram chat)

# ── System Prompt ─────────────────────────────────────────────────────

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


# ── Client ────────────────────────────────────────────────────────────

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


# ── Core Loop (delegates to claude_loop) ──────────────────────────────

async def chat(
    messages: list[dict],
    on_tool_call: Callable | None = None,
    max_rounds: int | None = None,
    budget_usd: float | None = None,
) -> tuple[str, dict]:
    """Run the Claude tool-use loop via claude_loop.chat_with_tools().

    Args:
        messages: Conversation history (Anthropic format).
        on_tool_call: Optional callback(name, input_summary) for UI updates.
        max_rounds: Override default max rounds.
        budget_usd: Override default budget.

    Returns:
        Tuple of (response_text, budget_info_dict).
        budget_info_dict contains: total_cost, rounds_used, was_interrupted.
    """
    tools, handlers = _get_tools_and_handlers()
    client = _get_client()
    sys_prompt = _SYSTEM_PROMPT_TEMPLATE.format(current_datetime=_current_datetime_str())

    budget_tracker: dict = {}

    # Bridge on_tool_call callback to on_progress
    async def _on_progress(event: str, detail: str):
        if on_tool_call and event == "tool_call":
            # Extract tool name and input from detail like "[1] 🔧 web_search({...})"
            on_tool_call(event, detail)

    reply = await chat_with_tools(
        messages,
        client=client,
        model=_MODEL,
        tools=tools,
        tool_handlers=handlers,
        system_prompt=sys_prompt,
        max_rounds=max_rounds or _MAX_ROUNDS,
        max_tokens=_MAX_TOKENS,
        budget_usd=budget_usd or _BUDGET_USD,
        budget_tracker=budget_tracker,
        on_progress=_on_progress,
    )

    return reply, budget_tracker
