"""Runtime tool allow-lists for constrained execution paths."""

from __future__ import annotations

from typing import Any

from tool_gateway.profiles import TELEGRAM_ORCHESTRATOR_TOOLS
from tool_gateway.selection import build_toolset, select_tools_by_name

# Backward-compatible public name; source of truth lives in tool_gateway.profiles.
ORCHESTRATOR_TOOL_NAMES: frozenset[str] = TELEGRAM_ORCHESTRATOR_TOOLS


def select_orchestrator_tools(tools: list[dict]) -> list[dict]:
    """Return the Telegram orchestrator's direct-use tool set."""
    return select_tools_by_name(tools, ORCHESTRATOR_TOOL_NAMES)


def build_orchestrator_toolset(
    tools: list[dict],
    handlers: dict[str, Any],
    extra_handlers: dict[str, Any] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Return the orchestrator schemas and executable handlers.

    Visibility and execution use the same fail-closed allow-list. Dynamic
    handlers such as the chat-bound ``mission`` implementation are accepted
    only when their tool name is already in the orchestrator profile.
    """
    selected_tools, selected_handlers = build_toolset(
        tools,
        handlers,
        ORCHESTRATOR_TOOL_NAMES,
    )
    if extra_handlers:
        selected_handlers.update({
            name: handler
            for name, handler in extra_handlers.items()
            if name in ORCHESTRATOR_TOOL_NAMES
        })
    return selected_tools, selected_handlers
