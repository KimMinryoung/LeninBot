"""Runtime tool allow-lists for constrained execution paths."""

from __future__ import annotations

from tool_gateway.profiles import TELEGRAM_ORCHESTRATOR_TOOLS
from tool_gateway.selection import select_tools_by_name

# Backward-compatible public name; source of truth lives in tool_gateway.profiles.
ORCHESTRATOR_TOOL_NAMES: frozenset[str] = TELEGRAM_ORCHESTRATOR_TOOLS


def select_orchestrator_tools(tools: list[dict]) -> list[dict]:
    """Return the Telegram orchestrator's direct-use tool set."""
    return select_tools_by_name(tools, ORCHESTRATOR_TOOL_NAMES)
