"""Tool visibility helpers for orchestrators, agents, and external surfaces."""

from __future__ import annotations

from typing import Any


def select_tools_by_name(tools: list[dict], allowed_names: set[str] | frozenset[str]) -> list[dict]:
    """Return tool schemas matching ``allowed_names``, preserving registry order.

    Selection is fail-closed: only explicitly named tools are returned. Duplicate
    schemas are deduped by name so provider payloads stay valid.
    """
    selected: list[dict] = []
    seen: set[str] = set()
    for tool in tools:
        name = str(tool.get("name") or "")
        if name in allowed_names and name not in seen:
            seen.add(name)
            selected.append(tool)
    return selected


def filter_handlers_by_name(handlers: dict[str, Any], allowed_names: set[str] | frozenset[str]) -> dict[str, Any]:
    """Return handlers matching ``allowed_names``."""
    return {name: handler for name, handler in handlers.items() if name in allowed_names}


def build_toolset(
    tools: list[dict],
    handlers: dict[str, Any],
    allowed_names: set[str] | frozenset[str],
) -> tuple[list[dict], dict[str, Any]]:
    """Build the visible schema list and executable handler map for a surface."""
    allowed = frozenset(allowed_names)
    return select_tools_by_name(tools, allowed), filter_handlers_by_name(handlers, allowed)


def filter_agent_tools(spec: Any, all_tools: list[dict], all_handlers: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    """Filter registry tools for an ``AgentSpec``.

    Empty ``spec.tools`` is intentionally fail-closed. Agents that need tools
    must declare each tool explicitly.
    """
    allowed = frozenset(getattr(spec, "tools", ()) or ())
    if not allowed:
        return [], {}
    return build_toolset(all_tools, all_handlers, allowed)
