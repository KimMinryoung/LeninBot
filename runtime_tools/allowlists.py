"""Runtime tool allow-lists for constrained execution paths."""

from __future__ import annotations

ORCHESTRATOR_TOOL_NAMES: frozenset[str] = frozenset({
    "delegate",
    "multi_delegate",
    "mission",
    "web_search",
    "fetch_url",
    "fetch_x_post",
    "knowledge_graph_search",
    "vector_search",
    "get_finance_data",
    "broadcast_to_channel",
    "recall_experience",
    "save_self_analysis",
    "write_kg_structured",
    "read_self",
    "list_agent_tools",
    "run_agent",
})


def select_tools_by_name(tools: list[dict], allowed_names: set[str] | frozenset[str]) -> list[dict]:
    """Return tools matching allowed_names, preserving registry order and deduping names."""
    selected: list[dict] = []
    seen: set[str] = set()
    for tool in tools:
        name = str(tool.get("name") or "")
        if name in allowed_names and name not in seen:
            seen.add(name)
            selected.append(tool)
    return selected


def select_orchestrator_tools(tools: list[dict]) -> list[dict]:
    """Return the Telegram orchestrator's direct-use tool set."""
    return select_tools_by_name(tools, ORCHESTRATOR_TOOL_NAMES)
