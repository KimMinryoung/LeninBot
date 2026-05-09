#!/usr/bin/env python3
"""Smoke checks for runtime tool allow-lists."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DYNAMIC_AGENT_TOOLS = {
    "*": {
        "save_finding",
    },
    "autonomous_project": {
        "add_research_note",
        "revise_plan",
        "set_project_state",
    },
}

DYNAMIC_HANDLER_TOOLS = {
    "mission",
    "run_agent",
}

REQUIRED_PUBLIC_TOOLS = {
    "read_self",
    "research_document",
    "edit_content",
    "route_task",
    "delegate",
}

REMOVED_PUBLIC_TOOLS = {
    "publish_research",
    "edit_research",
    "private_research_document",
    "save_private_research_document",
    "read_private_research_document",
    "list_private_research_documents",
    "publish_private_research_document",
    "edit_public_post",
}

REQUIRED_COMPAT_HANDLERS = {
    "publish_research",
    "edit_research",
    "private_research_document",
    "save_private_research_document",
    "read_private_research_document",
    "list_private_research_documents",
    "publish_private_research_document",
    "edit_public_post",
}


def _tool_names(tools: list[dict]) -> set[str]:
    return {str(tool.get("name") or "") for tool in tools if tool.get("name")}


def _assert_global_registry() -> tuple[set[str], set[str]]:
    from runtime_tools.registry import TOOLS, TOOL_HANDLERS

    tool_names = _tool_names(TOOLS)
    handler_names = set(TOOL_HANDLERS)
    missing_handlers = sorted(tool_names - handler_names - DYNAMIC_HANDLER_TOOLS)
    assert not missing_handlers, f"tools without handlers: {missing_handlers}"
    missing_public = sorted(REQUIRED_PUBLIC_TOOLS - tool_names)
    assert not missing_public, f"required public tools missing: {missing_public}"
    removed_still_public = sorted(REMOVED_PUBLIC_TOOLS & tool_names)
    assert not removed_still_public, f"removed tools still exposed: {removed_still_public}"
    missing_compat = sorted(REQUIRED_COMPAT_HANDLERS - handler_names)
    assert not missing_compat, f"compat handlers missing: {missing_compat}"
    return tool_names, handler_names


def _assert_orchestrator(tool_names: set[str]) -> None:
    from runtime_tools.allowlists import ORCHESTRATOR_TOOL_NAMES, select_orchestrator_tools
    from runtime_tools.registry import TOOLS

    missing = sorted(set(ORCHESTRATOR_TOOL_NAMES) - tool_names)
    assert not missing, f"orchestrator allowlist references unknown tools: {missing}"
    selected = select_orchestrator_tools(TOOLS)
    selected_names = _tool_names(selected)
    assert selected_names == set(ORCHESTRATOR_TOOL_NAMES)


def _assert_agents(tool_names: set[str]) -> None:
    from agents import list_agents

    for spec in list_agents():
        dynamic = DYNAMIC_AGENT_TOOLS.get("*", set()) | DYNAMIC_AGENT_TOOLS.get(spec.name, set())
        unknown = sorted(set(spec.tools) - tool_names - dynamic)
        assert not unknown, f"{spec.name} allowlist references unknown tools: {unknown}"
        terminal_unknown = sorted(set(spec.terminal_tools) - set(spec.tools))
        assert not terminal_unknown, f"{spec.name} terminal_tools outside allowlist: {terminal_unknown}"
        final_unknown = sorted(set(spec.finalization_tools) - set(spec.tools))
        assert not final_unknown, f"{spec.name} finalization_tools outside allowlist: {final_unknown}"


def _assert_web_chat(tool_names: set[str]) -> None:
    from web_chat import _WEB_ALLOWED_TOOLS, _web_handlers, _web_tools

    missing = sorted(set(_WEB_ALLOWED_TOOLS) - tool_names)
    assert not missing, f"web chat allowlist references unknown tools: {missing}"
    web_tool_names = _tool_names(_web_tools)
    assert "read_self" in web_tool_names
    assert set(_WEB_ALLOWED_TOOLS).issubset(web_tool_names)
    assert "read_self" in _web_handlers


def _assert_delegation_enums() -> None:
    from runtime_tools.registry import TOOLS

    tools_by_name = {str(tool.get("name") or ""): tool for tool in TOOLS}
    delegate_agents = tools_by_name["delegate"]["input_schema"]["properties"]["agent"]["enum"]
    multi_agents = (
        tools_by_name["multi_delegate"]["input_schema"]["properties"]["tasks"]["items"]["properties"]["agent"]["enum"]
    )
    assert "stasova" not in delegate_agents
    assert "stasova" not in multi_agents


def main() -> int:
    tool_names, _handler_names = _assert_global_registry()
    _assert_orchestrator(tool_names)
    _assert_agents(tool_names)
    _assert_web_chat(tool_names)
    _assert_delegation_enums()
    print("tool allowlist smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
