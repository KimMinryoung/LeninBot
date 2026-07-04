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
        "read_research_notes",
        "read_document",
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
    "private_research_document",
    "save_private_research_document",
    "read_private_research_document",
    "list_private_research_documents",
    "publish_private_research_document",
    "edit_public_post",
}

# Single source of truth: the risk taxonomy now lives in security_gateway.policy
# (the tool-call gateway enforces and audits against it). Imported here so the
# allow-list smoke checks validate the same map the runtime gateway uses.
from security_gateway.policy import TOOL_RISK_CLASS  # noqa: E402
from tool_gateway.profiles import (  # noqa: E402
    MCP_GATEWAY_LOCAL_TOOLS,
    MCP_INSPECT_PROFILE,
    MCP_OPERATOR_PROFILE,
    TELEGRAM_ORCHESTRATOR_TOOLS,
    WEB_CYBER_LENIN_TOOLS,
    WEB_ROLEPLAY_TOOLS,
    iter_tool_profiles,
    profile_tool_names,
)

WEB_FORBIDDEN_RISK_CLASSES = {
    "admin",
    "browser",
    "delegate",
    "execute",
    "file_read",
    "file_write",
    "pay",
    "publish",
    "send",
    "state",
    "write",
}

WEB_ALLOWED_RISK_CLASSES = {
    "fetch",
    "read",
    "wallet_read",
}


def _tool_names(tools: list[dict]) -> set[str]:
    return {str(tool.get("name") or "") for tool in tools if tool.get("name")}


def _risk_class(tool_name: str) -> str:
    return TOOL_RISK_CLASS.get(tool_name, "uncategorized")


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
    uncategorized = sorted(name for name in tool_names if _risk_class(name) == "uncategorized")
    assert not uncategorized, f"tool risk classes missing: {uncategorized}"
    return tool_names, handler_names


def _assert_tool_profiles(tool_names: set[str]) -> None:
    for profile in iter_tool_profiles():
        assert profile.tool_names, f"{profile.id} has empty fail-closed tool profile"
        unknown = sorted(set(profile.tool_names) - tool_names - MCP_GATEWAY_LOCAL_TOOLS)
        assert not unknown, f"{profile.id} references unknown tools: {unknown}"
        if profile.surface == "webchat":
            forbidden = sorted(
                name for name in profile.tool_names
                if _risk_class(name) in WEB_FORBIDDEN_RISK_CLASSES
            )
            assert not forbidden, f"{profile.id} exposes forbidden web risk classes: {forbidden}"
            unexpected = sorted(
                name for name in profile.tool_names
                if _risk_class(name) not in WEB_ALLOWED_RISK_CLASSES
            )
            assert not unexpected, f"{profile.id} exposes unexpected web risk classes: {unexpected}"


def _assert_orchestrator(tool_names: set[str]) -> None:
    from runtime_tools.allowlists import ORCHESTRATOR_TOOL_NAMES, select_orchestrator_tools
    from runtime_tools.registry import TOOLS

    assert set(ORCHESTRATOR_TOOL_NAMES) == set(TELEGRAM_ORCHESTRATOR_TOOLS)
    missing = sorted(set(ORCHESTRATOR_TOOL_NAMES) - tool_names)
    assert not missing, f"orchestrator allowlist references unknown tools: {missing}"
    selected = select_orchestrator_tools(TOOLS)
    selected_names = _tool_names(selected)
    assert selected_names == set(ORCHESTRATOR_TOOL_NAMES)
    forbidden = sorted(
        name for name in selected_names
        if _risk_class(name) in {"admin", "execute", "file_read", "file_write", "pay"}
    )
    assert not forbidden, f"orchestrator exposes high-risk direct tools: {forbidden}"


def _assert_agents(tool_names: set[str]) -> None:
    from agents import list_agents
    from agents.base import AgentSpec
    from llm.prompt_renderer import SystemPrompt

    for spec in list_agents():
        assert spec.tools, f"{spec.name} has empty fail-closed tool allowlist"
        dynamic = DYNAMIC_AGENT_TOOLS.get("*", set()) | DYNAMIC_AGENT_TOOLS.get(spec.name, set())
        unknown = sorted(set(spec.tools) - tool_names - dynamic)
        assert not unknown, f"{spec.name} allowlist references unknown tools: {unknown}"
        terminal_unknown = sorted(set(spec.terminal_tools) - set(spec.tools))
        assert not terminal_unknown, f"{spec.name} terminal_tools outside allowlist: {terminal_unknown}"
        final_unknown = sorted(set(spec.finalization_tools) - set(spec.tools))
        assert not final_unknown, f"{spec.name} finalization_tools outside allowlist: {final_unknown}"
        uncategorized = sorted(name for name in spec.tools if _risk_class(name) == "uncategorized")
        assert not uncategorized, f"{spec.name} exposes uncategorized tools: {uncategorized}"

    dummy = AgentSpec(
        name="dummy",
        description="empty tools must not expose the global registry",
        prompt_ir=SystemPrompt(identity="dummy"),
        tools=[],
    )
    filtered_tools, filtered_handlers = dummy.filter_tools(
        [{"name": "read_file"}, {"name": "write_file"}],
        {"read_file": object(), "write_file": object()},
    )
    assert filtered_tools == []
    assert filtered_handlers == {}


def _assert_web_chat(tool_names: set[str]) -> None:
    from web_chat import _WEB_ALLOWED_TOOLS, _web_handlers, _web_tools
    from web_personas import CYBER_LENIN_TOOLS, ROLEPLAY_TOOLS

    assert set(CYBER_LENIN_TOOLS) == set(WEB_CYBER_LENIN_TOOLS)
    assert set(ROLEPLAY_TOOLS) == set(WEB_ROLEPLAY_TOOLS)
    assert set(_WEB_ALLOWED_TOOLS) == set(WEB_CYBER_LENIN_TOOLS) - {"read_self"}
    missing = sorted(set(_WEB_ALLOWED_TOOLS) - tool_names)
    assert not missing, f"web chat allowlist references unknown tools: {missing}"
    web_tool_names = _tool_names(_web_tools)
    assert "read_self" in web_tool_names
    assert set(_WEB_ALLOWED_TOOLS).issubset(web_tool_names)
    assert "read_self" in _web_handlers
    web_risks = {name: _risk_class(name) for name in web_tool_names}
    forbidden = sorted(
        name for name, risk in web_risks.items()
        if risk in WEB_FORBIDDEN_RISK_CLASSES
    )
    assert not forbidden, f"web chat exposes forbidden tool risk classes: {forbidden}"
    unexpected = sorted(
        name for name, risk in web_risks.items()
        if risk not in WEB_ALLOWED_RISK_CLASSES
    )
    assert not unexpected, f"web chat exposes unexpected risk classes: {unexpected}"
    assert web_risks.get("check_wallet") == "wallet_read"
    assert "check_wallet" in _WEB_ALLOWED_TOOLS


def _assert_mcp_profiles() -> None:
    from mcp_gateway.policy import allowed_tool_names

    assert allowed_tool_names("inspect") == profile_tool_names(MCP_INSPECT_PROFILE)
    assert allowed_tool_names("readonly") == profile_tool_names(MCP_INSPECT_PROFILE)
    assert allowed_tool_names("operator") == profile_tool_names(MCP_OPERATOR_PROFILE)


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
    _assert_tool_profiles(tool_names)
    _assert_orchestrator(tool_names)
    _assert_agents(tool_names)
    _assert_web_chat(tool_names)
    _assert_mcp_profiles()
    _assert_delegation_enums()
    print("tool allowlist smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
