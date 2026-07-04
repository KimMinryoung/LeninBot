"""Named tool profiles for runtime and inbound gateway surfaces.

This module owns the reusable subject/surface allow-list constants. Agent specs
still keep their own role-specific tool lists next to their prompts; the gateway
reads those through ``AgentSpec.filter_tools``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ToolProfile:
    """Explicit tool allow-list for one runtime or gateway surface."""

    id: str
    tool_names: frozenset[str]
    surface: str
    description: str = ""


# -- Shared retrieval sets ------------------------------------------------

READ_SEARCH_TOOLS = frozenset({
    "knowledge_graph_search",
    "vector_search",
    "web_search",
    "fetch_url",
})

# -- Telegram / runtime profiles -----------------------------------------

TELEGRAM_ORCHESTRATOR_PROFILE = "telegram.orchestrator"
TELEGRAM_ORCHESTRATOR_TOOLS = frozenset({
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
    "route_task",
    "list_agent_tools",
    "run_agent",
})

ROLEPLAY_TELEGRAM_PROFILE = "telegram.roleplay"
ROLEPLAY_TELEGRAM_TOOLS = READ_SEARCH_TOOLS

# -- Public web chat profiles --------------------------------------------

WEB_CYBER_LENIN_PROFILE = "web.cyber-lenin"
WEB_CYBER_LENIN_TOOLS = frozenset({
    "knowledge_graph_search",
    "vector_search",
    "web_search",
    "fetch_url",
    "get_finance_data",
    "check_wallet",
    "read_self",
})

WEB_ROLEPLAY_PROFILE = "web.roleplay"
WEB_ROLEPLAY_TOOLS = frozenset({"vector_search", "web_search", "fetch_url"})

# -- A2A skill profiles --------------------------------------------------

A2A_GENERAL_PROFILE = "a2a.general"
A2A_GENERAL_TOOLS = READ_SEARCH_TOOLS | frozenset({"get_finance_data", "check_wallet"})

A2A_GEOPOLITICAL_PROFILE = "a2a.geopolitical-analysis"
A2A_GEOPOLITICAL_TOOLS = frozenset({
    "knowledge_graph_search",
    "vector_search",
    "web_search",
    "write_kg_structured",
})

A2A_RESEARCH_PROFILE = "a2a.research-synthesis"
A2A_RESEARCH_TOOLS = READ_SEARCH_TOOLS

# -- Inbound MCP profiles ------------------------------------------------

MCP_INSPECT_PROFILE = "mcp.inspect"
MCP_OPERATOR_PROFILE = "mcp.operator"

MCP_SAFE_RUNTIME_TOOLS = frozenset({
    "vector_search",
    "knowledge_graph_search",
    "fetch_url",
})

MCP_GATEWAY_INSPECT_TOOLS = frozenset({
    "gateway_status",
    "list_mcp_tools",
    "list_runtime_tool_profiles",
    "search_dev_docs",
    "get_project_runtime_summary",
    "list_recent_tasks",
    "get_task_status",
    "list_recent_task_reports",
    "corpus_metadata_audit",
    "kg_integrity_check",
})

MCP_GATEWAY_OPERATOR_TOOLS = frozenset({
    "readonly_query_db",
    "bounded_query_db",
    "kg_maintenance_run",
})

MCP_GATEWAY_LOCAL_TOOLS = MCP_GATEWAY_INSPECT_TOOLS | MCP_GATEWAY_OPERATOR_TOOLS

MCP_FORBIDDEN_TOOL_NAMES = frozenset({
    "a2a_send",
    "allowlist_sender",
    "broadcast_to_channel",
    "convert_document",
    "download_file",
    "download_image",
    "edit_content",
    "edit_public_post",
    "execute_python",
    "patch_file",
    "pay_and_fetch",
    "publish_comic",
    "publish_hub_curation",
    "publish_private_research_document",
    "publish_research",
    "publish_static_page",
    "publish_static_page_translation",
    "query_db",
    "restart_service",
    "save_diary",
    "save_private_research_document",
    "save_self_analysis",
    "send_email",
    "swap_eth_to_usdc",
    "transfer_usdc",
    "upload_to_r2",
    "write_file",
    "write_kg",
    "write_kg_structured",
})

TOOL_PROFILES: dict[str, ToolProfile] = {
    TELEGRAM_ORCHESTRATOR_PROFILE: ToolProfile(
        id=TELEGRAM_ORCHESTRATOR_PROFILE,
        tool_names=TELEGRAM_ORCHESTRATOR_TOOLS,
        surface="telegram",
        description="Top-level Cyber-Lenin Telegram orchestrator tools.",
    ),
    ROLEPLAY_TELEGRAM_PROFILE: ToolProfile(
        id=ROLEPLAY_TELEGRAM_PROFILE,
        tool_names=ROLEPLAY_TELEGRAM_TOOLS,
        surface="telegram",
        description="Standalone roleplay bot read-only retrieval tools.",
    ),
    WEB_CYBER_LENIN_PROFILE: ToolProfile(
        id=WEB_CYBER_LENIN_PROFILE,
        tool_names=WEB_CYBER_LENIN_TOOLS,
        surface="webchat",
        description="Default public Cyber-Lenin web persona tools.",
    ),
    WEB_ROLEPLAY_PROFILE: ToolProfile(
        id=WEB_ROLEPLAY_PROFILE,
        tool_names=WEB_ROLEPLAY_TOOLS,
        surface="webchat",
        description="Reduced retrieval-only tools for public roleplay personas.",
    ),
    A2A_GENERAL_PROFILE: ToolProfile(
        id=A2A_GENERAL_PROFILE,
        tool_names=A2A_GENERAL_TOOLS,
        surface="a2a",
        description="Default A2A conversation tool profile.",
    ),
    A2A_GEOPOLITICAL_PROFILE: ToolProfile(
        id=A2A_GEOPOLITICAL_PROFILE,
        tool_names=A2A_GEOPOLITICAL_TOOLS,
        surface="a2a",
        description="A2A geopolitical-analysis skill tool profile.",
    ),
    A2A_RESEARCH_PROFILE: ToolProfile(
        id=A2A_RESEARCH_PROFILE,
        tool_names=A2A_RESEARCH_TOOLS,
        surface="a2a",
        description="A2A research-synthesis skill tool profile.",
    ),
    MCP_INSPECT_PROFILE: ToolProfile(
        id=MCP_INSPECT_PROFILE,
        tool_names=MCP_SAFE_RUNTIME_TOOLS | MCP_GATEWAY_INSPECT_TOOLS,
        surface="mcp",
        description="Default read-only developer/operator MCP inspection profile.",
    ),
    MCP_OPERATOR_PROFILE: ToolProfile(
        id=MCP_OPERATOR_PROFILE,
        tool_names=MCP_SAFE_RUNTIME_TOOLS | MCP_GATEWAY_INSPECT_TOOLS | MCP_GATEWAY_OPERATOR_TOOLS,
        surface="mcp",
        description="Trusted local MCP operator profile with bounded diagnostics.",
    ),
}


def get_tool_profile(profile_id: str) -> ToolProfile:
    """Return a named tool profile, raising KeyError for unknown ids."""
    return TOOL_PROFILES[profile_id]


def profile_tool_names(profile_id: str) -> frozenset[str]:
    """Return the explicit tool allow-list for ``profile_id``."""
    return get_tool_profile(profile_id).tool_names


def iter_tool_profiles(*, surface: str | None = None) -> Iterable[ToolProfile]:
    """Iterate registered tool profiles, optionally filtered by surface."""
    for profile in TOOL_PROFILES.values():
        if surface is None or profile.surface == surface:
            yield profile
