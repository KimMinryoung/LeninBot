"""MCP Gateway tool policy.

The gateway is an inbound surface for developer/operator clients. It does not
export the global runtime registry wholesale; every profile is an explicit
allow-list.
"""

from __future__ import annotations

from tool_gateway.profiles import (
    MCP_FORBIDDEN_TOOL_NAMES,
    MCP_GATEWAY_INSPECT_TOOLS,
    MCP_GATEWAY_OPERATOR_TOOLS,
    MCP_INSPECT_PROFILE,
    MCP_OPERATOR_PROFILE,
    MCP_SAFE_RUNTIME_TOOLS,
    profile_tool_names,
)

INSPECT_PROFILE = "inspect"
READONLY_PROFILE_ALIAS = "readonly"
OPERATOR_PROFILE = "operator"

# Backward-compatible public names; source of truth lives in tool_gateway.profiles.
SAFE_RUNTIME_TOOLS: frozenset[str] = MCP_SAFE_RUNTIME_TOOLS
GATEWAY_INSPECT_TOOLS: frozenset[str] = MCP_GATEWAY_INSPECT_TOOLS
GATEWAY_OPERATOR_TOOLS: frozenset[str] = MCP_GATEWAY_OPERATOR_TOOLS

PROFILE_TOOLS: dict[str, frozenset[str]] = {
    INSPECT_PROFILE: profile_tool_names(MCP_INSPECT_PROFILE),
    OPERATOR_PROFILE: profile_tool_names(MCP_OPERATOR_PROFILE),
}

PROFILE_ALIASES: dict[str, str] = {
    READONLY_PROFILE_ALIAS: INSPECT_PROFILE,
}

FORBIDDEN_TOOL_NAMES: frozenset[str] = MCP_FORBIDDEN_TOOL_NAMES


def normalize_profile(profile: str | None) -> str:
    profile = (profile or INSPECT_PROFILE).strip().lower()
    profile = PROFILE_ALIASES.get(profile, profile)
    if profile not in PROFILE_TOOLS:
        return INSPECT_PROFILE
    return profile


def allowed_tool_names(profile: str | None) -> frozenset[str]:
    return PROFILE_TOOLS[normalize_profile(profile)]
