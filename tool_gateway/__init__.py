"""Unified tool gateway facade for LeninBot runtime tool use.

This package is the runtime control-plane entrypoint for tool visibility,
dispatch, and security/audit integration. It wraps the existing registry,
allow-list, and security gateway modules without changing their behavior.
"""

from tool_gateway.profiles import (
    ToolProfile,
    get_tool_profile,
    iter_tool_profiles,
    profile_tool_names,
)

from tool_gateway.selection import (
    build_toolset,
    filter_agent_tools,
    filter_handlers_by_name,
    select_tools_by_name,
)

__all__ = [
    "ToolProfile",
    "build_toolset",
    "get_tool_profile",
    "iter_tool_profiles",
    "profile_tool_names",
    "filter_agent_tools",
    "filter_handlers_by_name",
    "select_tools_by_name",
]
