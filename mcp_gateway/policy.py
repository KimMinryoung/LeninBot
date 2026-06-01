"""MCP Gateway tool policy.

The gateway is an inbound surface for developer/operator clients. It does not
export the global runtime registry wholesale; every profile is an explicit
allow-list.
"""

from __future__ import annotations

INSPECT_PROFILE = "inspect"
READONLY_PROFILE_ALIAS = "readonly"
OPERATOR_PROFILE = "operator"

SAFE_RUNTIME_TOOLS: frozenset[str] = frozenset({
    "vector_search",
    "knowledge_graph_search",
    "fetch_url",
})

GATEWAY_INSPECT_TOOLS: frozenset[str] = frozenset({
    "gateway_status",
    "list_mcp_tools",
    "search_dev_docs",
    "get_project_runtime_summary",
    "list_recent_tasks",
    "get_task_status",
    "list_recent_task_reports",
    "corpus_metadata_audit",
    "kg_integrity_check",
})

GATEWAY_OPERATOR_TOOLS: frozenset[str] = frozenset({
    "readonly_query_db",
    "bounded_query_db",
    "kg_maintenance_run",
})

PROFILE_TOOLS: dict[str, frozenset[str]] = {
    INSPECT_PROFILE: SAFE_RUNTIME_TOOLS | GATEWAY_INSPECT_TOOLS,
    OPERATOR_PROFILE: SAFE_RUNTIME_TOOLS | GATEWAY_INSPECT_TOOLS | GATEWAY_OPERATOR_TOOLS,
}

PROFILE_ALIASES: dict[str, str] = {
    READONLY_PROFILE_ALIAS: INSPECT_PROFILE,
}

FORBIDDEN_TOOL_NAMES: frozenset[str] = frozenset({
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


def normalize_profile(profile: str | None) -> str:
    profile = (profile or INSPECT_PROFILE).strip().lower()
    profile = PROFILE_ALIASES.get(profile, profile)
    if profile not in PROFILE_TOOLS:
        return INSPECT_PROFILE
    return profile


def allowed_tool_names(profile: str | None) -> frozenset[str]:
    return PROFILE_TOOLS[normalize_profile(profile)]
