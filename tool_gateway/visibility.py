"""Static registry/profile inventory, never a claim about an active tool session."""

INJECTION_CONDITIONS = {
    "mission": "Registered schema; handler bound to the current Telegram user/task.",
    "run_agent": "Injected by the active orchestrator with its run context and delegation limits.",
    "save_diary": "Diary agent: configured scheduled writing task only; hidden during maintenance.",
    "commulingo_review_decision": "Independent CommuLingo review session; proposal-specific citations.",
}


def tool_inventory(names=None):
    from agents import list_agents
    from runtime_tools.registry import TOOLS, TOOL_HANDLERS, RETIRED_TOOL_NAMES
    from security_gateway.policy import TOOL_RISK_CLASS
    from tool_gateway.profiles import iter_tool_profiles
    from mcp_gateway.tools import GATEWAY_TOOLS

    registered = {t['name'] for t in TOOLS} & set(TOOL_HANDLERS)
    local = {t['name'] for t in GATEWAY_TOOLS}
    exposure = {}
    for profile in iter_tool_profiles():
        for name in profile.tool_names:
            exposure.setdefault(name, []).append(profile.id)
    for spec in list_agents():
        for name in spec.tools:
            exposure.setdefault(name, []).append('agent.' + spec.name)
    selected = set(names) if names is not None else registered | local | set(TOOL_RISK_CLASS) | RETIRED_TOOL_NAMES
    rows = []
    for name in sorted(selected):
        retired = name in RETIRED_TOOL_NAMES
        condition = INJECTION_CONDITIONS.get(name)
        if name.startswith('commulingo_') and name not in registered and not retired:
            condition = condition or 'Stage-specific CommuLingo runner; injected only for its current task and stage.'
        rows.append({'name': name, 'globally_registered': name in registered,
                     'gateway_local': name in local, 'retired': retired,
                     'static_exposure': sorted(exposure.get(name, [])),
                     'injection_condition': condition,
                     'availability_scope': 'static_inventory_not_current_session'})
    return rows
