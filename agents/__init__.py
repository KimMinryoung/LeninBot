"""agents/ — Subagent registry for Cyber-Lenin's delegation system.

Usage:
    from agents import get_agent, list_agents, agent_names

    spec = get_agent("programmer")
    tools, handlers = spec.filter_tools(ALL_TOOLS, ALL_HANDLERS)
    prompt = spec.render_prompt(provider="claude")  # fully static, cacheable
"""

from agents.base import AgentSpec
from agents.programmer import PROGRAMMER
from agents.scout import SCOUT
from agents.visualizer import VISUALIZER
from agents.analyst import ANALYST
from agents.browser import BROWSER
from agents.diary import DIARY
from agents.kollontai import KOLLONTAI
from agents.autonomous import AUTONOMOUS_PROJECT
from agents.runtime_config import apply_agent_runtime_config, reload_agent_runtime_config_if_changed

_REGISTRY: dict[str, AgentSpec] = {
    "programmer": PROGRAMMER,
    "scout": SCOUT,
    "visualizer": VISUALIZER,
    "analyst": ANALYST,
    "browser": BROWSER,
    "diary": DIARY,
    "diplomat": KOLLONTAI,
    "autonomous_project": AUTONOMOUS_PROJECT,
}

apply_agent_runtime_config(_REGISTRY)


def get_agent(name: str) -> AgentSpec:
    """Get an agent spec by name. Raises ValueError if not found."""
    reload_agent_runtime_config_if_changed(_REGISTRY)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_agents() -> list[AgentSpec]:
    """Return all registered agent specs."""
    reload_agent_runtime_config_if_changed(_REGISTRY)
    return list(_REGISTRY.values())


def agent_names() -> list[str]:
    """Return all registered agent names."""
    reload_agent_runtime_config_if_changed(_REGISTRY)
    return list(_REGISTRY.keys())


def agent_descriptions() -> str:
    """Return formatted descriptions for all agents (used in delegate tool)."""
    reload_agent_runtime_config_if_changed(_REGISTRY)
    lines = []
    for spec in _REGISTRY.values():
        lines.append(f"- {spec.name}: {spec.description}")
    return "\n".join(lines)
