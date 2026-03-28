"""agents/ — Subagent registry for Cyber-Lenin's delegation system.

Usage:
    from agents import get_agent, list_agents, agent_names

    spec = get_agent("programmer")
    tools, handlers = spec.filter_tools(ALL_TOOLS, ALL_HANDLERS)
    prompt = spec.render_prompt(current_datetime="2026-03-26 15:00 KST")
"""

from agents.base import AgentSpec
from agents.general import GENERAL
from agents.programmer import PROGRAMMER
from agents.scout import SCOUT
from agents.visualizer import VISUALIZER

_REGISTRY: dict[str, AgentSpec] = {
    "general": GENERAL,
    "programmer": PROGRAMMER,
    "scout": SCOUT,
    "visualizer": VISUALIZER,
}


def get_agent(name: str) -> AgentSpec:
    """Get an agent spec by name. Raises ValueError if not found."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_agents() -> list[AgentSpec]:
    """Return all registered agent specs."""
    return list(_REGISTRY.values())


def agent_names() -> list[str]:
    """Return all registered agent names."""
    return list(_REGISTRY.keys())


def agent_descriptions() -> str:
    """Return formatted descriptions for all agents (used in delegate tool)."""
    lines = []
    for spec in _REGISTRY.values():
        lines.append(f"- {spec.name}: {spec.description}")
    return "\n".join(lines)
