"""Runtime policy overlay for AgentSpec objects.

Agent modules own identity, prompts, and tool allowlists. Operational knobs
such as provider, model, budgets, round limits, and terminal/finalization flags
live in config/agent_runtime.json so they can be reviewed and changed without
editing agent prompt code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.base import AgentSpec

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "agent_runtime.json"
_ALLOWED_PROVIDERS = {None, "claude", "openai", "deepseek", "local", "moon", "codex"}
_ALLOWED_KEYS = {
    "provider",
    "model",
    "budget_usd",
    "max_rounds",
    "finalization_tools",
    "terminal_tools",
    "skip_orchestrator_report",
}


def _load_runtime_config(path: Path = _CONFIG_PATH) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("agent runtime config not found: %s", path)
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"agent runtime config must be an object: {path}")
    return data


def _coerce_tool_list(value: Any, key: str, agent_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{agent_name}.{key} must be a list of strings")
    return list(value)


def _apply_one(spec: AgentSpec, cfg: dict[str, Any]) -> None:
    unknown = set(cfg) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown agent runtime keys for {spec.name}: {sorted(unknown)}")

    provider = cfg.get("provider", None)
    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(f"{spec.name}.provider must be one of {sorted(p for p in _ALLOWED_PROVIDERS if p)} or null")

    model = cfg.get("model", None)
    if model is not None and not isinstance(model, str):
        raise ValueError(f"{spec.name}.model must be a string or null")

    budget = float(cfg.get("budget_usd", spec.budget_usd))
    if budget <= 0:
        raise ValueError(f"{spec.name}.budget_usd must be positive")

    max_rounds = int(cfg.get("max_rounds", spec.max_rounds))
    if max_rounds <= 0:
        raise ValueError(f"{spec.name}.max_rounds must be positive")

    spec.provider = provider
    spec.model = model
    spec.budget_usd = budget
    spec.max_rounds = max_rounds
    spec.finalization_tools = _coerce_tool_list(
        cfg.get("finalization_tools", spec.finalization_tools),
        "finalization_tools",
        spec.name,
    )
    spec.terminal_tools = _coerce_tool_list(
        cfg.get("terminal_tools", spec.terminal_tools),
        "terminal_tools",
        spec.name,
    )
    spec.skip_orchestrator_report = bool(
        cfg.get("skip_orchestrator_report", spec.skip_orchestrator_report)
    )


def apply_agent_runtime_config(registry: dict[str, AgentSpec]) -> None:
    config = _load_runtime_config()
    unknown_agents = set(config) - set(registry)
    if unknown_agents:
        raise ValueError(f"agent runtime config references unknown agents: {sorted(unknown_agents)}")

    missing_agents = set(registry) - set(config)
    if missing_agents:
        logger.warning("agent runtime config missing agents: %s", sorted(missing_agents))

    for name, spec in registry.items():
        _apply_one(spec, config.get(name, {}))
