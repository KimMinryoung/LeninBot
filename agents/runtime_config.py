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
_last_mtime_ns: int | None = None
_base_runtime: dict[str, dict[str, Any]] = {}


def _load_runtime_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or _CONFIG_PATH
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


def _snapshot_runtime(spec: AgentSpec) -> dict[str, Any]:
    return {
        "provider": spec.provider,
        "model": spec.model,
        "budget_usd": spec.budget_usd,
        "max_rounds": spec.max_rounds,
        "finalization_tools": list(spec.finalization_tools),
        "terminal_tools": list(spec.terminal_tools),
        "skip_orchestrator_report": spec.skip_orchestrator_report,
    }


def _apply_one(spec: AgentSpec, base: dict[str, Any], cfg: dict[str, Any]) -> None:
    unknown = set(cfg) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown agent runtime keys for {spec.name}: {sorted(unknown)}")

    provider = cfg.get("provider", base["provider"])
    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(f"{spec.name}.provider must be one of {sorted(p for p in _ALLOWED_PROVIDERS if p)} or null")

    model = cfg.get("model", base["model"])
    if model is not None and not isinstance(model, str):
        raise ValueError(f"{spec.name}.model must be a string or null")

    budget = float(cfg.get("budget_usd", base["budget_usd"]))
    if budget <= 0:
        raise ValueError(f"{spec.name}.budget_usd must be positive")

    max_rounds = int(cfg.get("max_rounds", base["max_rounds"]))
    if max_rounds <= 0:
        raise ValueError(f"{spec.name}.max_rounds must be positive")

    spec.provider = provider
    spec.model = model
    spec.budget_usd = budget
    spec.max_rounds = max_rounds
    spec.finalization_tools = _coerce_tool_list(
        cfg.get("finalization_tools", base["finalization_tools"]),
        "finalization_tools",
        spec.name,
    )
    spec.terminal_tools = _coerce_tool_list(
        cfg.get("terminal_tools", base["terminal_tools"]),
        "terminal_tools",
        spec.name,
    )
    spec.skip_orchestrator_report = bool(
        cfg.get("skip_orchestrator_report", base["skip_orchestrator_report"])
    )


def _apply_config(registry: dict[str, AgentSpec], config: dict[str, dict[str, Any]]) -> None:
    unknown_agents = set(config) - set(registry)
    if unknown_agents:
        raise ValueError(f"agent runtime config references unknown agents: {sorted(unknown_agents)}")

    missing_agents = set(registry) - set(config)
    if missing_agents:
        logger.warning("agent runtime config missing agents: %s", sorted(missing_agents))

    for name, spec in registry.items():
        _apply_one(spec, _base_runtime[name], config.get(name, {}))


def apply_agent_runtime_config(registry: dict[str, AgentSpec]) -> None:
    """Load and apply runtime policy once at startup.

    This is strict: invalid startup config raises so the process does not run
    with misleading agent policy.
    """
    global _last_mtime_ns, _base_runtime
    _base_runtime = {name: _snapshot_runtime(spec) for name, spec in registry.items()}
    config = _load_runtime_config()
    _apply_config(registry, config)
    try:
        _last_mtime_ns = _CONFIG_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        _last_mtime_ns = None


def reload_agent_runtime_config_if_changed(registry: dict[str, AgentSpec]) -> bool:
    """Reload config/agent_runtime.json when it changes.

    Runtime reload is fail-safe: invalid JSON or validation errors are logged and
    the previously applied in-memory policy remains active.
    """
    global _last_mtime_ns
    try:
        mtime_ns = _CONFIG_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = None

    if mtime_ns == _last_mtime_ns:
        return False

    try:
        config = _load_runtime_config()
        _apply_config(registry, config)
    except Exception as exc:
        logger.error("agent runtime config reload failed; keeping previous policy: %s", exc)
        return False

    _last_mtime_ns = mtime_ns
    logger.info("agent runtime config reloaded from %s", _CONFIG_PATH)
    return True
