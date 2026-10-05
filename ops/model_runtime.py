"""Allowlisted model configuration shared by MCP and the model audit CLI."""
from __future__ import annotations
from typing import Any

CONFIG_FIELDS = ("provider", "task_provider", "autonomous_provider", "webchat_provider",
                 "chat_model", "task_model", "autonomous_model", "webchat_model",
                 "chat_budget", "task_budget", "webchat_budget", "max_rounds_chat", "max_rounds_task")

def _selection(kind: str, provider_override: str | None = None) -> dict[str, Any]:
    from bot_config import get_current_model_selection

    from llm.runtime_profile import runtime_limits, _effective_provider
    row = get_current_model_selection(kind=kind, provider_override=_effective_provider(kind, provider_override))
    rounds, tokens, budget = runtime_limits(kind, row["provider"])
    return {**row, "max_rounds": rounds, "max_output_tokens": tokens, "budget_usd": budget}


def _agent_rows() -> list[dict[str, Any]]:
    from agents import list_agents
    from bot_config import (
        _DEEPSEEK_MODEL_MAP,
        _MODEL_ALIAS_MAP,
        _OPENAI_MODEL_MAP,
        _display_name_for_model_id,
        _resolved_models,
        get_current_model_selection,
    )

    task_selection = get_current_model_selection(kind="task")
    config_task_provider = task_selection["provider"]
    rows = []
    for spec in sorted(list_agents(), key=lambda item: item.name):
        raw_provider = spec.provider
        effective_provider = raw_provider or config_task_provider
        render_provider = spec.effective_provider(config_task_provider)
        if raw_provider in {"codex", "moon"}:
            model_selection = {
                "provider": raw_provider,
                "tier": None,
                "alias": spec.model,
                "model_id": spec.model,
                "display_name": spec.model or raw_provider,
                "resolved": True,
            }
        elif spec.model:
            if effective_provider == "openai":
                model_id = _OPENAI_MODEL_MAP.get(spec.model, spec.model)
            elif effective_provider == "deepseek":
                model_id = _DEEPSEEK_MODEL_MAP.get(spec.model, spec.model)
            elif effective_provider == "claude":
                _model_alias, fallback = _MODEL_ALIAS_MAP.get(spec.model, (spec.model, spec.model))
                model_id = _resolved_models.get(spec.model, fallback)
            else:
                model_id = spec.model
            model_selection = {
                "provider": effective_provider,
                "tier": "override",
                "alias": spec.model,
                "model_id": model_id,
                "display_name": _display_name_for_model_id(model_id),
                "resolved": True,
            }
        else:
            model_selection = get_current_model_selection(kind="task", provider_override=effective_provider)
        rows.append(
            {
                "agent": spec.name,
                "provider_config": raw_provider,
                "provider_effective": effective_provider,
                "prompt_render_provider": render_provider,
                "model_config": spec.model,
                "model_id": model_selection.get("model_id"),
                "display_name": model_selection.get("display_name"),
                "budget_usd": spec.budget_usd,
                "max_rounds": spec.max_rounds,
                "max_input_tokens": spec.max_input_tokens,
                "max_output_tokens": spec.max_output_tokens,
                "max_output_continuations": spec.max_output_continuations,
                "thinking_policy": spec.thinking_policy,
                "thinking_budget_tokens": spec.thinking_budget_tokens,
                "tools": len(spec.tools),
                "finalization_tools": list(spec.finalization_tools),
                "terminal_tools": list(spec.terminal_tools),
                "skip_orchestrator_report": spec.skip_orchestrator_report,
            }
        )
    return rows


def _writer_policy_rows() -> list[dict[str, Any]]:
    from dataclasses import asdict
    from writer.config import WRITER_CALL_POLICIES

    return [
        {"role": role, **asdict(policy)}
        for role, policy in WRITER_CALL_POLICIES.items()
    ]


def build_snapshot() -> dict[str, Any]:
    from bot_config import _config

    return {
        "runtime_config": {key: _config[key] for key in CONFIG_FIELDS if key in _config},
        "scope": "current MCP/CLI process interpretation",
        "running_services": {"status": "unknown", "reason": "no live runtime configuration attestation endpoint"},
        "sources": {"runtime_config": "bot_config defaults < config.json (process load)",
                    "surfaces": "bot_config model selection + llm.runtime_profile.runtime_limits; environment token overrides",
                    "agents": "agents/*.py < config/agent_runtime.json (hot reload)",
                    "writer_call_policies": "writer.config defaults and environment"},
        "surfaces": {
            "telegram_chat": _selection("chat"),
            "delegated_task": _selection("task"),
            "autonomous": _selection("autonomous"),
            "webchat": _selection("webchat"),
        },
        "agents": _agent_rows(),
        "writer_call_policies": _writer_policy_rows(),
    }
