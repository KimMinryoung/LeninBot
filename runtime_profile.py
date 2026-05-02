"""runtime_profile.py - Central runtime/provider/model resolution.

Small compatibility layer over bot_config. It gives each call path a single
object describing the provider, model, token/round limits, budget, and prompt
format it should use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    kind: str
    provider: str
    prompt_format: str
    tier: str
    alias: str
    model_id: str
    display_name: str
    max_rounds: int
    max_tokens: int
    budget_usd: float
    resolved: bool


def _coerce_positive_float(value, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = fallback
    return result if result > 0 else fallback


def _coerce_positive_int(value, fallback: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = fallback
    return result if result > 0 else fallback


def _effective_provider(kind: str, provider_override: str | None = None) -> str:
    from bot_config import (
        _config,
        _get_autonomous_provider,
        _get_task_provider,
        _openai_client,
        _deepseek_client,
    )

    if provider_override:
        return provider_override
    if kind == "task":
        return _get_task_provider()
    if kind == "autonomous":
        return _get_autonomous_provider()
    if kind == "webchat":
        provider = str(_config.get("webchat_provider", "claude") or "claude")
        if provider == "local":
            provider = "openai" if _openai_client else "claude"
        if provider == "deepseek" and not _deepseek_client:
            provider = "openai" if _openai_client else "claude"
        if provider == "openai" and not _openai_client:
            provider = "claude"
        return provider
    return str(_config.get("provider", "claude") or "claude")


async def resolve_runtime_profile(
    kind: str = "chat",
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    tier_override: str | None = None,
    budget_override: float | None = None,
    max_rounds_override: int | None = None,
    max_tokens_override: int | None = None,
) -> RuntimeProfile:
    """Resolve provider/model/runtime limits for a chat-like execution path."""
    from bot_config import (
        _CLAUDE_MAX_TOKENS,
        _CLAUDE_MAX_TOKENS_TASK,
        _WEBCHAT_MAX_TOKENS,
        _TIER_MAP,
        _config,
        _display_name_for_model_id,
        _get_model_by_alias,
        _resolved_models,
        _resolve_deepseek_model,
        _resolve_openai_model,
        _resolve_tier,
    )

    kind = kind if kind in {"chat", "task", "webchat", "autonomous"} else "chat"
    provider = _effective_provider(kind, provider_override)
    tier_key = {
        "chat": "chat_model",
        "task": "task_model",
        "webchat": "webchat_model",
        "autonomous": "autonomous_model",
    }[kind]
    tier = str(tier_override or _config.get(tier_key, "high"))
    alias = _resolve_tier(tier, provider=provider)

    resolved = True
    if model_override:
        model_id = model_override
    elif provider == "local":
        from llm.client import _resolve_backend
        model_id = _resolve_backend()["model"]
    elif provider == "openai" or alias in _TIER_MAP.get("openai", {}).values():
        model_id = _resolve_openai_model(alias)
    elif provider == "deepseek" or alias in _TIER_MAP.get("deepseek", {}).values():
        model_id = _resolve_deepseek_model(alias)
    else:
        model_id = await _get_model_by_alias(alias)
        resolved = alias in _resolved_models

    if kind == "chat":
        default_rounds = int(_config.get("max_rounds_chat", 50))
        default_budget = float(_config.get("chat_budget", 0.30))
        default_tokens = _CLAUDE_MAX_TOKENS
    elif kind == "webchat":
        default_rounds = 20
        default_budget = float(_config.get("webchat_budget", _config.get("chat_budget", 0.30)))
        default_tokens = _coerce_positive_int(
            os.getenv("WEBCHAT_MAX_TOKENS"),
            _WEBCHAT_MAX_TOKENS,
        )
    else:
        default_rounds = int(_config.get("max_rounds_task", 50))
        default_budget = float(_config.get("task_budget", 1.00))
        default_tokens = _CLAUDE_MAX_TOKENS_TASK

    max_rounds = default_rounds if max_rounds_override is None else int(max_rounds_override)
    max_tokens = default_tokens if max_tokens_override is None else int(max_tokens_override)
    budget_usd = _coerce_positive_float(budget_override, default_budget)

    return RuntimeProfile(
        kind=kind,
        provider=provider,
        prompt_format="xml" if provider == "claude" else "markdown",
        tier=tier,
        alias=alias,
        model_id=model_id,
        display_name=_display_name_for_model_id(model_id),
        max_rounds=max_rounds,
        max_tokens=max_tokens,
        budget_usd=budget_usd,
        resolved=resolved,
    )
