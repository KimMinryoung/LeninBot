"""bot_config.py — LLM client setup, runtime config, and model resolution."""

import os
import json
import asyncio
import logging

import anthropic

from secrets_loader import get_secret

logger = logging.getLogger(__name__)

# ── API Keys ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY", "") or ""
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "") or ""

# ── LLM Clients ──────────────────────────────────────────────────────
_claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# OpenAI client (lazy — only created if key exists)
_openai_client = None
if OPENAI_API_KEY:
    from openai import AsyncOpenAI
    _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
_CLAUDE_MAX_TOKENS = 4096
_CLAUDE_MAX_TOKENS_TASK = 16384  # Tasks need longer output for full reports

# ── Runtime Config (mutable at runtime via /config) ──────────────────
_CONFIG_DEFAULTS = {
    "chat_budget": 0.30,       # USD per chat turn
    "task_budget": 1.00,       # USD per background task
    "chat_model": "high",      # "high" | "medium" | "low"
    "task_model": "high",      # "high" | "medium" | "low"
    "max_rounds_chat": 50,
    "max_rounds_task": 50,
    "provider": "claude",      # "claude" | "openai" | "local"
    "task_concurrency": 2,     # max parallel background tasks
    "autonomous_active": True, # toggle the hourly autonomous project loop (run_tick)
    # Web chat runs independently from Telegram's /config. These keys pin what
    # cyber-lenin.com users get; the API service snapshots them at startup
    # (bot_config is imported once, no live reload), so edits take effect on
    # the next `systemctl restart leninbot-api`.
    "webchat_provider": "claude",  # "claude" | "openai"  (no "local" — web uses corporate LLM)
    "webchat_model":    "medium",  # tier: "high" | "medium" | "low"
}

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config() -> dict:
    """Load config from disk, falling back to defaults for missing keys."""
    config = dict(_CONFIG_DEFAULTS)
    try:
        with open(_CONFIG_PATH, "r") as f:
            saved = json.loads(f.read())
        for key, val in saved.items():
            if key in _CONFIG_DEFAULTS:
                # Ensure type matches default
                default_val = _CONFIG_DEFAULTS[key]
                # bool check must come before int — bool is a subclass of int.
                if isinstance(default_val, bool):
                    config[key] = bool(val)
                elif isinstance(default_val, float):
                    config[key] = float(val)
                elif isinstance(default_val, int):
                    config[key] = int(val)
                else:
                    config[key] = str(val)
        logger.info("Config loaded from %s: %s", _CONFIG_PATH, config)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Config load failed, using defaults: %s", e)
    return config


def _save_config():
    """Persist current config to disk."""
    try:
        with open(_CONFIG_PATH, "w") as f:
            f.write(json.dumps(_config, indent=2))
    except Exception as e:
        logger.warning("Config save failed: %s", e)


_config = _load_config()

# Display metadata for config panel
_CONFIG_META = {
    "chat_budget":      {"label": "대화 예산",     "unit": "$",  "options": [0.10, 0.20, 0.30, 0.50, 1.00]},
    "task_budget":      {"label": "태스크 예산",   "unit": "$",  "options": [0.50, 1.00, 2.00, 3.00, 5.00]},
    "chat_model":       {"label": "대화 모델",     "unit": "",   "options": ["high", "medium", "low"]},
    "task_model":       {"label": "태스크 모델",   "unit": "",   "options": ["high", "medium", "low"]},
    "max_rounds_chat":  {"label": "대화 라운드",   "unit": "회", "options": [15, 30, 50, 80]},
    "max_rounds_task":  {"label": "태스크 라운드", "unit": "회", "options": [15, 30, 50, 80]},
    "provider":         {"label": "LLM 제공자",   "unit": "",   "options": ["claude", "openai", "local"]},
    "task_concurrency": {"label": "동시 태스크",  "unit": "개", "options": [1, 2, 3, 4]},
    "autonomous_active":{"label": "자율 에이전트", "unit": "",   "options": [True, False]},
}

_MODEL_ALIAS_MAP = {
    "haiku":  ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    "sonnet": ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    "opus":   ("claude-opus-4-7", "claude-opus-4-7"),
}

_OPENAI_MODEL_MAP = {
    "gpt54":     "gpt-5.5",
    "gpt54mini": "gpt-5.5-mini",
    "gpt54nano": "gpt-5.5-nano",
}

# Human-readable display names keyed by API model ID. Used when injecting
# "current model" context into the orchestrator prompt so the model sees its
# own official product name ("Claude Opus 4.7") rather than the internal API
# slug ("claude-opus-4-7"). Match by prefix: pinned date suffixes (e.g.
# "-20251001") share the display name of the base family.
_MODEL_DISPLAY_NAMES = {
    # Anthropic
    "claude-opus-4-7":   "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5":  "Claude Haiku 4.5",
    # OpenAI
    "gpt-5.5":      "GPT-5.5 Pro",
    "gpt-5.5-mini": "GPT-5.5 mini",
    "gpt-5.5-nano": "GPT-5.5 nano",
    # Local (Qwen family — common Ollama/llama.cpp tags)
    "qwen3.5-9b":   "Qwen 3.5 9B",
    "qwen3.6-9b":   "Qwen 3.6 9B",
    "qwen3.5":      "Qwen 3.5",
    "qwen3.6":      "Qwen 3.6",
}


def _display_name_for_model_id(model_id: str) -> str:
    """Return the human-readable product name for an API model ID.

    Resolves pinned-date variants ("claude-haiku-4-5-20251001") by prefix match
    against the base family. Falls back to the raw ID if no entry is found so
    new models gracefully surface their slug until a display name is registered.
    """
    if not model_id:
        return ""
    if model_id in _MODEL_DISPLAY_NAMES:
        return _MODEL_DISPLAY_NAMES[model_id]
    for base, name in _MODEL_DISPLAY_NAMES.items():
        if model_id.startswith(base + "-") or model_id.startswith(base + "."):
            return name
    return model_id

# Tier → provider-specific model mapping
_TIER_MAP = {
    "claude": {"high": "opus",   "medium": "sonnet", "low": "haiku"},
    "openai": {"high": "gpt54",  "medium": "gpt54mini", "low": "gpt54nano"},
    "local":  {"high": "local",  "medium": "local",  "low": "local"},
}

# Local LLM model map (single model — all tiers resolve to the same)
_LOCAL_MODEL_MAP = {"local": None}  # resolved at runtime via llm_client

def _tier_to_display(tier: str) -> str:
    """Return a display label showing tier + actual model for current provider."""
    provider = _config.get("provider", "claude")
    tier_map = _TIER_MAP.get(provider, _TIER_MAP["claude"])
    alias = tier_map.get(tier, tier)
    if provider == "openai":
        model_name = _OPENAI_MODEL_MAP.get(alias, alias)
    elif provider == "local":
        from llm.client import MOON_MODEL
        model_name = MOON_MODEL
    else:
        model_name = alias
    return f"{tier} ({model_name})"


async def _resolve_model(alias: str, fallback: str) -> str:
    """Resolve a Claude model alias to its actual ID via the Models API (non-blocking)."""
    try:
        resolved = await asyncio.to_thread(
            lambda: anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).models.retrieve(model_id=alias).id
        )
        logger.info("Resolved model %s => %s", alias, resolved)
        return resolved
    except Exception as e:
        logger.warning("Model resolve failed for %s, using fallback %s: %s", alias, fallback, e)
        return fallback


# Lazy model resolution cache — maps alias → resolved ID
_resolved_models: dict[str, str] = {}


async def _get_model_by_alias(alias: str) -> str:
    """Resolve a model short name (haiku/sonnet/opus) to its full ID, with caching."""
    if alias in _resolved_models:
        return _resolved_models[alias]
    model_alias, fallback = _MODEL_ALIAS_MAP.get(alias, ("claude-sonnet-4-6", "claude-sonnet-4-6"))
    resolved = await _resolve_model(model_alias, fallback)
    _resolved_models[alias] = resolved
    return resolved


def _resolve_openai_model(alias: str) -> str:
    """Resolve OpenAI model alias (gpt54/gpt54mini/gpt54nano) to actual model ID."""
    return _OPENAI_MODEL_MAP.get(alias, alias)


def _resolve_tier(tier: str, provider: str | None = None) -> str:
    """Resolve a tier name (high/medium/low) to provider-specific model alias.

    ``provider`` override lets callers (e.g. agents with a pinned provider)
    resolve against a different tier map than the runtime config. Defaults
    to whatever ``_config["provider"]`` currently says.
    """
    if provider is None:
        provider = _config.get("provider", "claude")
    tier_map = _TIER_MAP.get(provider, _TIER_MAP["claude"])
    return tier_map.get(tier, tier)  # passthrough if not a tier name


def is_autonomous_active() -> bool:
    """True when the hourly autonomous project loop should run.

    Reads from disk each call so systemd-invoked one-shots (separate Python
    process) see toggles made by the telegram bot without restarts.
    """
    return bool(_load_config().get("autonomous_active", True))


def set_autonomous_active(active: bool) -> bool:
    """Flip the autonomous toggle and persist. Returns the new state."""
    _config["autonomous_active"] = bool(active)
    _save_config()
    return _config["autonomous_active"]


def get_current_model_selection(
    kind: str = "chat",
    provider_override: str | None = None,
) -> dict:
    """Return runtime-selected provider/model metadata for chat or task path.

    ``provider_override`` lets callers that know they are running against a
    specific provider (e.g. agents with spec.provider pinned to "claude" even
    when config.provider="openai") resolve the tier against THAT provider's
    model map, so the surfaced model_id/display_name match what actually runs.
    Without the override, falls back to the runtime config's provider.
    """
    kind = "task" if kind == "task" else "chat"
    provider = provider_override or _config.get("provider", "claude")
    tier_key = "task_model" if kind == "task" else "chat_model"
    tier = str(_config.get(tier_key, "high"))
    alias = _resolve_tier(tier, provider=provider)
    if provider == "local":
        from llm.client import MOON_MODEL
        model_id = MOON_MODEL
    elif provider == "openai" or alias in _OPENAI_MODEL_MAP:
        model_id = _resolve_openai_model(alias)
    else:
        model_alias, fallback = _MODEL_ALIAS_MAP.get(alias, (alias, alias))
        model_id = _resolved_models.get(alias, fallback)
    return {
        "kind": kind,
        "provider": provider,
        "tier": tier,
        "alias": alias,
        "model_id": model_id,
        "display_name": _display_name_for_model_id(model_id),
        "resolved": True if provider in ("openai", "local") else alias in _resolved_models,
    }


async def _get_model() -> str:
    """Get the current chat model based on runtime config."""
    if _config.get("provider") == "local":
        from llm.client import _resolve_backend
        return _resolve_backend()["model"]
    alias = _resolve_tier(_config["chat_model"])
    if _config.get("provider") == "openai" or alias in _OPENAI_MODEL_MAP:
        return _resolve_openai_model(alias)
    return await _get_model_by_alias(alias)


async def _get_model_task() -> str:
    """Get the current task model based on runtime config."""
    if _config.get("provider") == "local":
        from llm.client import _resolve_backend
        return _resolve_backend()["model"]
    alias = _resolve_tier(_config["task_model"])
    if _config.get("provider") == "openai" or alias in _OPENAI_MODEL_MAP:
        return _resolve_openai_model(alias)
    return await _get_model_by_alias(alias)


async def _get_model_light() -> str:
    """Get the light model (Haiku) — used for compression, reflection, etc."""
    return await _get_model_by_alias("haiku")


async def _get_model_moon() -> str:
    """Get the Moon (local LLM) model name — async wrapper for await compatibility."""
    from llm.client import MOON_MODEL
    return MOON_MODEL


def _extract_text(response) -> str:
    """Safely extract text from Claude API response, handling empty content."""
    if response.content:
        return response.content[0].text
    return ""
