"""Writer model catalog, persisted selection, and provider resolution."""

from __future__ import annotations

from typing import Any

import anthropic

from secrets_loader import get_secret

from writer.store import get_writer_setting, set_writer_setting
from writer.config import WriterCallPolicy

WRITER_MODEL = "claude-fable-5"
WRITER_MODEL_DISPLAY = "Claude Fable 5"
WRITER_INPUT_PRICE_PER_MTOK = 10.0
WRITER_OUTPUT_PRICE_PER_MTOK = 50.0

# Selectable models. Fable is the default; DeepSeek options route through the
# same chat_with_tools loop via bot_config's Anthropic-compatible DeepSeek
# client. Prices are display hints only — authoritative cost comes from
# claude_loop.PRICING_TABLE (keyed by model id) at runtime.
# "extra" carries provider kwargs passed straight to chat_with_tools.
WRITER_MODEL_CHOICES: dict[str, dict] = {
    "fable": {
        "provider": "anthropic",
        "model": "claude-fable-5",
        "display": "Claude Fable 5",
        "input_price_per_mtok": 10.0,
        "output_price_per_mtok": 50.0,
        # Fable 5 uses adaptive thinking: the model decides when and how much
        # to think, scaled by output_config.effort. High effort buys scene
        # planning (structure, beats, imagery) on turns that warrant it;
        # trivial turns stay cheap because thinking is adaptive, not forced.
        # display="summarized": the default ("omitted") streams NO events
        # while the model thinks — long thinking reads as a dead stream and
        # trips the provider idle watchdog (observed 2026-07-07, 3 retries in
        # a row on one heavy turn). Summarized thinking deltas keep the event
        # stream alive; billing is identical under every display setting.
        "extra": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"},
        },
    },
    "fable_fast": {
        "provider": "anthropic",
        "model": "claude-fable-5",
        "display": "Claude Fable 5 (low effort)",
        "input_price_per_mtok": 10.0,
        "output_price_per_mtok": 50.0,
        "extra": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "low"},
        },
    },
    "deepseek_pro": {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "display": "DeepSeek V4 Pro",
        "input_price_per_mtok": 0.435,
        "output_price_per_mtok": 0.87,
    },
    "deepseek_flash": {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "display": "DeepSeek V4 Flash",
        "input_price_per_mtok": 0.14,
        "output_price_per_mtok": 0.28,
    },
}
WRITER_DEFAULT_CHOICE = "fable"

# Light-agent tiers for delegated subtasks. Easy work runs on cheap DeepSeek
# regardless of the (usually heavy) main writer model: the line-edit critic
# gets the pro tier for craft, web research digestion gets flash. Both fall
# back to the main model when DeepSeek is unconfigured.
WRITER_CRITIC_CHOICE = "deepseek_pro"
WRITER_RESEARCH_CHOICE = "deepseek_flash"

_writer_client: anthropic.AsyncAnthropic | None = None


def _client() -> anthropic.AsyncAnthropic:
    global _writer_client
    if _writer_client is None:
        api_key = get_secret("WRITER_ANTHROPIC_API_KEY", "") or get_secret("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("WRITER_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY is required")
        _writer_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _writer_client


def _deepseek_available() -> bool:
    try:
        import bot_config
        return bot_config._deepseek_anthropic_client is not None
    except Exception:
        return False


def list_writer_models() -> list[dict]:
    """Public metadata for the model picker: keys, display names, prices, availability."""
    out: list[dict] = []
    for key, spec in WRITER_MODEL_CHOICES.items():
        available = True if spec["provider"] == "anthropic" else _deepseek_available()
        out.append({
            "key": key,
            "id": spec["model"],
            "display_name": spec["display"],
            "provider": spec["provider"],
            "input_price_per_mtok": spec["input_price_per_mtok"],
            "output_price_per_mtok": spec["output_price_per_mtok"],
            "available": available,
            "default": key == WRITER_DEFAULT_CHOICE,
        })
    return out


def get_selected_model_choice() -> str:
    """The admin's persisted model choice; falls back to the built-in default
    when unset or when the saved key no longer exists."""
    saved = get_writer_setting("model_choice")
    if isinstance(saved, str) and saved in WRITER_MODEL_CHOICES:
        return saved
    return WRITER_DEFAULT_CHOICE


def set_selected_model_choice(choice: str) -> str:
    key = (choice or "").strip()
    if key not in WRITER_MODEL_CHOICES:
        raise ValueError(f"Unknown writer model choice: {choice!r}")
    set_writer_setting("model_choice", key)
    return key


def resolve_writer_model(choice: str | None) -> tuple[Any, str, str, dict]:
    """Resolve a model choice key to (client, model_id, display_name, extra_kwargs).

    extra_kwargs carries provider-specific chat_with_tools kwargs (e.g. DeepSeek
    thinking/output_config). When no explicit choice is given, the admin's
    persisted selection wins over the built-in default. Raises ValueError for
    an unknown key and RuntimeError if the chosen provider is not configured.
    """
    key = (choice or "").strip() or get_selected_model_choice()
    spec = WRITER_MODEL_CHOICES.get(key)
    if spec is None:
        raise ValueError(f"Unknown writer model choice: {choice!r}")
    if spec["provider"] == "deepseek":
        import bot_config
        client = bot_config._deepseek_anthropic_client
        if client is None:
            raise RuntimeError("DeepSeek is not configured (DEEPSEEK_API_KEY missing).")
        # Thinking-on, effort-high by default (same as other non-web DeepSeek
        # agents). The writer never forces tool_choice, so thinking is stable.
        extra = bot_config._get_deepseek_thinking_params()
        return client, spec["model"], spec["display"], extra
    return _client(), spec["model"], spec["display"], dict(spec.get("extra") or {})


def resolve_writer_call_extra(
    policy: WriterCallPolicy, model: str, default_extra: dict,
) -> dict:
    """Apply centrally declared provider controls for one writer call role."""
    if policy.thinking_policy == "tool_loop" and model.startswith("deepseek"):
        import bot_config
        return bot_config._get_deepseek_tool_thinking_params()
    return dict(default_extra)


def light_effort_extra(model: str, extra: dict) -> dict:
    """Effort for cheap delegated work (critic diagnosis, line edits, research
    digestion) that lands on the Claude writer model: high effort is reserved
    for prose-writing calls (main turn, author revision), so downgrade
    output_config.effort to "low" here. Non-Claude extras (DeepSeek — already
    the cheap tier, params shared with other agents) pass through untouched."""
    if not model.startswith("claude") or "output_config" not in extra:
        return extra
    out = dict(extra)
    out["output_config"] = {**out["output_config"], "effort": "low"}
    return out


def resolve_light_model(
    choice: str, fallback: tuple[Any, str, str, dict]
) -> tuple[Any, str, str, dict]:
    """Resolve a light-agent choice (critic/research delegation), falling back
    to the caller's already-resolved main model when the light provider is
    unavailable — delegation must never break a run. The fallback runs at low
    effort: it inherits the main model, not the main turn's prose-grade depth."""
    try:
        return resolve_writer_model(choice)
    except (ValueError, RuntimeError):
        client, model, display, extra = fallback
        light = light_effort_extra(model, extra)
        if light is extra:
            return fallback
        return client, model, display, light


# Backwards-compatible private alias (older call sites and tests).
_resolve_writer_model = resolve_writer_model
