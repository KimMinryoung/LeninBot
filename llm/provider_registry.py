"""Canonical provider models, tier routing, pricing, and request options."""

from __future__ import annotations

from datetime import date, datetime, timezone


CLAUDE_MODEL_ALIASES = {
    "haiku": ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    "sonnet": ("claude-sonnet-5", "claude-sonnet-5"),
    "opus": ("claude-opus-5", "claude-opus-5"),
}

OPENAI_MODEL_MAP = {
    "gpt56": "gpt-5.6-sol",
    "gpt56terra": "gpt-5.6-terra",
    "gpt56luna": "gpt-5.6-luna",
}

DEEPSEEK_MODEL_MAP = {
    "deepseek_pro": "deepseek-v4-pro",
    "deepseek_flash": "deepseek-v4-flash",
}

KIMI_MODEL_MAP = {"kimi_k3": "kimi-k3"}

TIER_MODEL_KEYS = {
    "claude": {"high": "opus", "medium": "sonnet", "low": "haiku"},
    "openai": {"high": "gpt56", "medium": "gpt56terra", "low": "gpt56luna"},
    "deepseek": {
        "high": "deepseek_pro",
        "medium": "deepseek_flash",
        "low": "deepseek_flash",
    },
    "kimi": {"high": "kimi_k3", "medium": "kimi_k3", "low": "kimi_k3"},
    "local": {"high": "local", "medium": "local", "low": "local"},
}

MODEL_DISPLAY_NAMES = {
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "kimi-k3": "Kimi K3",
    "qwen3.5-9b": "Qwen 3.5 9B",
    "qwen3.6-9b": "Qwen 3.6 9B",
    "qwen3.5": "Qwen 3.5",
    "qwen3.6": "Qwen 3.6",
}


def _per_token(input_price: float, output_price: float, cached: float) -> dict[str, float]:
    return {
        "input": input_price / 1_000_000,
        "output": output_price / 1_000_000,
        "cached_input": cached / 1_000_000,
    }


OPENAI_COMPATIBLE_PRICING = {
    "gpt-5.6-sol": _per_token(5.00, 30.00, 0.50),
    "gpt-5.6-terra": _per_token(2.50, 15.00, 0.25),
    "gpt-5.6-luna": _per_token(1.00, 6.00, 0.10),
    "deepseek-v4-flash": _per_token(0.14, 0.28, 0.0028),
    "deepseek-v4-pro": _per_token(0.435, 0.87, 0.003625),
    "kimi-k3": _per_token(3.00, 15.00, 0.30),
}


def openai_compatible_pricing(model: str) -> dict[str, float]:
    """Return pricing for an exact or provider-pinned model ID."""
    if model in OPENAI_COMPATIBLE_PRICING:
        return OPENAI_COMPATIBLE_PRICING[model]
    for base, price in OPENAI_COMPATIBLE_PRICING.items():
        if model.startswith(base + "-") or model.startswith(base + "."):
            return price
    return OPENAI_COMPATIBLE_PRICING["gpt-5.6-terra"]


def _anthropic_row(
    input_price: float,
    output_price: float,
    cache_read: float,
) -> dict[str, float]:
    return {
        "input": input_price / 1_000_000,
        "output": output_price / 1_000_000,
        "cache_creation": (input_price * 2.0) / 1_000_000,
        "cache_read": cache_read / 1_000_000,
    }


def anthropic_pricing_table(today: date | None = None) -> dict[str, dict[str, float]]:
    """Return current Messages pricing, including Sonnet 5 launch pricing."""
    current = today or datetime.now(timezone.utc).date()
    sonnet = (
        _anthropic_row(2.00, 10.00, 0.20)
        if current <= date(2026, 8, 31)
        else _anthropic_row(3.00, 15.00, 0.30)
    )
    return {
        "claude-fable-5": _anthropic_row(10.00, 50.00, 1.00),
        "claude-opus-5": _anthropic_row(5.00, 25.00, 0.50),
        "claude-sonnet-5": sonnet,
        "claude-haiku-4-5": _anthropic_row(1.00, 5.00, 0.10),
        "deepseek-v4-flash": {
            "input": 0.14 / 1_000_000,
            "output": 0.28 / 1_000_000,
            "cache_creation": 0.14 / 1_000_000,
            "cache_read": 0.0028 / 1_000_000,
        },
        "deepseek-v4-pro": {
            "input": 0.435 / 1_000_000,
            "output": 0.87 / 1_000_000,
            "cache_creation": 0.435 / 1_000_000,
            "cache_read": 0.003625 / 1_000_000,
        },
        "kimi-k3": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
            "cache_creation": 3.00 / 1_000_000,
            "cache_read": 0.30 / 1_000_000,
        },
    }


def kimi_openai_tool_options(
    *,
    fallback_client=None,
    fallback_model: str | None = None,
) -> dict:
    """Return the shared Kimi K3 OpenAI-compatible tool-loop contract."""
    return {
        "content_filter_fallback_client": fallback_client,
        "content_filter_fallback_model": fallback_model,
        "content_filter_fallback_label": "deepseek",
        "extra_body": {"reasoning_effort": "max"},
        "sdk_max_token_param": "max_tokens",
        "include_parallel_tool_calls": False,
        "preserve_reasoning_content": True,
    }
