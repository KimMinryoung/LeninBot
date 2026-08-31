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


def _per_token(
    input_price: float,
    output_price: float,
    cached: float,
    cache_write: float | None = None,
) -> dict[str, float]:
    return {
        "input": input_price / 1_000_000,
        "output": output_price / 1_000_000,
        "cached_input": cached / 1_000_000,
        "cache_write": (input_price if cache_write is None else cache_write) / 1_000_000,
    }


# ── DeepSeek V4 tiered pricing (effective 2026-08-16 16:00 UTC) ──────
# DeepSeek replaced its flat V4 rate with peak/off-peak tiers at 00:00 Beijing
# on 2026-08-17, i.e. 16:00 UTC on 2026-08-16. Peak windows are 01:00–04:00 and
# 06:00–10:00 UTC (09:00–12:00 and 14:00–18:00 Beijing, UTC+8); every other hour
# is off-peak at exactly half the peak rate (computed here, never stored twice).
# Source: DeepSeek pricing announcement 2026-08-13. The published USD list is
# the RMB table over 6.8182 RMB/USD (27.00元 output = $3.96, 9.00元 = $1.32); the
# cache-hit USD below is that same conversion of 0.10元 / 0.30元. Pre-cutover the
# flat rates in _DEEPSEEK_FLAT apply. Triple order: (cache-miss input, output,
# cache-hit input), USD per 1M tokens.
DEEPSEEK_TIERED_START = datetime(2026, 8, 16, 16, 0, tzinfo=timezone.utc)
_DEEPSEEK_FLAT = {
    "deepseek-v4-flash": (0.14, 0.28, 0.0028),
    "deepseek-v4-pro": (0.435, 0.87, 0.003625),
}
_DEEPSEEK_PEAK = {
    "deepseek-v4-flash": (0.44, 1.32, 0.014667),
    "deepseek-v4-pro": (1.32, 3.96, 0.044),
}


def _deepseek_key(model: str) -> str | None:
    if model in _DEEPSEEK_FLAT:
        return model
    for base in _DEEPSEEK_FLAT:
        if model.startswith(base + "-") or model.startswith(base + "."):
            return base
    return None


def deepseek_price_triple(
    model: str, now: datetime | None = None
) -> tuple[float, float, float] | None:
    """(cache-miss input, output, cache-hit input) USD **per 1M tokens** for a
    DeepSeek model at ``now`` (UTC, defaults to current time), or None if the
    model is not DeepSeek. Flat until the 2026-08-16 16:00 UTC cutover, then
    peak/off-peak; the caller divides by 1_000_000 for a per-token rate."""
    key = _deepseek_key(model)
    if key is None:
        return None
    now = now or datetime.now(timezone.utc)
    if now < DEEPSEEK_TIERED_START:
        return _DEEPSEEK_FLAT[key]
    miss, out, hit = _DEEPSEEK_PEAK[key]
    hour = now.astimezone(timezone.utc).hour
    if (1 <= hour < 4) or (6 <= hour < 10):
        return miss, out, hit
    return miss / 2, out / 2, hit / 2


OPENAI_COMPATIBLE_PRICING = {
    # OpenAI standard short-context rates, audited 2026-08-29. GPT-5.6 cache
    # writes cost 1.25x ordinary input. Sol's $4/$20 promotional price lasts
    # at least through 2026-11-21.
    "gpt-5.6-sol": _per_token(4.00, 20.00, 0.40, 5.00),
    "gpt-5.6-terra": _per_token(2.00, 12.00, 0.20, 2.50),
    "gpt-5.6-luna": _per_token(0.20, 1.20, 0.02, 0.25),
    # DeepSeek is intentionally NOT a static row: its price is time-of-day
    # dependent since 2026-08-16 and is resolved via deepseek_price_triple().
    "kimi-k3": _per_token(3.00, 15.00, 0.30),
}

GPT56_LONG_CONTEXT_THRESHOLD = 272_000
_OPENAI_GPT56_LONG_CONTEXT_PRICING = {
    "gpt-5.6-sol": _per_token(8.00, 30.00, 0.80, 10.00),
    "gpt-5.6-terra": _per_token(4.00, 18.00, 0.40, 5.00),
    "gpt-5.6-luna": _per_token(0.40, 1.80, 0.04, 0.50),
}


# Google AI Gemini Developer API standard-tier pricing (USD/token).  Keep
# this separate from the OpenAI-compatible table: Gemini reports prompt and
# cached prompt tokens with OpenAI-like semantics, but has its own models and
# prices.  See dev_docs/llm_gateway.md for the pricing-source link/date.
GEMINI_PRICING = {
    # 2026-08-31 갱신: 현행 라인업은 pro=3.1-pro-preview(stable 별칭 없음,
    # 3.5 Pro는 존재하지 않음), flash=3.7, flash-lite=3.5로 고정한다(사용자
    # 결정 — 3.5/3.6 flash는 선택지로 두지 않음). >200K 입력의 $4/$18 장문
    # 티어는 미모델링. 3.7 Flash는 출시가 $0.75/$3.75, 2027-01-01부터
    # $1.50/$7.50 예정. 캐시 입력가는 관례(입력의 10%) — 3.1 Pro만 공식 $0.20.
    # 2.5/3.1 구모델 행은 라이브 call site들이 아직 쓰므로 가격 산정용으로만 유지.
    "gemini-3.1-pro-preview": _per_token(2.00, 12.00, 0.20),
    "gemini-3.7-flash": _per_token(0.75, 3.75, 0.075),
    "gemini-3.5-flash-lite": _per_token(0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": _per_token(0.25, 1.50, 0.025),
    "gemini-2.5-flash-lite": _per_token(0.10, 0.40, 0.01),
    "gemini-2.5-flash": _per_token(0.30, 2.50, 0.03),
    "gemini-embedding-001": _per_token(0.15, 0.0, 0.15),
}


def openai_compatible_pricing(
    model: str, now: datetime | None = None, *, input_tokens: int = 0,
) -> dict[str, float]:
    """Return pricing for an exact or provider-pinned model ID.

    DeepSeek rows are time-of-day dependent (peak/off-peak) and resolved live;
    GPT-5.6 uses the full-request long-context tier above 272K input tokens;
    everything else is a static row with a Terra fallback for unknowns."""
    triple = deepseek_price_triple(model, now)
    if triple is not None:
        miss, out, hit = triple
        return {
            "input": miss / 1_000_000,
            "output": out / 1_000_000,
            "cached_input": hit / 1_000_000,
        }
    resolved_key = None
    if model in OPENAI_COMPATIBLE_PRICING:
        resolved_key = model
    else:
        for base in OPENAI_COMPATIBLE_PRICING:
            if model.startswith(base + "-") or model.startswith(base + "."):
                resolved_key = base
                break
    if resolved_key is not None:
        if (
            input_tokens > GPT56_LONG_CONTEXT_THRESHOLD
            and resolved_key in _OPENAI_GPT56_LONG_CONTEXT_PRICING
        ):
            return _OPENAI_GPT56_LONG_CONTEXT_PRICING[resolved_key]
        return OPENAI_COMPATIBLE_PRICING[resolved_key]
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


def anthropic_pricing_table(
    now: datetime | date | None = None,
) -> dict[str, dict[str, float]]:
    """Return current Messages pricing, including Sonnet 5 launch pricing and
    the DeepSeek V4 peak/off-peak tiers (from 2026-08-16 16:00 UTC).

    Accepts a datetime (its UTC hour picks the DeepSeek peak/off-peak tier) or a
    bare date (treated as 00:00 UTC — fine for the Sonnet cutover, which is
    date-only). No argument means now."""
    if now is None:
        now = datetime.now(timezone.utc)
    current = now.date() if isinstance(now, datetime) else now
    now_dt = (
        now if isinstance(now, datetime)
        else datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    )
    sonnet = (
        _anthropic_row(2.00, 10.00, 0.20)
        if current <= date(2026, 8, 31)
        else _anthropic_row(3.00, 15.00, 0.30)
    )

    def _deepseek(model: str) -> dict[str, float]:
        miss, out, hit = deepseek_price_triple(model, now_dt)
        # Anthropic semantics: cache_creation is priced as ordinary (cache-miss)
        # input, cache_read as the cache-hit rate.
        return {
            "input": miss / 1_000_000,
            "output": out / 1_000_000,
            "cache_creation": miss / 1_000_000,
            "cache_read": hit / 1_000_000,
        }

    return {
        "claude-fable-5": _anthropic_row(10.00, 50.00, 1.00),
        "claude-opus-5": _anthropic_row(5.00, 25.00, 0.50),
        "claude-sonnet-5": sonnet,
        "claude-haiku-4-5": _anthropic_row(1.00, 5.00, 0.10),
        "deepseek-v4-flash": _deepseek("deepseek-v4-flash"),
        "deepseek-v4-pro": _deepseek("deepseek-v4-pro"),
        "kimi-k3": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
            "cache_creation": 3.00 / 1_000_000,
            "cache_read": 0.30 / 1_000_000,
        },
    }


def kimi_openai_tool_options() -> dict:
    """Return the shared Kimi K3 OpenAI-compatible tool-loop contract.

    (콘텐츠필터 시 DeepSeek으로 요청 단위 스위칭하던 fallback 계약은
    2026-08-04 제거 — Kimi 미사용 상태에서 루프 복잡도만 키우고 있었다.)
    """
    return {
        "extra_body": {"reasoning_effort": "max"},
        "sdk_max_token_param": "max_tokens",
        "include_parallel_tool_calls": False,
        "preserve_reasoning_content": True,
    }
