#!/usr/bin/env python3
"""Smoke checks for the DeepSeek autonomous Anthropic-compatible harness."""

from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claude_loop import _calculate_cost, _pricing_for


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_deepseek_anthropic_client_is_configured() -> None:
    source = _read("bot_config.py")
    assert "DEEPSEEK_ANTHROPIC_BASE_URL" in source
    assert "_deepseek_anthropic_client = anthropic.AsyncAnthropic" in source
    assert "https://api.deepseek.com/anthropic" in source


def test_autonomous_deepseek_routes_before_openai_compatible_fallback() -> None:
    source = _read("telegram/bot.py")
    anth = 'effective_provider == "deepseek" and _runtime_kind == "autonomous" and _deepseek_anthropic_client'
    openai = 'effective_provider == "deepseek" and _deepseek_client'
    assert anth in source
    assert openai in source
    assert source.index(anth) < source.index(openai)
    assert "client=_deepseek_anthropic_client" in source
    assert 'provider_label="deepseek"' in source


def test_deepseek_pricing_uses_deepseek_rows() -> None:
    pro = _pricing_for("deepseek-v4-pro")
    flash = _pricing_for("deepseek-v4-flash")
    assert pro["input"] == 1.74 / 1_000_000
    assert pro["output"] == 3.48 / 1_000_000
    assert flash["cache_read"] == 0.028 / 1_000_000

    usage = SimpleNamespace(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    assert round(_calculate_cost(usage, "deepseek-v4-pro"), 2) == 5.22


if __name__ == "__main__":
    test_deepseek_anthropic_client_is_configured()
    test_autonomous_deepseek_routes_before_openai_compatible_fallback()
    test_deepseek_pricing_uses_deepseek_rows()
    print("deepseek autonomous harness smoke ok")
