#!/usr/bin/env python3
"""Smoke checks for DeepSeek Anthropic-compatible agent harness routing."""

from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claude_loop import _calculate_cost, _content_block_for_replay, _pricing_for
from bot_config import _get_deepseek_thinking_params


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_deepseek_anthropic_client_is_configured() -> None:
    source = _read("bot_config.py")
    assert "DEEPSEEK_ANTHROPIC_BASE_URL" in source
    assert "_deepseek_anthropic_client = anthropic.AsyncAnthropic" in source
    assert "https://api.deepseek.com/anthropic" in source


def test_telegram_deepseek_routes_to_anthropic_harness() -> None:
    source = _read("telegram/bot.py")
    anth = 'effective_provider == "deepseek" and _deepseek_anthropic_client'
    assert anth in source
    assert "client=_deepseek_anthropic_client" in source
    assert "_get_deepseek_thinking_params" in source
    assert "output_config=deepseek_thinking.get" in source
    assert 'provider_label="deepseek"' not in source


def test_a2a_deepseek_routes_to_anthropic_harness() -> None:
    source = _read("a2a_handler.py")
    assert 'provider == "deepseek" and _deepseek_anthropic_client' in source
    assert "client=_deepseek_anthropic_client" in source
    assert "_get_deepseek_thinking_params" in source
    assert "output_config=deepseek_thinking.get" in source
    assert 'provider_label="deepseek:a2a"' not in source


def test_browser_worker_deepseek_routes_to_anthropic_harness() -> None:
    source = _read("browser/worker.py")
    assert "DEEPSEEK_ANTHROPIC_BASE_URL" in source
    assert "anthropic.AsyncAnthropic" in source
    assert 'if provider == "deepseek":' in source
    assert "_get_deepseek_thinking_params" in source
    assert "output_config=deepseek_thinking.get" in source
    assert "from openai_tool_loop import chat_with_tools as openai_chat" in source


def test_browser_use_deepseek_routes_to_anthropic_harness() -> None:
    source = _read("browser/use_agent.py")
    assert "class _DeepSeekAnthropicBrowserChat(ChatAnthropic)" in source
    assert "DEEPSEEK_ANTHROPIC_BASE_URL" in source
    assert "params.update(_get_deepseek_thinking_params())" in source
    assert "ChatDeepSeek" not in source


def test_webchat_deepseek_routes_to_anthropic_harness_with_tool_progress() -> None:
    source = _read("web_chat.py")
    assert "_deepseek_anthropic_client" in source
    assert 'provider == "deepseek"' in source
    assert "client=_deepseek_anthropic_client" in source
    assert 'thinking={"type": "disabled"}' in source
    assert "_deepseek_client" not in source
    assert 'event == "tool_call"' in source
    assert '"type": "tool_done" if done else "tool_start"' in source
    assert "on_progress=on_progress" in source


def test_deepseek_thinking_config_is_enabled_by_default() -> None:
    params = _get_deepseek_thinking_params()
    assert params["thinking"] == {"type": "enabled"}
    assert params["output_config"]["effort"] in {"high", "max"}


def test_thinking_blocks_are_replayed_not_coerced_to_text() -> None:
    thinking_block = {
        "type": "thinking",
        "thinking": "provider-private reasoning payload",
        "signature": "sig",
    }
    redacted_block = {"type": "redacted_thinking", "data": "opaque"}
    assert _content_block_for_replay(thinking_block) == thinking_block
    assert _content_block_for_replay(redacted_block) == redacted_block


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
    test_telegram_deepseek_routes_to_anthropic_harness()
    test_a2a_deepseek_routes_to_anthropic_harness()
    test_browser_worker_deepseek_routes_to_anthropic_harness()
    test_browser_use_deepseek_routes_to_anthropic_harness()
    test_webchat_deepseek_routes_to_anthropic_harness_with_tool_progress()
    test_deepseek_thinking_config_is_enabled_by_default()
    test_thinking_blocks_are_replayed_not_coerced_to_text()
    test_deepseek_pricing_uses_deepseek_rows()
    print("deepseek harness smoke ok")
