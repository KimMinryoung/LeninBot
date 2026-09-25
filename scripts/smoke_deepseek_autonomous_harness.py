#!/usr/bin/env python3
"""Smoke checks for DeepSeek agent harness routing (tool loops on the OpenAI-compatible endpoint)."""

from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.claude_loop import _calculate_cost, _content_block_for_replay, _pricing_for
from bot_config import (
    _get_deepseek_browser_params,
    _get_deepseek_thinking_params,
    _get_deepseek_tool_thinking_params,
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_deepseek_clients_are_configured() -> None:
    source = _read("bot_config.py")
    assert "DEEPSEEK_ANTHROPIC_BASE_URL" in source
    # Wrapped in the audit layer since 2026-08-09 (a5323cd) so every importer
    # of the shared client is metered.
    assert "_deepseek_anthropic_client = AuditedAsyncAnthropic(" in source
    assert "_deepseek_client = AuditedAsyncOpenAI(" in source
    assert "https://api.deepseek.com/anthropic" in source


def test_tool_loops_route_through_deepseek_tool_loop() -> None:
    # Since 2026-09-25 every DeepSeek tool loop takes its loop, client and
    # options from bot_config.deepseek_tool_loop (OpenAI-compatible endpoint
    # by default, DEEPSEEK_TOOL_LOOP_PROTOCOL=anthropic to revert).
    import os
    import bot_config
    os.environ.pop("DEEPSEEK_TOOL_LOOP_PROTOCOL", None)
    assert bot_config.deepseek_tool_protocol() == "openai"
    for path in ("telegram/chat_runtime.py", "services/a2a_handler.py", "services/web_chat.py",
                 "browser/worker.py", "roleplay/bot.py", "writer/models.py", "bot_config.py"):
        source = _read(path)
        assert "deepseek_tool_loop" in source or "deepseek_tool_available" in source, path
        assert "client=_deepseek_anthropic_client" not in source, path
    assert "deepseek_thinking_override or resolve_inference_extra(" in _read("telegram/chat_runtime.py")
    assert "_get_deepseek_tool_thinking_params" in _read("services/a2a_handler.py")


def test_browser_use_deepseek_routes_to_anthropic_harness() -> None:
    source = _read("browser/use_agent.py")
    assert "class _DeepSeekAnthropicBrowserChat(_AuditedBrowserChatMixin, ChatAnthropic)" in source
    assert "DEEPSEEK_ANTHROPIC_BASE_URL" in source
    assert "params.update(_get_deepseek_browser_params())" in source
    assert "_browser_use_vision_enabled" in source
    assert "_resolve_vision_fallback" in source
    assert "ChatDeepSeek" not in source


def test_webchat_deepseek_tool_progress() -> None:
    source = _read("services/web_chat.py")
    assert 'provider == "deepseek"' in source
    assert "content_filter_fallback" not in source
    assert 'event == "tool_call"' in source
    assert '"type": "tool_done" if done else "tool_start"' in source
    assert "on_progress=on_progress" in source
    assert "def _build_web_model_context" in source
    assert "### Current Model" in source
    assert "profile.display_name" in source
    assert "model-id" not in source


def test_deepseek_thinking_config_is_enabled_by_default() -> None:
    params = _get_deepseek_thinking_params()
    assert params["thinking"] == {"type": "enabled"}
    assert params["output_config"]["effort"] in {"high", "max"}


def test_deepseek_tool_thinking_disabled_by_default() -> None:
    import os
    params = _get_deepseek_tool_thinking_params()
    assert params == {"thinking": {"type": "disabled"}}
    os.environ["DEEPSEEK_TOOL_THINKING_MODE"] = "thinking"
    try:
        assert _get_deepseek_tool_thinking_params()["thinking"] == {"type": "enabled"}
    finally:
        del os.environ["DEEPSEEK_TOOL_THINKING_MODE"]


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
    # DeepSeek pricing is time-of-day tiered since 2026-08-16 (36cdb86) and
    # V4.1 since 2026-09-10, so the flat V4 constants are gone; tier math is
    # covered by tests/test_deepseek_pricing.py. Here: the agent loop resolves
    # DeepSeek ids to the current DeepSeek row, not the Claude fallback.
    from llm.provider_registry import deepseek_price_triple

    for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
        miss, out, hit = deepseek_price_triple(model)
        row = _pricing_for(model)
        assert row["input"] == miss / 1_000_000, model
        assert row["output"] == out / 1_000_000, model
        assert row["cache_read"] == hit / 1_000_000, model
        assert row != _pricing_for("claude-sonnet-5"), model

    miss, out, _hit = deepseek_price_triple("deepseek-v4-pro")
    usage = SimpleNamespace(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    assert round(_calculate_cost(usage, "deepseek-v4-pro"), 6) == round(miss + out, 6)

if __name__ == "__main__":
    test_deepseek_clients_are_configured()
    test_tool_loops_route_through_deepseek_tool_loop()
    test_browser_use_deepseek_routes_to_anthropic_harness()
    test_webchat_deepseek_tool_progress()
    test_deepseek_thinking_config_is_enabled_by_default()
    test_deepseek_tool_thinking_disabled_by_default()
    test_thinking_blocks_are_replayed_not_coerced_to_text()
    test_deepseek_pricing_uses_deepseek_rows()
    print("deepseek harness smoke ok")
