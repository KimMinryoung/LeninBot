import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from telegram import chat_runtime


@pytest.mark.asyncio
async def test_claude_branch_calls_the_claude_tool_loop():
    # The 2026-09-25 split left the Claude branch calling this module's own
    # chat_with_tools wrapper, which rejects `client=` (TypeError on every
    # Claude chat/task). The branch must reach llm.claude_loop.
    loop = AsyncMock(return_value="ok")
    with patch("llm.claude_loop.chat_with_tools", loop), \
         patch("tool_gateway.inference.resolve_inference_extra", return_value={}), \
         patch("tool_gateway.security.caller_scope", lambda _ctx: contextlib.nullcontext()):
        out = await chat_runtime._dispatch_chat_provider(
            [{"role": "user", "content": "hi"}],
            effective_provider="claude",
            profile=SimpleNamespace(model_id="claude-sonnet-5-5"),
            loop_kwargs={"max_tokens": 4096},
            call_inference_policy=None,
            is_orchestrator=True,
            runtime_kind="chat",
            deepseek_thinking_override=None,
            budget_tracker=None,
            on_progress=None,
            gw_ctx=None,
        )
    assert out == "ok"
    kwargs = loop.await_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert "client" in kwargs



def test_claude_wire_tools_drop_top_level_combinators_but_validation_keeps_them():
    # The Messages API 400s on a top-level anyOf (knowledge_graph_search's
    # query-or-entity rule); local validation must still enforce it.
    from llm.claude_loop import _ClaudeProtocolAdapter
    from runtime_tools.registry import TOOLS

    adapter = _ClaudeProtocolAdapter(
        client=None, model="claude-sonnet-5-5", tools=TOOLS, tool_handlers={},
        system_prompt="s", max_tokens=1024, max_input_tokens=None,
        recover_input_via_tools=False, on_progress=None, log_event=None,
        thinking=None, output_config=None, provider_idle_timeout_sec=None,
    )
    for tool in adapter.cached_tools:
        assert not {"anyOf", "oneOf", "allOf"} & set(tool["input_schema"]), tool["name"]
    kg = next(t for t in adapter.tool_definitions if t["name"] == "knowledge_graph_search")
    assert "anyOf" in kg["input_schema"]
