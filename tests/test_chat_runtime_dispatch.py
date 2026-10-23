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
