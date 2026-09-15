"""Keep tool commentary out of task results without losing continued answers."""
import asyncio
from unittest.mock import patch

import llm.claude_loop as claude_loop
from test_agent_loop_engine import (
    FakeAnthropicClient, _response, _text_block, _tool_use_block,
    _fake_batch_factory, TOOLS, HANDLERS,
)


def run(responses, **kwargs):
    tracker = {}
    client = FakeAnthropicClient(responses)
    with patch.object(claude_loop, "execute_tools_batch", _fake_batch_factory({"echo": ("source text", False)})):
        text = asyncio.run(claude_loop.chat_with_tools(
            [{"role": "user", "content": "Write report"}], client=client,
            model="claude-sonnet-5", tools=TOOLS, tool_handlers=HANDLERS,
            system_prompt="s", budget_tracker=tracker, **kwargs,
        ))
    return text, tracker


def test_final_response_excludes_tool_commentary():
    progress = "I will now independently verify the source before writing."
    text, tracker = run([
        _response([_text_block(progress), _tool_use_block("t1", "echo")], stop_reason="tool_use"),
        _response([_text_block("완료: 보고서를 저장했다.")]),
    ])
    assert progress in text  # Legacy chat behavior is preserved.
    assert tracker["final_response"] == "완료: 보고서를 저장했다."
    assert tracker["progress_text"] == progress


def test_continuation_is_part_of_final_not_progress():
    progress = "First I will retrieve evidence for the report and compare sources."
    _, tracker = run([
        _response([_text_block(progress), _tool_use_block("t1", "echo")], stop_reason="tool_use"),
        _response([_text_block("보고서 전반부")], stop_reason="max_tokens"),
        _response([_text_block("보고서 후반부")]),
    ], continue_on_length=True, max_length_continuations=1)
    assert tracker["final_response"] == "보고서 전반부\n보고서 후반부"
    assert tracker["progress_text"] == progress


def test_terminal_tool_receipt_is_final():
    _, tracker = run([
        _response([_text_block("I am about to perform the requested final action."), _tool_use_block("t1", "echo")], stop_reason="tool_use"),
    ], terminal_tools=["echo"])
    assert tracker["final_response"] == "source text"
