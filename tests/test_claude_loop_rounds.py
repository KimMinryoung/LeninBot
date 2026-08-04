"""Round-loop tests for claude_loop.chat_with_tools with a fake client.

No network, no Redis, no DB: the Anthropic client is a stub returning canned
responses, and execute_tools_batch is patched with an in-memory executor.
These pin the loop's core contracts before any unification refactor:
  * plain text turn returns text and fills budget_tracker
  * tool_use round → tool executed → results paired → final text
  * safety net synthesizes a tool_result for unexecuted tool_use ids
  * budget exhaustion forces the limit path (limit message + final call)
  * terminal_tools short-circuit returns the tool result without extra call

Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import claude_loop
from claude_loop import chat_with_tools


def _usage(inp=100, out=50):
    return SimpleNamespace(
        input_tokens=inp, output_tokens=out,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(tid, name, tinput=None):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=tinput or {})


def _response(blocks, stop_reason="end_turn", usage=None):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason,
        usage=usage or _usage(), stop_details=None,
    )


class FakeClient:
    """client.messages.create(**kwargs) returning canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # kwargs of every create() call

        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if not outer._responses:
                    raise AssertionError("FakeClient ran out of canned responses")
                return outer._responses.pop(0)

        self.messages = _Messages()


def _fake_batch_factory(results_by_name, skip_names=()):
    """Return an execute_tools_batch stand-in resolving from a name→result map."""

    async def _fake_batch(tool_uses, tool_handlers, **kwargs):
        out = []
        for tid, tname, tinput in tool_uses:
            if tname in skip_names:
                continue  # simulate an executor that dropped this call
            result, is_error = results_by_name.get(tname, (f"ran:{tname}", False))
            out.append((tid, tname, tinput, result, is_error))
        return out

    return _fake_batch


TOOLS = [{
    "name": "echo",
    "description": "echo tool",
    "input_schema": {"type": "object", "properties": {}},
}]


class TestPlainTextTurn(unittest.TestCase):
    def test_returns_text_and_tracker(self):
        client = FakeClient([_response([_text_block("최종 답변")])])
        tracker = {}
        result = asyncio.run(chat_with_tools(
            [{"role": "user", "content": "질문"}],
            client=client, model="claude-sonnet-5",
            tools=[], tool_handlers={}, system_prompt="sys",
            budget_usd=1.0, budget_tracker=tracker,
        ))
        self.assertEqual(result, "최종 답변")
        self.assertEqual(tracker["rounds_used"], 1)
        self.assertFalse(tracker["was_interrupted"])
        self.assertGreater(tracker["total_cost"], 0)
        self.assertEqual(len(client.calls), 1)
        # system prompt got the 1h cache_control marker
        sys_blocks = client.calls[0]["system"]
        self.assertEqual(sys_blocks[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})


class TestToolRound(unittest.TestCase):
    def test_tool_executed_then_final_text(self):
        long_pretool = "이 작업을 위해 먼저 echo 도구를 호출해서 결과를 확인하겠다"  # >20 chars → accumulated
        client = FakeClient([
            _response(
                [_text_block(long_pretool), _tool_use_block("t1", "echo", {"q": 1})],
                stop_reason="tool_use",
            ),
            _response([_text_block("완료")]),
        ])
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({"echo": ("echo-result", False)})):
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers={"echo": None}, system_prompt="s",
                budget_usd=5.0,
            ))
        self.assertIn("완료", result)
        self.assertIn(long_pretool, result)  # substantial (>20 char) pre-tool text accumulates

    def test_short_pretool_text_not_accumulated(self):
        client = FakeClient([
            _response([_text_block("짧은 텍스트"), _tool_use_block("t1", "echo")],
                      stop_reason="tool_use"),
            _response([_text_block("완료")]),
        ])
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({"echo": ("r", False)})):
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers={"echo": None}, system_prompt="s",
                budget_usd=5.0,
            ))
        self.assertEqual(result, "완료")  # ≤20-char round text is dropped from the final answer
        # Second call's messages must contain the paired tool_use / tool_result
        msgs = client.calls[1]["messages"]
        tool_use_ids = [
            b["id"] for m in msgs if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        tool_result_ids = [
            b["tool_use_id"] for m in msgs if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        self.assertEqual(tool_use_ids, ["t1"])
        self.assertEqual(tool_result_ids, ["t1"])


class TestSafetyNet(unittest.TestCase):
    def test_missing_result_synthesized_as_error(self):
        client = FakeClient([
            _response([_tool_use_block("t1", "echo")], stop_reason="tool_use"),
            _response([_text_block("done")]),
        ])
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({}, skip_names={"echo"})):
            asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers={"echo": None}, system_prompt="s",
                budget_usd=5.0,
            ))
        msgs = client.calls[1]["messages"]
        results = [
            b for m in msgs if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool_use_id"], "t1")
        self.assertTrue(results[0].get("is_error"))
        self.assertIn("skipped", results[0]["content"])


class TestBudgetExhaustion(unittest.TestCase):
    def test_forced_final_after_budget_break(self):
        # Huge usage → round cost far above the 0.01 budget floor.
        expensive = _usage(inp=10_000_000, out=1_000_000)
        client = FakeClient([
            _response([_tool_use_block("t1", "echo")], stop_reason="tool_use",
                      usage=expensive),
            _response([_text_block("강제 마무리 답변")]),  # forced final
        ])
        tracker = {}
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({"echo": ("r", False)})):
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers={"echo": None}, system_prompt="s",
                budget_usd=0.01, budget_tracker=tracker,
            ))
        self.assertIn("강제 마무리 답변", result)
        self.assertTrue(tracker["was_interrupted"])
        # Forced-final call: no tools, and the injected [SYSTEM] limit message
        final_kwargs = client.calls[1]
        self.assertNotIn("tools", final_kwargs)
        joined = str(final_kwargs["messages"])
        self.assertIn("[SYSTEM] 예산 소진", joined)


class TestTerminalToolShortCircuit(unittest.TestCase):
    def test_terminal_success_returns_without_extra_call(self):
        client = FakeClient([
            _response([_tool_use_block("t1", "finish_report")], stop_reason="tool_use"),
        ])
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({"finish_report": ("보고서 저장 완료", False)})):
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=[{"name": "finish_report", "description": "d",
                        "input_schema": {"type": "object", "properties": {}}}],
                tool_handlers={"finish_report": None}, system_prompt="s",
                budget_usd=5.0, terminal_tools=["finish_report"],
            ))
        self.assertEqual(result, "보고서 저장 완료")
        self.assertEqual(len(client.calls), 1)  # no trailing text round


if __name__ == "__main__":
    unittest.main()
