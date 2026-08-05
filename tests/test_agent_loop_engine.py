"""Engine-path tests for agent_loop.run_tool_loop through both adapters.

The claude/openai round tests pin each loop's basic contracts; these pin the
control-flow paths that were unified into agent_loop and were previously
untested: finalization tools on the forced-final call, the followup-skip
heuristic, and max-token length continuation.

Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import claude_loop
import openai_tool_loop


# ── Anthropic-shaped fakes (mirrors test_claude_loop_rounds) ─────────

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


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if not outer._responses:
                    raise AssertionError("FakeAnthropicClient ran out of responses")
                return outer._responses.pop(0)

        self.messages = _Messages()


# ── OpenAI-shaped fakes (mirrors test_openai_loop_rounds) ────────────

def _oai_usage(prompt=1000, completion=100):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0,
        prompt_tokens_details=None,
    )


def _tc(tc_id, name, arguments="{}"):
    return SimpleNamespace(
        id=tc_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _oai_resp(content, finish="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        reasoning_content="", refusal=None,
    )
    return SimpleNamespace(
        model="deepseek-chat",
        choices=[SimpleNamespace(finish_reason=finish, message=msg)],
        usage=usage or _oai_usage(),
    )


class FakeSDKClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if not outer._responses:
                    raise AssertionError("FakeSDKClient ran out of responses")
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=_Completions())


def _fake_batch_factory(results_by_name):
    async def _fake_batch(tool_uses, tool_handlers, **kwargs):
        out = []
        for tid, tname, tinput in tool_uses:
            result, is_error = results_by_name.get(tname, (f"ran:{tname}", False))
            out.append((tid, tname, tinput, result, is_error))
        return out

    return _fake_batch


TOOLS = [
    {"name": "echo", "description": "echo tool",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "save_diary", "description": "persist the diary",
     "input_schema": {"type": "object", "properties": {}}},
]
HANDLERS = {"echo": None, "save_diary": None}
EXPENSIVE = _usage(inp=10_000_000, out=1_000_000)


class TestClaudeFinalizationTools(unittest.TestCase):
    def test_forced_final_exposes_only_finalization_tools_then_followup(self):
        client = FakeAnthropicClient([
            # Round 1: tool call whose cost blows the budget.
            _response([_tool_use_block("t1", "echo")], stop_reason="tool_use",
                      usage=EXPENSIVE),
            # Forced final: short text + finalization call → followup happens.
            _response([_text_block("짧은 마무리"), _tool_use_block("t2", "save_diary")],
                      stop_reason="tool_use"),
            # Followup: closing text.
            _response([_text_block("일기 저장을 마쳤다")]),
        ])
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({"save_diary": ("saved", False),
                                               "echo": ("r", False)})):
            result = asyncio.run(claude_loop.chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers=HANDLERS, system_prompt="s",
                budget_usd=0.01, finalization_tools=["save_diary"],
            ))
        self.assertEqual(len(client.calls), 3)
        # Forced-final call exposes exactly the finalization tool.
        final_tools = client.calls[1].get("tools")
        self.assertEqual([t["name"] for t in final_tools], ["save_diary"])
        # Followup is tool-less and output-capped.
        self.assertNotIn("tools", client.calls[2])
        self.assertEqual(client.calls[2]["max_tokens"], 2048)
        self.assertIn("일기 저장을 마쳤다", result)

    def test_followup_skipped_when_pretool_text_substantive(self):
        long_text = "충분히 긴 마무리 자기평가 텍스트다. " * 20  # ≥200 chars
        client = FakeAnthropicClient([
            _response([_tool_use_block("t1", "echo")], stop_reason="tool_use",
                      usage=EXPENSIVE),
            _response([_text_block(long_text), _tool_use_block("t2", "save_diary")],
                      stop_reason="tool_use"),
        ])
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({"save_diary": ("saved", False),
                                               "echo": ("r", False)})):
            result = asyncio.run(claude_loop.chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers=HANDLERS, system_prompt="s",
                budget_usd=0.01, finalization_tools=["save_diary"],
            ))
        self.assertEqual(len(client.calls), 2)  # no followup roundtrip
        self.assertIn(long_text.strip(), result)

    def test_malformed_forced_final_input_is_not_executed(self):
        executed = []

        async def fake_batch(tool_uses, tool_handlers, **kwargs):
            executed.extend(name for _tid, name, _input in tool_uses)
            return [(tid, name, inp, "ok", False) for tid, name, inp in tool_uses]

        client = FakeAnthropicClient([
            _response([_tool_use_block("t1", "echo")], stop_reason="tool_use",
                      usage=EXPENSIVE),
            _response([_tool_use_block("t2", "save_diary", ["not", "object"])],
                      stop_reason="tool_use"),
            _response([_text_block("malformed 호출은 실행하지 않았다")]),
        ])
        with patch.object(claude_loop, "execute_tools_batch", fake_batch):
            result = asyncio.run(claude_loop.chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers=HANDLERS, system_prompt="s",
                budget_usd=0.01, finalization_tools=["save_diary"],
            ))
        self.assertEqual(executed, ["echo"])
        self.assertIn("tool input must be an object", str(client.calls[2]["messages"]))
        self.assertIn("실행하지 않았다", result)


class TestClaudeLengthContinuation(unittest.TestCase):
    def test_partial_text_stitched_across_continuation(self):
        client = FakeAnthropicClient([
            _response([_text_block("첫 부분이 여기서 끊겼다")], stop_reason="max_tokens"),
            _response([_text_block("이어지는 마지막 부분")]),
        ])
        result = asyncio.run(claude_loop.chat_with_tools(
            [{"role": "user", "content": "q"}],
            client=client, model="claude-sonnet-5",
            tools=[], tool_handlers={}, system_prompt="s",
            budget_usd=5.0, continue_on_length=True, max_length_continuations=1,
        ))
        self.assertEqual(len(client.calls), 2)
        self.assertIn("첫 부분이 여기서 끊겼다", result)
        self.assertIn("이어지는 마지막 부분", result)
        # Continuation instruction went out as the next user turn.
        joined = str(client.calls[1]["messages"])
        self.assertIn("Continue exactly from where the previous answer stopped", joined)


class TestClaudePauseTurn(unittest.TestCase):
    def test_pause_turn_continues_with_user_nudge(self):
        client = FakeAnthropicClient([
            _response([_text_block("생각 정리 중")], stop_reason="pause_turn"),
            _response([_text_block("최종 답")]),
        ])
        result = asyncio.run(claude_loop.chat_with_tools(
            [{"role": "user", "content": "q"}],
            client=client, model="claude-sonnet-5",
            tools=[], tool_handlers={}, system_prompt="s",
            budget_usd=5.0,
        ))
        self.assertIn("최종 답", result)
        # The pause round got a plain "continue" user turn, not tool_results.
        msgs = client.calls[1]["messages"]
        user_texts = [
            b.get("text") for m in msgs if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "text"
        ]
        self.assertIn("continue", user_texts)


class TestRoundLimitWarning(unittest.TestCase):
    def test_warning_lands_two_rounds_before_limit(self):
        client = FakeAnthropicClient([
            _response([_tool_use_block("t1", "echo")], stop_reason="tool_use"),
            _response([_text_block("끝")]),
        ])
        with patch.object(claude_loop, "execute_tools_batch",
                          _fake_batch_factory({"echo": ("r", False)})):
            asyncio.run(claude_loop.chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="claude-sonnet-5",
                tools=TOOLS, tool_handlers=HANDLERS, system_prompt="s",
                budget_usd=5.0, max_rounds=3,
            ))
        # round 1 == total_round_limit(3) - 2 → warning rides with the results.
        joined = str(client.calls[1]["messages"])
        self.assertIn("라운드 한도 임박", joined)


class TestOpenAIFinalizationTools(unittest.TestCase):
    def test_forced_final_metadata_and_finalization_call(self):
        expensive = _oai_usage(prompt=10_000_000, completion=1_000_000)
        client = FakeSDKClient([
            _oai_resp("", finish="tool_calls", tool_calls=[_tc("t1", "echo")],
                      usage=expensive),
            _oai_resp("짧은 마무리", finish="tool_calls",
                      tool_calls=[_tc("t2", "save_diary")]),
            _oai_resp("일기 저장 완료 보고"),
        ])
        with patch.object(openai_tool_loop, "execute_tools_batch",
                          _fake_batch_factory({"save_diary": ("saved", False),
                                               "echo": ("r", False)})):
            result = asyncio.run(openai_tool_loop.chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                tools=TOOLS, tool_handlers=HANDLERS, system_prompt="s",
                budget_usd=0.01, finalization_tools=["save_diary"],
                return_metadata=True,
            ))
        self.assertEqual(len(client.calls), 3)
        final_tools = client.calls[1].get("tools")
        self.assertEqual(
            [t["function"]["name"] for t in final_tools], ["save_diary"],
        )
        self.assertNotIn("tools", client.calls[2])
        self.assertEqual(result["finish_reason"], "forced_final")
        self.assertTrue(result["truncated"])
        self.assertIn("일기 저장 완료 보고", result["text"])

    def test_length_continuation_metadata(self):
        client = FakeSDKClient([
            _oai_resp("앞부분 텍스트", finish="length"),
            _oai_resp("뒷부분 텍스트"),
        ])
        result = asyncio.run(openai_tool_loop.chat_with_tools(
            [{"role": "user", "content": "q"}],
            client=client, model="deepseek-chat",
            tools=TOOLS, tool_handlers=HANDLERS, system_prompt="s",
            budget_usd=5.0, continue_on_length=True, max_length_continuations=1,
            return_metadata=True,
        ))
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["continuations_used"], 1)
        self.assertIn("앞부분 텍스트", result["text"])
        self.assertIn("뒷부분 텍스트", result["text"])

    def test_malformed_forced_final_arguments_are_not_executed(self):
        executed = []

        async def fake_batch(tool_uses, tool_handlers, **kwargs):
            executed.extend(name for _tid, name, _input in tool_uses)
            return [(tid, name, inp, "ok", False) for tid, name, inp in tool_uses]

        expensive = _oai_usage(prompt=10_000_000, completion=1_000_000)
        client = FakeSDKClient([
            _oai_resp("", finish="tool_calls", tool_calls=[_tc("t1", "echo")],
                      usage=expensive),
            _oai_resp("", finish="tool_calls",
                      tool_calls=[_tc("t2", "save_diary", "[]")]),
            _oai_resp("malformed 호출은 실행하지 않았다"),
        ])
        with patch.object(openai_tool_loop, "execute_tools_batch", fake_batch):
            result = asyncio.run(openai_tool_loop.chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                tools=TOOLS, tool_handlers=HANDLERS, system_prompt="s",
                budget_usd=0.01, finalization_tools=["save_diary"],
            ))
        self.assertEqual(executed, ["echo"])
        self.assertIn("malformed JSON arguments", str(client.calls[2]["messages"]))
        self.assertIn("실행하지 않았다", result)


if __name__ == "__main__":
    unittest.main()
