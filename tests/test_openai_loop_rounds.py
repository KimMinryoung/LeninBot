"""Round-loop tests for openai_tool_loop.chat_with_tools with a fake SDK client.

No network, no Redis, no DB: the AsyncOpenAI-shaped client is a stub returning
canned ChatCompletion-shaped namespaces, and execute_tools_batch is patched.
Pins the loop's core contracts before retry/idle-guard hardening:
  * plain text turn returns text; return_metadata returns the dict shape
  * tool round → tool executed → role:"tool" results paired → final text
  * safety net synthesizes a tool message for unexecuted tool_call ids
  * malformed tool arguments produce an error tool message, not a crash
  * transient API error is retried and the turn still succeeds
  * tool-protocol 400 triggers strip-and-retry recovery
  * refusal surfaces as a ⚠️ answer with finish_reason="refusal"

Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import openai_tool_loop
import tool_loop_common
from openai_tool_loop import chat_with_tools


def _usage(prompt=1000, completion=100):
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


def _resp(content, finish="stop", tool_calls=None, usage=None, refusal=None):
    msg = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        reasoning_content="", refusal=refusal,
    )
    return SimpleNamespace(
        model="deepseek-chat",
        choices=[SimpleNamespace(finish_reason=finish, message=msg)],
        usage=usage or _usage(),
    )


class FakeSDKClient:
    """AsyncOpenAI-shaped stub: responses list may contain Exceptions to raise."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if not outer._responses:
                    raise AssertionError("FakeSDKClient ran out of canned responses")
                item = outer._responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        self.chat = SimpleNamespace(completions=_Completions())


def _fake_batch_factory(results_by_name, skip_names=()):
    async def _fake_batch(tool_uses, tool_handlers, **kwargs):
        out = []
        for tid, tname, tinput in tool_uses:
            if tname in skip_names:
                continue
            result, is_error = results_by_name.get(tname, (f"ran:{tname}", False))
            out.append((tid, tname, tinput, result, is_error))
        return out

    return _fake_batch


TOOLS = [{
    "name": "echo",
    "description": "echo tool",
    "input_schema": {"type": "object", "properties": {}},
}]

BASE_KWARGS = dict(
    tools=TOOLS,
    tool_handlers={"echo": None},
    system_prompt="sys",
    max_rounds=5,
    max_tokens=256,
    budget_usd=5.0,
)


class TestPlainTextTurn(unittest.TestCase):
    def test_returns_text_and_tracker(self):
        client = FakeSDKClient([_resp("최종 답변")])
        tracker = {}
        result = asyncio.run(chat_with_tools(
            [{"role": "user", "content": "질문"}],
            client=client, model="deepseek-chat",
            budget_tracker=tracker,
            **BASE_KWARGS,
        ))
        self.assertEqual(result, "최종 답변")
        self.assertEqual(tracker["rounds_used"], 1)
        self.assertGreater(tracker["total_cost"], 0)
        self.assertEqual(len(client.calls), 1)
        # system prompt injected as the first message
        self.assertEqual(client.calls[0]["messages"][0],
                         {"role": "system", "content": "sys"})

    def test_return_metadata_shape(self):
        client = FakeSDKClient([_resp("답")])
        result = asyncio.run(chat_with_tools(
            [{"role": "user", "content": "q"}],
            client=client, model="deepseek-chat",
            return_metadata=True,
            **BASE_KWARGS,
        ))
        self.assertEqual(result["text"], "답")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(result["complete"])
        self.assertEqual(result["rounds"], 1)
        self.assertGreater(result["cost_usd"], 0)


class TestToolRound(unittest.TestCase):
    def test_tool_executed_then_final_text(self):
        client = FakeSDKClient([
            _resp("", finish="tool_calls", tool_calls=[_tc("t1", "echo", '{"q": 1}')]),
            _resp("완료"),
        ])
        with patch.object(openai_tool_loop, "execute_tools_batch",
                          _fake_batch_factory({"echo": ("echo-result", False)})):
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                **BASE_KWARGS,
            ))
        self.assertEqual(result, "완료")
        msgs = client.calls[1]["messages"]
        assistant = [m for m in msgs if m.get("tool_calls")]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["tool_calls"][0]["id"], "t1")
        self.assertEqual(tool_msgs, [{
            "role": "tool", "tool_call_id": "t1", "content": "echo-result",
        }])

    def test_safety_net_synthesizes_missing_result(self):
        client = FakeSDKClient([
            _resp("", finish="tool_calls", tool_calls=[_tc("t1", "echo")]),
            _resp("done"),
        ])
        with patch.object(openai_tool_loop, "execute_tools_batch",
                          _fake_batch_factory({}, skip_names={"echo"})):
            asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                **BASE_KWARGS,
            ))
        tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "t1")
        self.assertIn("skipped", tool_msgs[0]["content"])

    def test_malformed_arguments_become_error_result(self):
        client = FakeSDKClient([
            _resp("", finish="tool_calls",
                  tool_calls=[_tc("t1", "echo", "{not json")]),
            _resp("done"),
        ])
        with patch.object(openai_tool_loop, "execute_tools_batch",
                          _fake_batch_factory({})):
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                **BASE_KWARGS,
            ))
        self.assertEqual(result, "done")
        tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("malformed JSON arguments", tool_msgs[0]["content"])


class TestTransientRetry(unittest.TestCase):
    def test_transient_error_retried_and_succeeds(self):
        err = Exception("service unavailable")
        err.status_code = 503
        client = FakeSDKClient([err, _resp("살아남")])
        with patch.object(tool_loop_common.asyncio, "sleep", new=AsyncMock()):
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                **BASE_KWARGS,
            ))
        self.assertEqual(result, "살아남")
        self.assertEqual(len(client.calls), 2)

    def test_two_transient_errors_then_success_with_backoff(self):
        def _err():
            e = Exception("rate limited")
            e.status_code = 429
            return e

        client = FakeSDKClient([_err(), _err(), _resp("3수만에")])
        with patch.object(tool_loop_common.asyncio, "sleep", new=AsyncMock()) as sleep:
            result = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                **BASE_KWARGS,
            ))
        self.assertEqual(result, "3수만에")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [1.5, 3.0])

    def test_non_transient_error_raises(self):
        err = Exception("401 invalid api key")
        err.status_code = 401
        client = FakeSDKClient([err])
        with self.assertRaises(Exception):
            asyncio.run(chat_with_tools(
                [{"role": "user", "content": "q"}],
                client=client, model="deepseek-chat",
                **BASE_KWARGS,
            ))


class TestProtocolErrorRecovery(unittest.TestCase):
    def test_tool_protocol_400_strips_and_retries(self):
        err = Exception("400 Bad Request: messages with role tool must follow tool_calls")
        client = FakeSDKClient([err, _resp("복구됨")])
        result = asyncio.run(chat_with_tools(
            [{"role": "user", "content": "q"}],
            client=client, model="deepseek-chat",
            **BASE_KWARGS,
        ))
        self.assertEqual(result, "복구됨")
        self.assertEqual(len(client.calls), 2)
        # retried request must contain no tool protocol
        retry_msgs = client.calls[1]["messages"]
        self.assertTrue(all(m.get("role") != "tool" and not m.get("tool_calls")
                            for m in retry_msgs))


class TestRefusal(unittest.TestCase):
    def test_refusal_surfaces_with_reason(self):
        client = FakeSDKClient([_resp("", refusal="정책 위반")])
        result = asyncio.run(chat_with_tools(
            [{"role": "user", "content": "q"}],
            client=client, model="deepseek-chat",
            return_metadata=True,
            **BASE_KWARGS,
        ))
        self.assertEqual(result["finish_reason"], "refusal")
        self.assertIn("정책 위반", result["text"])


if __name__ == "__main__":
    unittest.main()
