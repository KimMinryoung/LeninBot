"""GPT-6 Responses transport preserves tool calls and reasoning across rounds."""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llm.openai_tool_loop as openai_tool_loop
os.environ["LENINBOT_LLM_AUDIT_DB"] = "0"

from llm.openai_tool_loop import _call_sdk, _responses_input, chat_with_tools
from llm.provider_registry import anthropic_pricing_table, openai_compatible_pricing


class OutputItem:
    def __init__(self, data):
        self.data = data

    def model_dump(self, **_kwargs):
        return self.data


class ResponsesEndpoint:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 2:
            return SimpleNamespace(
                model="gpt-6-sol", status="completed", output_text="done",
                output=[OutputItem({"type": "message", "id": "msg_2", "role": "assistant",
                                    "content": [{"type": "output_text", "text": "done"}]})],
                usage=SimpleNamespace(input_tokens=140, output_tokens=5,
                                      input_tokens_details=SimpleNamespace(cached_tokens=0)),
            )
        return SimpleNamespace(
            model="gpt-6-sol", status="completed", output_text="",
            output=[
                OutputItem({"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}),
                OutputItem({"type": "function_call", "id": "fc_1", "call_id": "call_1",
                            "name": "echo", "arguments": "{}", "status": "completed"}),
            ],
            usage=SimpleNamespace(input_tokens=120, output_tokens=30,
                                  input_tokens_details=SimpleNamespace(cached_tokens=20)),
        )


class GPT6ResponsesLoopTest(unittest.TestCase):
    def test_streamed_text_progress(self):
        final = SimpleNamespace(
            model="gpt-6-luna", status="completed", output_text="hello",
            output=[OutputItem({"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "hello"}]})],
            usage=SimpleNamespace(input_tokens=3, output_tokens=1, input_tokens_details=None),
        )

        class Endpoint:
            async def create(self, **kwargs):
                self.kwargs = kwargs

                async def events():
                    yield SimpleNamespace(type="response.output_text.delta", delta="hel")
                    yield SimpleNamespace(type="response.output_text.delta", delta="lo")
                    yield SimpleNamespace(type="response.completed", response=final)

                return events()

        endpoint = Endpoint()
        progress = []

        async def callback(kind, value):
            progress.append((kind, value))

        response = asyncio.run(_call_sdk(
            SimpleNamespace(responses=endpoint), "gpt-6-luna",
            [{"role": "user", "content": "hi"}], max_tokens=100,
            on_progress=callback,
        ))
        self.assertTrue(endpoint.kwargs["stream"])
        self.assertEqual(progress, [("text_delta", "hel"), ("text_delta", "lo")])
        self.assertEqual(response.choices[0].message.content, "hello")

    def test_full_tool_round_uses_responses(self):
        endpoint = ResponsesEndpoint()
        client = SimpleNamespace(responses=endpoint)

        async def run_batch(tool_uses, _handlers, **_kwargs):
            return [(call_id, name, args, "tool-result", False)
                    for call_id, name, args in tool_uses]

        with patch.object(openai_tool_loop, "execute_tools_batch", run_batch):
            answer = asyncio.run(chat_with_tools(
                [{"role": "user", "content": "go"}], client=client, model="gpt-6-sol",
                tools=[{"name": "echo", "description": "echo", "input_schema": {
                    "type": "object", "properties": {}}}],
                tool_handlers={"echo": None}, system_prompt="sys", max_rounds=3,
                max_tokens=1000, budget_usd=5,
            ))
        self.assertEqual(answer, "done")
        self.assertEqual(len(endpoint.calls), 2)
        replay = endpoint.calls[1]["input"]
        self.assertEqual([item["type"] for item in replay if "type" in item][-3:],
                         ["reasoning", "function_call", "function_call_output"])

    def test_new_model_prices(self):
        sol = openai_compatible_pricing("gpt-6-sol")
        self.assertEqual(sol["input"], 2 / 1_000_000)
        self.assertEqual(sol["cache_write"], 2.5 / 1_000_000)
        self.assertEqual(openai_compatible_pricing("gpt-6-sol", input_tokens=272_001)["output"],
                         15 / 1_000_000)
        self.assertEqual(openai_compatible_pricing("gpt-6-luna")["output"], 0.5 / 1_000_000)
        self.assertEqual(anthropic_pricing_table()["claude-opus-5-5"]["cache_creation"],
                         5 / 1_000_000)

    def test_reasoning_and_tool_call_replayed_with_result(self):
        endpoint = ResponsesEndpoint()
        client = SimpleNamespace(responses=endpoint)
        response = asyncio.run(_call_sdk(
            client, "gpt-6-sol", [{"role": "user", "content": "go"}],
            [{"type": "function", "function": {"name": "echo", "parameters": {
                "type": "object", "properties": {}}}}],
            1000, extra_body={"reasoning_effort": "high"},
        ))
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        self.assertEqual(response.choices[0].message.tool_calls[0].id, "call_1")
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 20)
        self.assertEqual(endpoint.calls[0]["reasoning"], {"effort": "high"})
        self.assertEqual(endpoint.calls[0]["tools"][0]["strict"], False)
        self.assertFalse(endpoint.calls[0]["store"])
        replay = _responses_input([
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {
                "name": "echo", "arguments": "{}"}}],
             "_responses_output": response.choices[0].message._responses_output},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ])
        self.assertEqual([item["type"] for item in replay[1:]],
                         ["reasoning", "function_call", "function_call_output"])
        self.assertEqual(replay[1]["encrypted_content"], "opaque")
        self.assertEqual(replay[3]["call_id"], "call_1")


if __name__ == "__main__":
    unittest.main()
