"""Characterization tests for openai_tool_loop's pure helpers.

Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import unittest
from types import SimpleNamespace

import httpx

from llm.openai_tool_loop import (
    _build_tc_list,
    _calculate_cost,
    _checkpoint_tool_results_for_replay,
    _convert_tool_anthropic_to_openai,
    _convert_tools,
    _ensure_system_first,
    _estimate_tokens,
    _extract_response,
    _is_strict_safe_schema,
    _is_tool_protocol_error,
    _is_transient_transport_error,
    _normalize_messages,
    _rescue_inline_tool_calls,
    _strip_tool_protocol,
    _truncate_to_context,
    _validate_tool_results,
)


class TestTransientTransportErrors(unittest.TestCase):
    def test_httpx_types(self):
        self.assertTrue(_is_transient_transport_error(httpx.ConnectError("x")))
        self.assertTrue(_is_transient_transport_error(httpx.ReadError("x")))

    def test_status_codes(self):
        exc = Exception()
        exc.status_code = 529
        self.assertTrue(_is_transient_transport_error(exc))
        exc2 = Exception()
        exc2.status_code = 400
        self.assertFalse(_is_transient_transport_error(exc2))

    def test_name_tokens(self):
        class APITimeoutError(Exception):
            pass

        self.assertTrue(_is_transient_transport_error(APITimeoutError()))
        self.assertFalse(_is_transient_transport_error(ValueError()))


class TestToolProtocolError(unittest.TestCase):
    def test_positive_match_requires_400_and_signal(self):
        err = Exception("400 Bad Request: messages with role tool must be a response")
        self.assertTrue(_is_tool_protocol_error(err))

    def test_generic_400_not_matched(self):
        self.assertFalse(_is_tool_protocol_error(Exception("400 invalid api key")))

    def test_signal_without_400_not_matched(self):
        self.assertFalse(_is_tool_protocol_error(Exception("tool_call_id missing")))


class TestSchemaConversion(unittest.TestCase):
    def test_strict_safe_schema(self):
        params = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        self.assertTrue(_is_strict_safe_schema(params))

    def test_missing_required_not_strict(self):
        params = {"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a"]}
        self.assertFalse(_is_strict_safe_schema(params))

    def test_default_blocks_strict(self):
        params = {
            "type": "object",
            "properties": {"a": {"type": "string", "default": "x"}},
            "required": ["a"],
        }
        self.assertFalse(_is_strict_safe_schema(params))

    def test_convert_sets_strict_only_when_safe(self):
        tool = {
            "name": "t",
            "description": "d",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        }
        out = _convert_tool_anthropic_to_openai(tool)
        self.assertEqual(out["type"], "function")
        self.assertTrue(out["function"]["strict"])

        tool2 = {"name": "t2", "input_schema": {"type": "object", "properties": {"a": {}}}}
        out2 = _convert_tool_anthropic_to_openai(tool2)
        self.assertNotIn("strict", out2["function"])

    def test_convert_tools_dedupes_and_strips_cache_control(self):
        tools = [
            {"name": "a", "input_schema": {"type": "object", "properties": {}},
             "cache_control": {"type": "ephemeral"}},
            {"name": "a", "input_schema": {"type": "object", "properties": {}}},
        ]
        out = _convert_tools(tools)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "a")
        self.assertNotIn("cache_control", out[0])


class TestNormalizeMessages(unittest.TestCase):
    def test_drops_tool_role_and_merges(self):
        out = _normalize_messages([
            {"role": "user", "content": "u1"},
            {"role": "tool", "tool_call_id": "x", "content": "r"},
            {"role": "user", "content": "u2"},
        ])
        self.assertEqual(out, [{"role": "user", "content": "u1\nu2"}])

    def test_tool_blocks_become_readable_text(self):
        out = _normalize_messages([
            {"role": "assistant", "content": [
                {"type": "text", "text": "before"},
                {"type": "tool_use", "name": "search", "input": {"q": "x"}},
            ]},
        ])
        self.assertEqual(len(out), 1)
        self.assertIn("before", out[0]["content"])
        self.assertIn("[도구 호출: search", out[0]["content"])

    def test_assistant_tool_calls_without_text(self):
        out = _normalize_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "fn"}},
            ]},
        ])
        self.assertEqual(out, [{"role": "assistant", "content": "[도구 호출: fn]"}])


class TestStripToolProtocol(unittest.TestCase):
    def test_full_round_trip(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "function": {"name": "f", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "result-body"},
            {"role": "assistant", "content": "answer"},
        ]
        out = _strip_tool_protocol(msgs)
        self.assertTrue(all(m["role"] != "tool" for m in out))
        self.assertTrue(all(not m.get("tool_calls") for m in out))
        joined = "\n".join(m["content"] for m in out)
        self.assertIn("[도구 호출: f({})]", joined)
        self.assertIn("result-body", joined)
        self.assertIn("answer", joined)

    def test_no_empty_messages(self):
        out = _strip_tool_protocol([{"role": "user", "content": "  "}])
        self.assertEqual(out, [])


class TestEnsureSystemFirst(unittest.TestCase):
    def test_dedupes_and_prepends(self):
        msgs = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "mid"},
        ]
        out = _ensure_system_first(msgs, "new-sys")
        self.assertEqual(out[0], {"role": "system", "content": "new-sys"})
        self.assertEqual([m["role"] for m in out], ["system", "user"])

    def test_empty_system_prompt_removes_only(self):
        out = _ensure_system_first([{"role": "system", "content": "old"}], "")
        self.assertEqual(out, [])


class TestEstimateAndTruncate(unittest.TestCase):
    def test_cjk_counts_heavier(self):
        ascii_est = _estimate_tokens([{"role": "user", "content": "abcd" * 10}])
        hangul_est = _estimate_tokens([{"role": "user", "content": "한" * 40}])
        self.assertEqual(ascii_est, 10)
        self.assertEqual(hangul_est, 40)

    def test_truncate_keeps_system_and_recent(self):
        msgs = [{"role": "system", "content": "S" * 40}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i} " + "x" * 400}
            for i in range(10)
        ]
        out = _truncate_to_context(msgs, context_limit=500, max_tokens=100)
        self.assertEqual(out[0]["role"], "system")
        self.assertLess(len(out), len(msgs))
        self.assertIn("msg9", out[-1]["content"])  # most recent survives

    def test_no_truncation_when_fits(self):
        msgs = [{"role": "user", "content": "short"}]
        self.assertEqual(_truncate_to_context(msgs, 10000, 100), msgs)


class TestValidateToolResults(unittest.TestCase):
    def test_missing_ids_detected(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "a"}, {"id": "b"},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "ok"},
        ]
        self.assertEqual(_validate_tool_results(msgs), ["b"])

    def test_all_resolved(self):
        msgs = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "ok"},
        ]
        self.assertEqual(_validate_tool_results(msgs), [])


class TestInlineToolCallRescue(unittest.TestCase):
    def test_rescues_and_strips(self):
        content = 'prose <tool_call>{"name": "f", "arguments": {"q": 1}}</tool_call> tail'
        cleaned, calls = _rescue_inline_tool_calls(content)
        self.assertNotIn("<tool_call>", cleaned)
        self.assertIn("prose", cleaned)
        self.assertIn("tail", cleaned)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "f")
        self.assertEqual(calls[0]["function"]["arguments"], '{"q": 1}')

    def test_double_encoded_arguments(self):
        content = '<tool_call>{"name": "f", "arguments": "{\\"q\\": 1}"}</tool_call>'
        _, calls = _rescue_inline_tool_calls(content)
        self.assertEqual(calls[0]["function"]["arguments"], '{"q": 1}')

    def test_no_tags_passthrough(self):
        self.assertEqual(_rescue_inline_tool_calls("plain"), ("plain", None))

    def test_invalid_json_ignored(self):
        content = "<tool_call>{not json}</tool_call>"
        cleaned, calls = _rescue_inline_tool_calls(content)
        self.assertIsNone(calls)


class TestExtractResponse(unittest.TestCase):
    @staticmethod
    def _raw(content, finish="stop", tool_calls=None, reasoning=None):
        msg = {"content": content, "tool_calls": tool_calls}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return {"choices": [{"message": msg, "finish_reason": finish}]}

    def test_plain_text(self):
        finish, content, tool_calls, _msg, usage = _extract_response(
            False, self._raw("hello"))
        self.assertEqual((finish, content, tool_calls, usage), ("stop", "hello", None, None))

    def test_think_tags_stripped_and_promoted(self):
        finish, content, *_ = _extract_response(
            False, self._raw("<think>secret</think>answer"))
        self.assertEqual(finish, "stop")
        self.assertIn("answer", content)
        self.assertNotIn("<think>", content)
        self.assertIn("💭 secret", content)  # reasoning surfaced on final reply

    def test_reasoning_not_surfaced_when_disabled(self):
        _, content, *_ = _extract_response(
            False, self._raw("answer", reasoning="r"), surface_reasoning=False)
        self.assertEqual(content, "answer")

    def test_inline_tool_call_rescued_flips_finish_reason(self):
        raw = self._raw('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        finish, content, tool_calls, *_ = _extract_response(False, raw)
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(len(tool_calls), 1)


class TestBuildTcList(unittest.TestCase):
    def test_dict_mode(self):
        out = _build_tc_list(False, [
            {"id": "1", "function": {"name": "f", "arguments": '{"a":1}'}},
            {"id": "", "function": {"name": "g", "arguments": ""}},  # dropped
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "f")

    def test_sdk_mode(self):
        tc = SimpleNamespace(id="1", function=SimpleNamespace(name="f", arguments=None))
        out = _build_tc_list(True, [tc])
        self.assertEqual(out[0]["function"]["arguments"], "{}")


class TestCalculateCost(unittest.TestCase):
    def test_deepseek_style_cache_fields(self):
        usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=100,
            prompt_cache_hit_tokens=500, prompt_cache_miss_tokens=1500,
        )
        cost = _calculate_cost(usage, "deepseek-chat")
        self.assertGreater(cost, 0)

    def test_openai_style_cached_details(self):
        usage = SimpleNamespace(
            prompt_tokens=2000, completion_tokens=100,
            prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=500),
        )
        cost = _calculate_cost(usage, "gpt-4o")
        self.assertGreater(cost, 0)


class TestCheckpointToolResults(unittest.TestCase):
    def test_noop_when_under_limit(self):
        msgs = [{"role": "user", "content": "small"}]
        out, est = _checkpoint_tool_results_for_replay(msgs, max_input_tokens=10_000)
        self.assertIs(out, msgs)

    def test_large_replay_safe_result_checkpointed(self):
        # read_file is registered replay-safe in tool_gateway.inference
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "X" * 8000},
        ]
        out, est = _checkpoint_tool_results_for_replay(msgs, max_input_tokens=100)
        self.assertIn("[Input checkpoint:", out[1]["content"])
        # original untouched
        self.assertEqual(len(msgs[1]["content"]), 8000)


if __name__ == "__main__":
    unittest.main()
