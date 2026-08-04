"""Characterization tests for claude_loop's pure helpers.

These pin down current behavior before any loop refactoring.
Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import unittest
from types import SimpleNamespace

from claude_loop import (
    dedupe_tools_by_name,
    estimate_tokens,
    _append_user_text_message,
    _calculate_cost,
    _coerce_text,
    _content_block_for_replay,
    _is_transient_provider_error,
    _normalize_initial_messages,
    _pricing_for,
    _provider_status_code,
    _to_block_dict,
    _update_budget_tracker,
    _usage_to_dict,
    _with_message_cache_breakpoint,
)


class TestDedupeTools(unittest.TestCase):
    def test_keeps_first_occurrence(self):
        tools = [{"name": "a", "v": 1}, {"name": "b"}, {"name": "a", "v": 2}]
        out = dedupe_tools_by_name(tools)
        self.assertEqual(out, [{"name": "a", "v": 1}, {"name": "b"}])

    def test_empty_and_none(self):
        self.assertEqual(dedupe_tools_by_name(None), [])
        self.assertEqual(dedupe_tools_by_name([]), [])

    def test_non_dict_passthrough(self):
        out = dedupe_tools_by_name(["raw", {"name": "a"}])
        self.assertEqual(out, ["raw", {"name": "a"}])


class TestTransientErrors(unittest.TestCase):
    def test_status_code_direct_attr(self):
        exc = Exception()
        exc.status_code = 429
        self.assertEqual(_provider_status_code(exc), 429)
        self.assertTrue(_is_transient_provider_error(exc))

    def test_status_code_via_response(self):
        exc = Exception()
        exc.response = SimpleNamespace(status_code=503)
        self.assertEqual(_provider_status_code(exc), 503)
        self.assertTrue(_is_transient_provider_error(exc))

    def test_400_is_not_transient(self):
        exc = Exception()
        exc.status_code = 400
        self.assertFalse(_is_transient_provider_error(exc))

    def test_class_name_token_match(self):
        class APIConnectionError(Exception):
            pass

        class SomeTimeoutThing(Exception):
            pass

        class ValueBoom(Exception):
            pass

        self.assertTrue(_is_transient_provider_error(APIConnectionError()))
        self.assertTrue(_is_transient_provider_error(SomeTimeoutThing()))
        self.assertFalse(_is_transient_provider_error(ValueBoom()))


class TestPricing(unittest.TestCase):
    def test_known_model_exact(self):
        p = _pricing_for("claude-sonnet-5")
        self.assertIn("input", p)
        self.assertIn("cache_read", p)

    def test_pinned_date_variant_prefix_match(self):
        base = _pricing_for("claude-haiku-4-5")
        pinned = _pricing_for("claude-haiku-4-5-20251001")
        self.assertEqual(base, pinned)

    def test_unknown_falls_back_to_sonnet(self):
        self.assertEqual(_pricing_for("totally-unknown"), _pricing_for("claude-sonnet-5"))
        self.assertEqual(_pricing_for(""), _pricing_for("claude-sonnet-5"))

    def test_calculate_cost_uses_all_token_kinds(self):
        p = _pricing_for("claude-sonnet-5")
        usage = SimpleNamespace(
            input_tokens=1000, output_tokens=100,
            cache_creation_input_tokens=50, cache_read_input_tokens=10,
        )
        expected = (
            1000 * p["input"] + 100 * p["output"]
            + 50 * p["cache_creation"] + 10 * p["cache_read"]
        )
        self.assertAlmostEqual(_calculate_cost(usage, "claude-sonnet-5"), expected)

    def test_usage_to_dict(self):
        usage = SimpleNamespace(input_tokens=5, output_tokens=7,
                                cache_creation_input_tokens=0, cache_read_input_tokens=None)
        self.assertEqual(_usage_to_dict(usage), {
            "input_tokens": 5, "output_tokens": 7,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        })
        self.assertEqual(_usage_to_dict(None), {})


class TestCoerceText(unittest.TestCase):
    def test_scalars(self):
        self.assertEqual(_coerce_text(None), "")
        self.assertEqual(_coerce_text("x"), "x")
        self.assertEqual(_coerce_text(3), "3")

    def test_text_block(self):
        self.assertEqual(_coerce_text({"type": "text", "text": "hi"}), "hi")

    def test_nested_list(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        self.assertEqual(_coerce_text(blocks), "a\nb")

    def test_content_key_recursion(self):
        self.assertEqual(_coerce_text({"content": [{"type": "text", "text": "in"}]}), "in")


class TestNormalizeInitialMessages(unittest.TestCase):
    def test_merges_consecutive_same_role(self):
        out = _normalize_initial_messages([
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ])
        self.assertEqual(out, [
            {"role": "user", "content": "a\nb"},
            {"role": "assistant", "content": "c"},
        ])

    def test_flattens_tool_blocks_to_text(self):
        out = _normalize_initial_messages([
            {"role": "assistant", "content": [{"type": "text", "text": "prose"}]},
        ])
        self.assertEqual(out, [{"role": "assistant", "content": "prose"}])

    def test_alias_roles(self):
        out = _normalize_initial_messages([
            {"role": "model", "content": "m"},
            {"role": "weird", "content": "w"},
        ])
        self.assertEqual([m["role"] for m in out], ["assistant", "user"])

    def test_non_dict_becomes_user(self):
        out = _normalize_initial_messages(["raw text"])
        self.assertEqual(out, [{"role": "user", "content": "raw text"}])


class TestAppendUserText(unittest.TestCase):
    def test_appends_new_user_message(self):
        msgs = [{"role": "assistant", "content": "a"}]
        _append_user_text_message(msgs, "hello")
        self.assertEqual(msgs[-1], {"role": "user", "content": "hello"})

    def test_merges_into_trailing_user_string(self):
        msgs = [{"role": "user", "content": "first"}]
        _append_user_text_message(msgs, "second")
        self.assertEqual(msgs, [{"role": "user", "content": "first\nsecond"}])

    def test_appends_block_to_trailing_user_list(self):
        msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "r"}]}]
        _append_user_text_message(msgs, "note")
        self.assertEqual(msgs[0]["content"][-1], {"type": "text", "text": "note"})


class TestCacheBreakpoints(unittest.TestCase):
    MSGS = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]

    @staticmethod
    def _marked_indexes(msgs):
        marked = []
        for i, m in enumerate(msgs):
            content = m.get("content")
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("cache_control") for b in content
            ):
                marked.append(i)
        return marked

    def test_zero_marks_noop(self):
        out = _with_message_cache_breakpoint(self.MSGS, 0)
        self.assertEqual(out, self.MSGS)

    def test_one_mark_last_assistant_only(self):
        out = _with_message_cache_breakpoint(self.MSGS, 1)
        self.assertEqual(self._marked_indexes(out), [3])

    def test_two_marks_first_and_last_assistant(self):
        out = _with_message_cache_breakpoint(self.MSGS, 2)
        self.assertEqual(self._marked_indexes(out), [1, 3])

    def test_input_not_mutated(self):
        msgs = [dict(m) for m in self.MSGS]
        _with_message_cache_breakpoint(msgs, 2)
        self.assertEqual(msgs, self.MSGS)

    def test_no_assistant_noop(self):
        msgs = [{"role": "user", "content": "u"}]
        out = _with_message_cache_breakpoint(msgs, 2)
        self.assertEqual(out, msgs)


class TestBlockDicts(unittest.TestCase):
    def test_dict_passthrough_drops_none(self):
        self.assertEqual(_to_block_dict({"type": "text", "text": "t", "id": None}),
                         {"type": "text", "text": "t"})

    def test_namespace_object(self):
        b = SimpleNamespace(type="tool_use", id="i1", name="n", input={"a": 1})
        out = _to_block_dict(b)
        self.assertEqual(out["type"], "tool_use")
        self.assertEqual(out["id"], "i1")
        self.assertEqual(out["input"], {"a": 1})

    def test_replay_keeps_thinking_and_text_only(self):
        self.assertIsNone(_content_block_for_replay({"type": "tool_use", "id": "x"}))
        self.assertEqual(
            _content_block_for_replay({"type": "text", "text": "t"}),
            {"type": "text", "text": "t"},
        )
        thinking = {"type": "thinking", "thinking": "...", "signature": "s"}
        self.assertEqual(_content_block_for_replay(thinking), thinking)


class TestBudgetTrackerUpdate(unittest.TestCase):
    def test_none_tracker_noop(self):
        _update_budget_tracker(None, total_cost=1, rounds_used=1,
                               was_interrupted=False, tool_work_details=[])

    def test_response_metadata_and_refusal_details(self):
        tracker = {}
        response = SimpleNamespace(
            stop_reason="refusal",
            usage=SimpleNamespace(input_tokens=1, output_tokens=2,
                                  cache_creation_input_tokens=0, cache_read_input_tokens=0),
            stop_details=SimpleNamespace(category="cat", explanation="why"),
        )
        _update_budget_tracker(tracker, total_cost=0.5, rounds_used=3,
                               was_interrupted=False, tool_work_details=["d"],
                               response=response, model="m")
        self.assertEqual(tracker["stop_reason"], "refusal")
        self.assertEqual(tracker["stop_details"], {"category": "cat", "explanation": "why"})
        self.assertEqual(tracker["usage"]["output_tokens"], 2)
        self.assertEqual(tracker["model"], "m")


class TestEstimateTokens(unittest.TestCase):
    def test_cjk_counts_one_per_char(self):
        self.assertEqual(estimate_tokens("한" * 40), 40)

    def test_ascii_counts_four_chars_per_token(self):
        self.assertEqual(estimate_tokens("x" * 40), 10)
        self.assertEqual(estimate_tokens(""), 0)

    def test_shared_with_openai_loop(self):
        from tool_loop_common import estimate_text_tokens
        self.assertIs(estimate_tokens, estimate_text_tokens)


if __name__ == "__main__":
    unittest.main()
