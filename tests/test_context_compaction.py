"""Staged in-call compaction, token calibration and cross-turn tool evidence."""

import json
import unittest
from unittest import mock

from llm.context_compaction import (
    CHECKPOINT_PREFIX,
    KEEP_RECENT_ROUNDS,
    REPLAY_STUB_TEXT,
    compact_anthropic_messages,
    compact_openai_messages,
)
from llm.tool_loop_common import InputTokenCalibrator, usage_input_tokens


def _replay_safe(name):
    return name == "read_file"


def _fits_under(limit):
    def fits(messages):
        size = len(json.dumps(messages, ensure_ascii=False))
        return size <= limit, size
    return fits


def _anthropic_round(i, name, result):
    return [
        {"role": "assistant", "content": [
            {"type": "text", "text": f"step {i}"},
            {"type": "tool_use", "id": f"t{i}", "name": name, "input": {"q": f"query {i}"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": result},
        ]},
    ]


def _openai_round(i, name, result):
    return [
        {"role": "assistant", "content": f"step {i}", "tool_calls": [
            {"id": f"t{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps({"q": f"query {i}"})}},
        ]},
        {"role": "tool", "tool_call_id": f"t{i}", "content": result},
    ]


def _anthropic_pairs_valid(messages):
    """Every tool_use has a tool_result in the next message and vice versa."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and isinstance(m["content"], list):
            ids = {b["id"] for b in m["content"] if b.get("type") == "tool_use"}
            if ids:
                nxt = messages[i + 1]["content"]
                got = {b["tool_use_id"] for b in nxt if b.get("type") == "tool_result"}
                if ids != got:
                    return False
        if m["role"] == "user" and isinstance(m["content"], list):
            ids = {b["tool_use_id"] for b in m["content"] if b.get("type") == "tool_result"}
            if ids:
                prev = messages[i - 1]["content"]
                if ids != {b["id"] for b in prev if b.get("type") == "tool_use"}:
                    return False
    roles = [m["role"] for m in messages]
    return all(a != b for a, b in zip(roles, roles[1:]))


class AnthropicCompactionTest(unittest.TestCase):
    def test_noop_returns_same_object(self):
        msgs = [{"role": "user", "content": "hi"}]
        out, _ = compact_anthropic_messages(msgs, _fits_under(10_000), is_replay_safe=_replay_safe)
        self.assertIs(out, msgs)

    def test_stage_a_replay_stub_first(self):
        msgs = [{"role": "user", "content": "go"}]
        msgs += _anthropic_round(1, "read_file", "R" * 5000)
        msgs += _anthropic_round(2, "web_search", "W" * 5000)
        out, _ = compact_anthropic_messages(msgs, _fits_under(7000), is_replay_safe=_replay_safe)
        self.assertEqual(out[2]["content"][0]["content"], REPLAY_STUB_TEXT)
        self.assertEqual(out[4]["content"][0]["content"], "W" * 5000)
        self.assertEqual(len(msgs[2]["content"][0]["content"]), 5000)  # input untouched

    def test_stage_b_excerpts_old_non_replayable_but_not_newest(self):
        msgs = [{"role": "user", "content": "go"}]
        msgs += _anthropic_round(1, "web_search", "A" * 5000)
        msgs += _anthropic_round(2, "web_search", "B" * 5000)
        out, _ = compact_anthropic_messages(msgs, _fits_under(7000), is_replay_safe=_replay_safe)
        first = out[2]["content"][0]["content"]
        self.assertTrue(first.startswith("A" * 600))
        self.assertIn("shortened from 5000 chars", first)
        self.assertEqual(out[4]["content"][0]["content"], "B" * 5000)

    def test_stage_c_drops_oldest_rounds_with_digest(self):
        msgs = [{"role": "user", "content": "go"}]
        for i in range(1, 7):
            msgs += _anthropic_round(i, "web_search", f"result-{i} " + "x" * 3000)
        out, _ = compact_anthropic_messages(msgs, _fits_under(5000), is_replay_safe=_replay_safe)
        self.assertTrue(_anthropic_pairs_valid(out))
        anchor = out[0]["content"]
        self.assertTrue(anchor.startswith("go"))
        self.assertIn("earlier tool round(s)", anchor)
        self.assertIn("web_search", anchor)
        self.assertIn("result-1", anchor)
        kept_rounds = sum(1 for m in out if m["role"] == "assistant")
        self.assertGreaterEqual(kept_rounds, KEEP_RECENT_ROUNDS)
        self.assertEqual(out[-1]["content"][0]["tool_use_id"], "t6")

    def test_stage_c_keeps_recent_rounds_even_when_still_over(self):
        msgs = [{"role": "user", "content": "go"}]
        for i in range(1, 5):
            msgs += _anthropic_round(i, "web_search", "y" * 3000)
        out, est = compact_anthropic_messages(msgs, _fits_under(100), is_replay_safe=_replay_safe)
        self.assertGreater(est, 100)
        self.assertEqual(sum(1 for m in out if m["role"] == "assistant"), KEEP_RECENT_ROUNDS)
        self.assertTrue(_anthropic_pairs_valid(out))


class OpenAICompactionTest(unittest.TestCase):
    def test_stage_c_preserves_tool_call_pairs(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        for i in range(1, 7):
            msgs += _openai_round(i, "fetch_x_post", f"post-{i} " + "z" * 3000)
        out, _ = compact_openai_messages(msgs, _fits_under(5000), is_replay_safe=_replay_safe)
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("earlier tool round(s)", out[1]["content"])
        call_ids = {c["id"] for m in out for c in m.get("tool_calls") or []}
        result_ids = {m["tool_call_id"] for m in out if m["role"] == "tool"}
        self.assertEqual(call_ids, result_ids)
        self.assertIn("t6", call_ids)
        self.assertEqual(out[2]["role"], "assistant")

    def test_checkpoint_already_applied_is_not_rewritten(self):
        msgs = [{"role": "user", "content": "go"}]
        msgs += _openai_round(1, "read_file", CHECKPOINT_PREFIX + " old]" + " " * 900)
        msgs += _openai_round(2, "read_file", "Q" * 3000)
        out, _ = compact_openai_messages(msgs, _fits_under(1500), is_replay_safe=_replay_safe)
        self.assertTrue(out[2]["content"].startswith(CHECKPOINT_PREFIX + " old]"))
        self.assertEqual(out[4]["content"], REPLAY_STUB_TEXT)


class CalibratorTest(unittest.TestCase):
    def test_learns_ratio_and_ignores_small_samples(self):
        cal = InputTokenCalibrator()
        cal.mark(500)
        cal.observe(1500)
        self.assertEqual(cal.ratio, 1.0)
        cal.mark(10_000)
        cal.observe(20_000)
        self.assertAlmostEqual(cal.ratio, 2.0)
        self.assertEqual(cal.calibrate(1000), 2000)
        cal.mark(10_000)
        cal.observe(10_000)
        self.assertAlmostEqual(cal.ratio, 1.5)

    def test_clamped_and_requires_mark(self):
        cal = InputTokenCalibrator()
        cal.observe(99_999)  # no pending mark
        self.assertEqual(cal.samples, 0)
        cal.mark(10_000)
        cal.observe(1_000_000)
        self.assertEqual(cal.ratio, InputTokenCalibrator.MAX_RATIO)

    def test_usage_input_tokens(self):
        anthropic = mock.Mock(spec=["input_tokens", "cache_creation_input_tokens",
                                    "cache_read_input_tokens"],
                              input_tokens=100, cache_creation_input_tokens=20,
                              cache_read_input_tokens=880)
        self.assertEqual(usage_input_tokens(anthropic), 1000)
        self.assertEqual(usage_input_tokens({"prompt_tokens": 42, "input_tokens": 7}), 42)
        self.assertEqual(usage_input_tokens({"input_tokens": 7}), 7)
        self.assertEqual(usage_input_tokens(mock.MagicMock()), 0)
        self.assertEqual(usage_input_tokens(None), 0)


class ToolEvidenceTest(unittest.TestCase):
    def test_capture_turn_collects_and_nested_disable(self):
        from telegram import tool_evidence
        from tool_gateway.observations import observe_tool_result

        with tool_evidence.capture_turn(True) as entries:
            observe_tool_result("web_search", {"query": "a"}, "found A", False)
            with tool_evidence.capture_turn(False):
                observe_tool_result("fetch_url", {"url": "x"}, "nested", False)
            observe_tool_result("list_agent_tools", {}, "meta", False)
        observe_tool_result("web_search", {}, "after", False)
        self.assertEqual([e["name"] for e in entries], ["web_search"])
        self.assertEqual(entries[0]["result"], "found A")

    def test_owner_user_id(self):
        from telegram.tool_evidence import owner_user_id
        self.assertEqual(owner_user_id("telegram:123"), 123)
        self.assertIsNone(owner_user_id("web:abc"))
        self.assertIsNone(owner_user_id("telegram:abc"))
        self.assertIsNone(owner_user_id(None))

    def test_recent_tool_evidence_budget_and_order(self):
        from telegram import tool_evidence
        rows = [
            {"id": n, "created_at": f"t{n}", "scope_type": "telegram_message",
             "tool_name": "web_search", "args_excerpt": "{}", "result_excerpt": "r" * 5000,
             "result_chars": 5000, "is_error": False}
            for n in (5, 4, 3, 2)
        ]
        with mock.patch("db.query", return_value=rows):
            records = tool_evidence.recent_tool_evidence(1, "since")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["kind"], "prior_tool_results")
        ids = [p["id"] for p in records[0]["payload"]]
        self.assertEqual(ids, [4, 5])  # newest within 12k chars (2 × 5002), chronological

    def test_recent_tool_evidence_degrades_when_table_missing(self):
        from telegram import tool_evidence
        with mock.patch("db.query", side_effect=RuntimeError("relation does not exist")):
            records = tool_evidence.recent_tool_evidence(1, "since")
        self.assertEqual(records[0]["coverage"], "unavailable")

    def test_persist_turn_redacts_and_truncates(self):
        from telegram import tool_evidence
        cursor = mock.MagicMock()
        conn = mock.MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = conn
        with mock.patch("db.get_conn", return_value=ctx):
            tool_evidence.persist_turn(
                7, [{"name": "fetch_url", "args": {"url": "u"}, "result": "x" * 5000,
                     "is_error": False}],
                scope_type="telegram_message", scope_id=9,
            )
        rows = cursor.executemany.call_args[0][1]
        self.assertEqual(rows[0][0], 7)
        self.assertEqual(rows[0][3], "fetch_url")
        self.assertIn("excerpt: 1200 of 5000 chars", rows[0][5])
        self.assertEqual(rows[0][6], 5000)


class ClaudeAdapterCompactionTest(unittest.IsolatedAsyncioTestCase):
    async def test_call_once_compacts_instead_of_raising_and_calibrates(self):
        from types import SimpleNamespace
        from llm.claude_loop import _ClaudeProtocolAdapter

        sent = {}

        async def create(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(usage=SimpleNamespace(
                input_tokens=3000, cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ))

        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        adapter = _ClaudeProtocolAdapter(
            client=client, model="claude-sonnet-5-5", tools=[], tool_handlers={},
            system_prompt="s", max_tokens=1024, max_input_tokens=1000,
            recover_input_via_tools=True, on_progress=None, log_event=None,
            thinking=None, output_config=None, provider_idle_timeout_sec=None,
        )
        msgs = [{"role": "user", "content": "go"}]
        # A tool outside REPLAY_SAFE_TOOLS, so stages A and B cannot suffice.
        for i in range(1, 6):
            msgs += _anthropic_round(i, "post_status_update", "w" * 6000)
        msgs += _anthropic_round(6, "post_status_update", "short result")
        with mock.patch("llm.claude_loop.with_audit_owner", side_effect=lambda c, k, o: k):
            await adapter._call_once(model="m", max_tokens=10, system=[], messages=msgs)
        self.assertLess(len(sent["messages"]), len(msgs))
        self.assertIn("earlier tool round(s)", json.dumps(sent["messages"]))
        self.assertTrue(_anthropic_pairs_valid(sent["messages"]))
        # The mark was consumed by the response; the compacted request is under
        # MIN_SAMPLE_TOKENS, so it does not move the ratio.
        self.assertIsNone(adapter.token_calibrator._pending)
        self.assertEqual(adapter.token_calibrator.samples, 0)


if __name__ == "__main__":
    unittest.main()
