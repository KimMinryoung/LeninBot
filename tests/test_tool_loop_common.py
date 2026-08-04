"""Unit tests for tool_loop_common — shared loop utilities.

Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import asyncio
import unittest

from tool_loop_common import (
    validate_budget,
    build_budget_tracker,
    build_limit_message,
    build_budget_warning,
    build_round_warning,
    build_stripped_limit_message,
    emit_progress,
    check_cancelled,
)


class TestValidateBudget(unittest.TestCase):
    def test_valid_float_passthrough(self):
        self.assertEqual(validate_budget(1.5), 1.5)

    def test_string_number_coerced(self):
        self.assertEqual(validate_budget("0.5"), 0.5)

    def test_invalid_falls_back_to_030(self):
        self.assertEqual(validate_budget(None), 0.30)
        self.assertEqual(validate_budget("abc"), 0.30)

    def test_nonpositive_clamped_to_001(self):
        self.assertEqual(validate_budget(0), 0.01)
        self.assertEqual(validate_budget(-5), 0.01)


class TestBudgetTracker(unittest.TestCase):
    def test_shape_and_copy(self):
        details = ["a"]
        t = build_budget_tracker(1.23, 4, True, details)
        self.assertEqual(t, {
            "total_cost": 1.23, "rounds_used": 4,
            "was_interrupted": True, "tool_work_details": ["a"],
        })
        details.append("b")  # must have been copied
        self.assertEqual(t["tool_work_details"], ["a"])


class TestLimitMessages(unittest.TestCase):
    def test_limit_message_includes_cost_and_rounds(self):
        msg = build_limit_message("예산 소진", 0.31, 0.30, 7, 10, was_still_working=False)
        self.assertIn("[SYSTEM]", msg)
        self.assertIn("$0.310/$0.30", msg)
        self.assertIn("7/10", msg)
        self.assertNotIn("orchestrator", msg)

    def test_limit_message_escalation_hint(self):
        msg = build_limit_message("예산 소진", 0.31, 0.30, 7, 10, was_still_working=True)
        self.assertIn("orchestrator", msg)

    def test_limit_message_finalization_tools(self):
        msg = build_limit_message(
            "예산 소진", 0.31, 0.30, 7, 10, False,
            finalization_tools=["save_diary", "publish"],
        )
        self.assertIn("save_diary, publish", msg)
        self.assertIn("마감 도구", msg)

    def test_warnings_contain_numbers(self):
        self.assertIn("$0.240/$0.30", build_budget_warning(0.24, 0.30))
        self.assertIn("8/10", build_round_warning(8, 10))
        self.assertIn("한도", build_stripped_limit_message("도구 호출 한도 도달"))


class TestEmitProgress(unittest.TestCase):
    def test_none_callback_is_noop(self):
        asyncio.run(emit_progress(None, "thinking", "x"))  # must not raise

    def test_callback_receives_event(self):
        seen = []

        async def cb(event, detail):
            seen.append((event, detail))

        asyncio.run(emit_progress(cb, "tool_call", "detail"))
        self.assertEqual(seen, [("tool_call", "detail")])

    def test_callback_exception_swallowed(self):
        async def cb(event, detail):
            raise RuntimeError("boom")

        asyncio.run(emit_progress(cb, "x", "y"))  # must not raise


class TestCheckCancelled(unittest.TestCase):
    def test_none_task_id_is_noop(self):
        check_cancelled(None)  # must not raise, must not touch Redis


if __name__ == "__main__":
    unittest.main()
