"""Unit tests for tool_loop_common — shared loop utilities.

Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import asyncio
import unittest

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

import llm.tool_loop_common as tool_loop_common
from llm.tool_loop_common import (
    TRANSIENT_PROVIDER_STATUSES,
    call_with_transient_retry,
    validate_budget,
    build_budget_tracker,
    build_limit_message,
    build_budget_warning,
    build_round_warning,
    build_stripped_limit_message,
    dedupe_tools_by_name,
    emit_progress,
    check_cancelled,
    is_transient_provider_error,
    provider_status_code,
)


class TestTransientClassifier(unittest.TestCase):
    def test_all_transient_statuses(self):
        for code in TRANSIENT_PROVIDER_STATUSES:
            exc = Exception()
            exc.status_code = code
            self.assertTrue(is_transient_provider_error(exc), code)

    def test_status_via_response_object(self):
        exc = Exception()
        exc.response = SimpleNamespace(status_code=502)
        self.assertEqual(provider_status_code(exc), 502)
        self.assertTrue(is_transient_provider_error(exc))

    def test_4xx_not_transient(self):
        for code in (400, 401, 403, 404, 422):
            exc = Exception()
            exc.status_code = code
            self.assertFalse(is_transient_provider_error(exc), code)

    def test_every_httpx_transport_class_matched_by_name(self):
        # The openai loop's old isinstance() check is now redundant with the
        # shared name tokens — this pins that equivalence.
        for cls in (httpx.TimeoutException, httpx.ReadError, httpx.ConnectError,
                    httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.NetworkError):
            self.assertTrue(is_transient_provider_error(cls("x")), cls.__name__)

    def test_sdk_error_names(self):
        class APIConnectionError(Exception):
            pass

        class APITimeoutError(Exception):
            pass

        self.assertTrue(is_transient_provider_error(APIConnectionError()))
        self.assertTrue(is_transient_provider_error(APITimeoutError()))
        self.assertFalse(is_transient_provider_error(ValueError("nope")))


class TestCallWithTransientRetry(unittest.TestCase):
    @staticmethod
    def _transient(msg="rate limited"):
        exc = Exception(msg)
        exc.status_code = 429
        return exc

    def test_succeeds_on_third_attempt_with_backoff(self):
        attempts = []

        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise self._transient()
            return "ok"

        with patch.object(tool_loop_common.asyncio, "sleep", new=AsyncMock()) as sleep:
            result = asyncio.run(call_with_transient_retry(flaky, label="m"))
        self.assertEqual(result, "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [1.5, 3.0])

    def test_non_transient_raises_immediately(self):
        async def bad():
            raise ValueError("schema error")

        with patch.object(tool_loop_common.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(ValueError):
                asyncio.run(call_with_transient_retry(bad, label="m"))
        sleep.assert_not_awaited()

    def test_exhausted_attempts_raise_last_error(self):
        attempts = []

        async def always_transient():
            attempts.append(1)
            raise self._transient(f"fail {len(attempts)}")

        with patch.object(tool_loop_common.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(call_with_transient_retry(always_transient, label="m"))
        self.assertEqual(len(attempts), 3)
        self.assertIn("fail 3", str(ctx.exception))

    def test_progress_event_emitted(self):
        seen = []

        async def cb(event, detail):
            seen.append((event, detail))

        calls = []

        async def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise self._transient()
            return "ok"

        with patch.object(tool_loop_common.asyncio, "sleep", new=AsyncMock()):
            asyncio.run(call_with_transient_retry(flaky, label="m", on_progress=cb))
        self.assertEqual(seen[0][0], "provider_retry")


class TestDedupeToolsShared(unittest.TestCase):
    def test_first_occurrence_wins(self):
        out = dedupe_tools_by_name([{"name": "a", "v": 1}, {"name": "a", "v": 2}])
        self.assertEqual(out, [{"name": "a", "v": 1}])


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
