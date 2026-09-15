"""No-network billing tests, including independent-process contention."""
import asyncio
import json
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import tavily

from web_gateway import budget
from web_gateway import search
from security_gateway.context import CallerContext, caller_scope
from tool_gateway.results import ToolFailure


def contend(paths):
    budget.STORE_PATH, budget.CONFIG_PATH = map(Path, paths)
    try:
        with budget.paid_request("tavily", "search", "basic") as charge:
            charge.complete(1)
        return True
    except budget.PaidWebBudgetError:
        return False


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "usage.sqlite3"
        self.config = Path(self.tmp.name) / "policy.json"
        self.set_cap(0.016)
        for mock in (patch.object(budget, "STORE_PATH", self.path),
                     patch.object(budget, "CONFIG_PATH", self.config)):
            mock.start()
            self.addCleanup(mock.stop)
        search._SEARCH_CACHE.clear()
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()
        self.addCleanup(search._SEARCH_CACHE.clear)
        self.addCleanup(search._PROVIDER_UNAVAILABLE_UNTIL.clear)

    def set_cap(self, cap):
        self.config.write_text(json.dumps({"daily_budget_usd": cap,
            "tavily_credit_usd": 0.008, "brave_search_usd": 0.005}))

    def rows(self, by="service"):
        return budget.report("2000-01-01", by=by)

    def test_parallel_processes_cannot_overdraw(self):
        paths = (str(self.path), str(self.config))
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(contend, [paths] * 12))
        self.assertEqual(sum(results), 2)
        self.assertAlmostEqual(sum(r["accounted_usd"] for r in self.rows()), 0.016)

    def test_reported_zero_and_fractional_extract_release_reservations(self):
        self.set_cap(0.008)
        with budget.paid_request("tavily", "search", "basic") as charge:
            charge.complete(0)
        with budget.paid_request("tavily", "extract", "basic") as charge:
            charge.complete(0.2)
        with budget.paid_request("brave", "search", "basic") as charge:
            charge.complete()
        self.assertAlmostEqual(sum(r["accounted_usd"] for r in self.rows()), 0.0066)
        self.assertEqual(sum(r["estimated_requests"] for r in self.rows()), 1)
        with self.assertRaises(budget.PaidWebBudgetError):
            budget.reserve("tavily", "search", "basic")

    def test_unknown_failure_crash_and_bad_usage_keep_reservations(self):
        with self.assertRaises(TimeoutError):
            with budget.paid_request("tavily", "search", "basic"):
                raise TimeoutError()
        budget.reserve("tavily", "extract", "basic")  # Simulated crash, no settlement
        self.assertEqual(sum(r["unknown_requests"] for r in self.rows()), 2)
        with self.assertRaises(budget.PaidWebBudgetError):
            budget.reserve("brave", "search", "basic")
        self.set_cap(0.024)
        with budget.paid_request("tavily", "search", "basic") as charge:
            charge.complete(float("nan"))
        self.assertAlmostEqual(sum(r["accounted_usd"] for r in self.rows()), 0.024)

    def test_zero_cap_bad_config_and_store_failure_fail_closed(self):
        self.set_cap(0)
        with self.assertRaises(budget.PaidWebBudgetError):
            budget.reserve("tavily", "search", "basic")
        for content in ('{', '{"daily_budget_usd": -1}', '[]'):
            self.config.write_text(content)
            with self.assertRaises(budget.PaidWebBudgetError):
                budget.reserve("tavily", "search", "basic")
        self.set_cap(1)
        with patch.object(budget, "STORE_PATH", Path(self.tmp.name)):
            with self.assertRaises(budget.PaidWebBudgetError):
                budget.reserve("tavily", "search", "basic")

    def test_utc_rollover_and_live_config_change(self):
        self.set_cap(0.008)
        for day in (11, 12):
            with patch.object(budget, "datetime") as clock:
                clock.now.return_value = datetime(2026, 9, day, 23, 59, tzinfo=timezone.utc)
                with budget.paid_request("tavily", "search", "basic") as charge:
                    charge.complete(1)
                with self.assertRaises(budget.PaidWebBudgetError):
                    budget.reserve("tavily", "search", "basic")
        self.assertEqual({r["day"] for r in self.rows()}, {"2026-09-11", "2026-09-12"})

    async def test_context_survives_extract_thread_and_report_is_readonly(self):
        self.assertEqual(self.rows(), [])
        self.assertFalse(self.path.exists())
        ctx = CallerContext(interface="agent", agent_name="analyst", task_id="t42",
                            scope_type="project", scope_id="p7", request_id="r8")
        def request():
            with budget.paid_request("tavily", "extract", "basic") as charge:
                charge.complete(0.2)
        with caller_scope(ctx), patch.object(budget, "_service", return_value="example.service"):
            await asyncio.to_thread(request)
        row = self.rows("task")[0]
        self.assertEqual((row["service"], row["task_id"], row["scope_id"], row["request_id"]),
                         ("example.service", "t42", "p7", "r8"))
        self.assertEqual(self.rows()[0]["agent"], "analyst")

    async def test_cache_hits_free_but_bypass_and_fallback_cannot_escape_budget(self):
        self.set_cap(0.008)
        client = Mock()
        client.search = AsyncMock(return_value={"results": [{"url": "https://example.com", "content": "evidence"}],
                                                 "usage": {"credits": 1}})
        fallback = AsyncMock()
        with patch.object(tavily, "AsyncTavilyClient", return_value=client), \
             patch.object(search, "credential", return_value="test"), \
             patch.dict(search.os.environ, WEB_SEARCH_PROVIDERS="tavily,brave"), \
             patch.dict(search._PROVIDER_SEARCH, brave=fallback):
            first = await search.execute_web_search("question")
            self.assertEqual(await search.execute_web_search("question"), first)
            self.assertIsInstance(await search.execute_web_search("question", use_cache=False), ToolFailure)
            self.assertIsInstance(await search.execute_web_search("other question"), ToolFailure)
        client.search.assert_awaited_once()
        fallback.assert_not_awaited()
        self.assertFalse(search._PROVIDER_UNAVAILABLE_UNTIL)

    async def test_cancelled_provider_request_keeps_reservation(self):
        entered = asyncio.Event()
        async def hang(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        client = Mock(search=AsyncMock(side_effect=hang))
        with patch.object(tavily, "AsyncTavilyClient", return_value=client), \
             patch.object(search, "credential", return_value="test"):
            task = asyncio.create_task(search._search_tavily("q", max_results=5, search_depth="advanced",
                                                            topic="general", time_range=None))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.rows()[0]["unknown_requests"], 1)
        self.assertEqual(self.rows()[0]["accounted_usd"], 0.016)


if __name__ == "__main__":
    unittest.main()
