"""Search billing regressions; all providers are fake, no paid API calls."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runtime_tools import web_search as search
from tool_gateway.results import ToolFailure


class SearchCacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.now = 1000.0
        self.result = [{"title": "Source", "url": "https://example.com", "content": "Evidence"}]

        async def provider(query, **kwargs):
            self.calls.append((query, kwargs))
            return self.result

        for mock in (
            patch.dict(search.os.environ, {
                "WEB_SEARCH_PROVIDERS": "tavily",
                "WEB_SEARCH_CACHE_TTL_SECONDS": "300",
                "WEB_SEARCH_PROVIDER_COOLDOWN_SECONDS": "0",
            }),
            patch.dict(search._PROVIDER_SEARCH, tavily=provider, brave=provider),
            patch.object(search, "time", SimpleNamespace(monotonic=lambda: self.now)),
        ):
            mock.start()
            self.addCleanup(mock.stop)
        search._SEARCH_CACHE.clear()
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()

    async def asyncTearDown(self):
        await asyncio.sleep(0)
        self.assertFalse(search._SEARCH_INFLIGHT)
        search._SEARCH_CACHE.clear()
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()

    async def test_normalized_repeat_reuses_wrapped_result(self):
        first = await search.execute_web_search("a  query")
        second = await search.execute_web_search(" a query ")
        self.assertEqual(first, second)
        self.assertIn('<external source="web_search:tavily:', second)
        self.assertEqual(len(self.calls), 1)

    async def test_concurrent_duplicates_bill_once(self):
        results = await asyncio.gather(*(search.execute_web_search("same") for _ in range(8)))
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(len(self.calls), 1)

    async def test_cancellation_does_not_cancel_shared_request(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(*args, **kwargs):
            self.calls.append(args)
            entered.set()
            await release.wait()
            return self.result

        with patch.dict(search._PROVIDER_SEARCH, tavily=slow):
            first = asyncio.create_task(search.execute_web_search("same"))
            await entered.wait()
            second = asyncio.create_task(search.execute_web_search("same"))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertIn("Evidence", await second)
        self.assertEqual(len(self.calls), 1)

    async def test_ttl_and_sensitive_searches(self):
        for kwargs, ttl in [({}, 300), ({"topic": "news"}, 60),
                            ({"topic": "finance"}, 60), ({"time_range": "day"}, 60)]:
            search._SEARCH_CACHE.clear()
            self.calls.clear()
            await search.execute_web_search("timed", **kwargs)
            self.now += ttl - 1
            await search.execute_web_search("timed", **kwargs)
            self.assertEqual(len(self.calls), 1)
            self.now += 1
            await search.execute_web_search("timed", **kwargs)
            self.assertEqual(len(self.calls), 2)

    async def test_empty_results_have_short_ttl_and_no_fallback(self):
        self.result = []
        with patch.dict(search.os.environ, WEB_SEARCH_PROVIDERS="tavily,brave"):
            self.assertEqual(await search.execute_web_search("empty"), "No results for: empty")
            await search.execute_web_search("empty")
            self.assertEqual(len(self.calls), 1)
            self.now += 30
            await search.execute_web_search("empty")
            self.assertEqual(len(self.calls), 2)

    async def test_request_parameters_and_provider_order_are_isolated(self):
        await search.execute_web_search("q")
        for kwargs in ({"max_results": 6}, {"search_depth": "advanced"},
                       {"topic": "news"}, {"time_range": "week"}):
            await search.execute_web_search("q", **kwargs)
        with patch.dict(search.os.environ, WEB_SEARCH_PROVIDERS="brave,tavily"):
            result = await search.execute_web_search("q")
            self.assertIn("web_search:brave:", result)
        self.assertEqual(len(self.calls), 6)

    async def test_domain_filters_are_forwarded_and_cache_isolated(self):
        await search.execute_web_search("q")
        await search.execute_web_search("q", include_domains=["EXAMPLE.COM.", "example.com"])
        await search.execute_web_search("q", include_domains=["example.com"])
        await search.execute_web_search("q", exclude_domains=["other.org"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.calls[1][1]["include_domains"], ("example.com",))
        self.assertEqual(self.calls[2][1]["exclude_domains"], ("other.org",))

    async def test_invalid_requests_do_not_spend(self):
        for kwargs in (
            {"query": "x" * 1501},
            {"query": "q", "include_domains": ["https://example.com/path"]},
            {"query": "q", "include_domains": ["a.example.com"], "exclude_domains": ["example.com"]},
        ):
            self.assertIsInstance(await search.execute_web_search(**kwargs), ToolFailure)
        self.assertEqual(self.calls, [])

    async def test_long_query_is_not_truncated_for_tavily(self):
        query = "historical " * 55 + "1923 appointment date"
        await search.execute_web_search(query)
        self.assertEqual(self.calls[0][0], query)

    async def test_filtered_results_never_leak_other_domains(self):
        self.result = [
            {"url": "https://docs.example.com/page", "content": "allowed"},
            {"url": "https://example.com.evil.org/page", "content": "wrong domain"},
            {"url": "https://blocked.example.com/page", "content": "excluded"},
        ]
        result = await search.execute_web_search(
            "q", include_domains=["example.com"], exclude_domains=["blocked.example.com"],
        )
        self.assertIn("allowed", result)
        self.assertNotIn("wrong domain", result)
        self.assertNotIn("excluded", result)

    async def test_bypass_and_disabled_cache_always_call_provider(self):
        await search.execute_web_search("q")
        await search.execute_web_search("q", use_cache=False)
        with patch.dict(search.os.environ, WEB_SEARCH_CACHE_TTL_SECONDS="0"):
            await search.execute_web_search("q")
            await search.execute_web_search("q")
        self.assertEqual(len(self.calls), 4)

    async def test_failures_are_not_cached(self):
        async def failed(*args, **kwargs):
            raise search.SearchProviderError("tavily", "unavailable")

        with patch.dict(search._PROVIDER_SEARCH, tavily=failed):
            self.assertIsInstance(await search.execute_web_search("q"), ToolFailure)
        self.assertFalse(search._SEARCH_CACHE)
        self.assertIn("Evidence", await search.execute_web_search("q"))

    async def test_lru_is_bounded(self):
        with patch.object(search, "_CACHE_MAX_ENTRIES", 2):
            for query in ("a", "b", "a", "c", "b"):
                await search.execute_web_search(query)
            self.assertEqual(len(search._SEARCH_CACHE), 2)
            self.assertEqual([q for q, _ in self.calls], ["a", "b", "c", "b"])

    def test_config_is_bounded(self):
        for value, expected in [("bad", 300), ("nan", 300), ("inf", 300),
                                ("900", 300), ("-1", 0), ("20", 20)]:
            with patch.dict(search.os.environ, WEB_SEARCH_CACHE_TTL_SECONDS=value):
                self.assertEqual(search._cache_ttl("general", None), expected)
                self.assertLessEqual(search._cache_ttl("news", None), 60)


if __name__ == "__main__":
    unittest.main()
