"""Gateway custody, validation, attribution and keyless transport, no paid calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import tavily

from web_gateway import app as server, budget, client, credentials, search
from tool_gateway.results import ToolFailure


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "policy.json"
        self.config.write_text(json.dumps({"daily_budget_usd": 0.008, "tavily_credit_usd": 0.008,
                                          "brave_search_usd": 0.005}))
        self.cred_dir = self.root / "creds"
        self.cred_dir.mkdir()
        (self.cred_dir / "tavily_api_key").write_text("fake-gateway-key")
        for mock in (patch.object(budget, "CONFIG_PATH", self.config),
                     patch.object(budget, "ROOT", self.root),
                     patch.object(budget, "STORE_PATH", self.root / "private/usage.sqlite3"),
                     patch.dict("os.environ", {"CREDENTIALS_DIRECTORY": str(self.cred_dir),
                                               "WEB_SEARCH_PROVIDERS": "tavily,brave"})):
            mock.start()
            self.addCleanup(mock.stop)
        search._SEARCH_CACHE.clear()
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()
        self.addCleanup(search._SEARCH_CACHE.clear)
        self.addCleanup(search._PROVIDER_UNAVAILABLE_UNTIL.clear)

    def transport(self, host="127.0.0.1"):
        return httpx.ASGITransport(app=server.app, client=(host, 50000))

    async def test_different_clients_share_cache_and_server_budget(self):
        sdk = Mock(search=AsyncMock(return_value={"results": [{"url": "https://example.com", "content": "evidence"}],
                                                 "usage": {"credits": 1}}))
        with patch.object(tavily, "AsyncTavilyClient", return_value=sdk) as factory:
            async with httpx.AsyncClient(transport=self.transport(), base_url=client.BASE_URL) as http:
                first = await http.post("/search", json={"arguments": {"query": "question"},
                    "caller": {"service": "first.service", "task_id": "task-1"}})
                second = await http.post("/search", json={"arguments": {"query": "question"},
                    "caller": {"service": "second.service", "task_id": "task-2"}})
                denied = await http.post("/search", json={"arguments": {"query": "new question", "use_cache": False}})
                usage = (await http.get("/usage", params={"by": "task"})).json()
        self.assertEqual(first.json(), second.json())
        self.assertFalse(first.json()["error"])
        self.assertTrue(denied.json()["error"])
        self.assertIn("budget exhausted", denied.json()["result"])
        self.assertEqual(usage["accounted_usd"], 0.008)
        self.assertEqual(usage["rows"][0]["task_id"], "task-1")
        self.assertEqual(usage["rows"][0]["service"], "first.service")
        sdk.search.assert_awaited_once()
        self.assertEqual(factory.call_args.kwargs["api_key"], "fake-gateway-key")
        self.assertIsNone(budget.usage_identity.get())

    async def test_fixed_routes_and_strict_payloads_cannot_bypass_policy(self):
        with patch.object(tavily, "AsyncTavilyClient") as sdk:
            async with httpx.AsyncClient(transport=self.transport(), base_url=client.BASE_URL) as http:
                for body in (
                    {"arguments": {"query": "q", "api_key": "caller-key"}},
                    {"arguments": {"query": "q"}, "budget_usd": 100},
                    {"arguments": {"query": "q", "max_results": 999}},
                    {"arguments": {"query": "q", "search_depth": "research"}},
                ):
                    self.assertEqual((await http.post("/search", json=body)).status_code, 422)
                self.assertEqual((await http.post("/tavily/research", json={})).status_code, 404)
                self.assertEqual((await http.post("/search", content=b"x" * 32769)).status_code, 413)
                self.assertEqual((await http.post("/extract", json={"url": "http://127.0.0.1/admin"})).status_code, 422)
        sdk.assert_not_called()
        self.assertFalse(budget.STORE_PATH.exists())

    async def test_extract_enforces_budget_and_records_reported_credits(self):
        sdk = Mock(extract=AsyncMock(return_value={"results": [{"raw_content": "source text"}], "usage": {"credits": 0.2}}))
        with patch.object(server, "AsyncTavilyClient", return_value=sdk), \
             patch.object(server, "validate_public_http_url", side_effect=lambda url: url):
            async with httpx.AsyncClient(transport=self.transport(), base_url=client.BASE_URL) as http:
                response = (await http.post("/extract", json={"url": "https://example.com", "caller": {"service": "writer"}})).json()
                denied = (await http.post("/extract", json={"url": "https://example.com"})).json()
        self.assertEqual(response["results"][0]["raw_content"], "source text")
        self.assertTrue(denied["error"])
        sdk.extract.assert_awaited_once_with(urls=["https://example.com"], extract_depth="basic", include_usage=True)
        self.assertEqual(budget.report("2000-01-01")[0]["accounted_usd"], 0.0016)

    async def test_remote_and_forwarded_identity_cannot_reach_gateway(self):
        async with httpx.AsyncClient(transport=self.transport("10.0.0.2"), base_url=client.BASE_URL) as http:
            response = await http.get("/health", headers={"X-Forwarded-For": "127.0.0.1"})
        self.assertEqual(response.status_code, 403)

    async def test_health_and_env_keys_do_not_replace_credentials(self):
        (self.cred_dir / "tavily_api_key").unlink()
        with patch.dict("os.environ", TAVILY_API_KEY="fake-env-key"):
            self.assertEqual(credentials.credential("TAVILY_API_KEY"), "")
            async with httpx.AsyncClient(transport=self.transport(), base_url=client.BASE_URL) as http:
                response = await http.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("fake-env-key", response.text)

    def test_legacy_import_keeps_spend_without_double_counting(self):
        legacy = self.root / "data/web_research_usage.sqlite3"
        with patch.object(budget, "STORE_PATH", legacy):
            with budget.paid_request("tavily", "search", "basic") as charge:
                charge.complete(1)
        budget.import_legacy_usage()
        budget.import_legacy_usage()
        self.assertEqual(budget.report("2000-01-01")[0]["requests"], 1)
        with self.assertRaises(budget.PaidWebBudgetError):
            budget.reserve("tavily", "search", "basic")

    async def test_keyless_client_keeps_failure_type_and_never_retries_directly(self):
        local = Mock()
        local.post = AsyncMock(return_value=httpx.Response(200, json={"result": "budget exhausted", "error": True}))
        with patch.object(client.httpx, "AsyncClient") as factory, patch.object(tavily, "AsyncTavilyClient") as sdk:
            factory.return_value.__aenter__.return_value = local
            result = await client.search({"query": "q"})
            self.assertIsInstance(result, ToolFailure)
            self.assertEqual(local.post.call_args.args, ("http://127.0.0.1:8111/search",))
            self.assertNotIn("api_key", json.dumps(local.post.call_args.kwargs))
            local.post.side_effect = httpx.ConnectError("down")
            result = await client.search({"query": "q"})
            self.assertIsInstance(result, ToolFailure)
            self.assertEqual(local.post.await_count, 2)
            sdk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
