"""Provider request contracts; no network or credentials required."""

import unittest
from unittest.mock import AsyncMock, patch

import httpx
import tavily

from runtime_tools import web_search as search


class QueryTests(unittest.IsolatedAsyncioTestCase):
    def test_search_guidance_survives_provider_compaction(self):
        from runtime_tools.registry import TOOLS
        from tool_gateway.dispatcher import compact_tool_definitions

        # Web and Telegram share this registry. Both model payload formats must
        # retain every instruction, including nested parameter descriptions.
        tool = next(t for t in TOOLS if t["name"] == "web_search")
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for payload in (tool, openai_tool):
            with self.subTest(format="openai" if "function" in payload else "anthropic"):
                self.assertEqual(compact_tool_definitions([payload]), [payload])

    async def test_shared_schema_and_handler_forward_filters(self):
        import jsonschema
        import runtime_tools.registry as registry

        schema = next(t["input_schema"] for t in registry.TOOLS if t["name"] == "web_search")
        jsonschema.Draft7Validator.check_schema(schema)
        request = {
            "query": "Python 3.12 asyncio.timeout cancellation",
            "include_domains": ["docs.python.org"],
            "exclude_domains": ["old.python.org"], "use_cache": False,
        }
        jsonschema.validate(request, schema)
        with patch.object(registry, "execute_web_search", new_callable=AsyncMock, return_value="ok") as handler:
            self.assertEqual(await registry._exec_web_search(**request), "ok")
            for field, value in request.items():
                self.assertEqual(handler.call_args.kwargs[field], value)

    def test_domain_validation(self):
        self.assertEqual(search._normalize_domains([" Docs.Python.org. "]), ("docs.python.org",))
        for invalid in (["*.org"], ["a.com/path"], ["a.com OR site:b.com"], ["a.com:443"],
                        ["bad..com"], [123], "example.com", ["a.com"] * 11):
            with self.assertRaises(ValueError):
                search._normalize_domains(invalid)

    def test_brave_query_preserves_constraints(self):
        query = search._brave_query('"Python 3.12" timeout', ("docs.python.org", "python.org"), ("old.python.org",))
        self.assertEqual(query, '"Python 3.12" timeout (site:docs.python.org OR site:python.org) NOT site:old.python.org')
        for long_query in ("x" * 401, "word " * 51):
            with self.assertRaises(search.SearchRequestUnsupported):
                search._brave_query(long_query, (), ())
        with self.assertRaises(search.SearchRequestUnsupported):
            search._brave_query("x" * 390, ("example.com",), ())

    async def test_tavily_receives_native_domain_parameters(self):
        client = AsyncMock()
        client.search.return_value = {"results": []}
        with patch.object(tavily, "AsyncTavilyClient", return_value=client), patch.object(search, "get_secret", return_value="test"):
            await search._search_tavily(
                "Python 3.12 timeout", max_results=5, search_depth="basic", topic="general",
                time_range=None, include_domains=("docs.python.org",), exclude_domains=("old.python.org",),
            )
        args, kwargs = client.search.call_args
        self.assertEqual(args, ("Python 3.12 timeout",))
        self.assertEqual(kwargs["include_domains"], ["docs.python.org"])
        self.assertEqual(kwargs["exclude_domains"], ["old.python.org"])
        self.assertFalse(kwargs["auto_parameters"])

    async def test_brave_web_and_news_receive_operators(self):
        for topic, url in (("general", search._BRAVE_WEB_URL), ("news", search._BRAVE_NEWS_URL)):
            client = AsyncMock()
            client.get.return_value = httpx.Response(200, json={}, request=httpx.Request("GET", url))
            with patch.object(search.httpx, "AsyncClient") as factory, patch.object(search, "get_secret", return_value="test"):
                factory.return_value.__aenter__.return_value = client
                await search._search_brave(
                    "elections", max_results=5, search_depth="basic", topic=topic,
                    time_range="week", include_domains=("reuters.com",), exclude_domains=("old.reuters.com",),
                )
            self.assertEqual(client.get.call_args.args, (url,))
            self.assertEqual(client.get.call_args.kwargs["params"]["q"], "elections (site:reuters.com) NOT site:old.reuters.com")

    async def test_unsupported_input_does_not_disable_provider(self):
        with patch.dict(search._PROVIDER_UNAVAILABLE_UNTIL, {}, clear=True), patch.dict(
            search._PROVIDER_SEARCH,
            brave=AsyncMock(side_effect=search.SearchRequestUnsupported("brave", "too long")),
            tavily=AsyncMock(return_value=[]),
        ):
            result = await search._execute_provider_chain(
                "q", providers=("brave", "tavily"), max_results=5,
                search_depth="basic", topic="general", time_range=None,
            )
            self.assertEqual(result, "No results for: q")
            self.assertFalse(search._PROVIDER_UNAVAILABLE_UNTIL)


if __name__ == "__main__":
    unittest.main()
