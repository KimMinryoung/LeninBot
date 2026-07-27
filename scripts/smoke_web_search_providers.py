#!/usr/bin/env python3
"""Hermetic checks for the Tavily/Brave web-search provider chain."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _check_fallback_and_circuit() -> None:
    import runtime_tools.web_search as search

    calls: list[str] = []

    async def failed_tavily(*_args, **_kwargs):
        calls.append("tavily")
        raise search.SearchProviderError(
            "tavily",
            "pay-as-you-go limit exceeded",
            status_code=433,
        )

    async def working_brave(*_args, **_kwargs):
        calls.append("brave")
        return [{"title": "Result", "url": "https://example.com", "description": "Useful context"}]

    original = dict(search._PROVIDER_SEARCH)
    original_breakers = dict(search._PROVIDER_UNAVAILABLE_UNTIL)
    previous_order = os.environ.get("WEB_SEARCH_PROVIDERS")
    try:
        os.environ["WEB_SEARCH_PROVIDERS"] = "tavily,brave"
        search._PROVIDER_SEARCH.update(tavily=failed_tavily, brave=working_brave)
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()

        result = await search.execute_web_search("test query")
        assert "Useful context" in result
        assert calls == ["tavily", "brave"]

        calls.clear()
        result = await search.execute_web_search("second query")
        assert "Useful context" in result
        assert calls == ["brave"], "open Tavily circuit should prevent repeated paid failures"
    finally:
        search._PROVIDER_SEARCH.clear()
        search._PROVIDER_SEARCH.update(original)
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()
        search._PROVIDER_UNAVAILABLE_UNTIL.update(original_breakers)
        if previous_order is None:
            os.environ.pop("WEB_SEARCH_PROVIDERS", None)
        else:
            os.environ["WEB_SEARCH_PROVIDERS"] = previous_order


async def _check_empty_result_does_not_double_spend() -> None:
    import runtime_tools.web_search as search

    calls: list[str] = []

    async def empty_tavily(*_args, **_kwargs):
        calls.append("tavily")
        return []

    async def unexpected_brave(*_args, **_kwargs):
        calls.append("brave")
        return []

    original = dict(search._PROVIDER_SEARCH)
    previous_order = os.environ.get("WEB_SEARCH_PROVIDERS")
    try:
        os.environ["WEB_SEARCH_PROVIDERS"] = "tavily,brave"
        search._PROVIDER_SEARCH.update(tavily=empty_tavily, brave=unexpected_brave)
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()
        result = await search.execute_web_search("no match")
        assert result == "No results for: no match"
        assert calls == ["tavily"]
    finally:
        search._PROVIDER_SEARCH.clear()
        search._PROVIDER_SEARCH.update(original)
        search._PROVIDER_UNAVAILABLE_UNTIL.clear()
        if previous_order is None:
            os.environ.pop("WEB_SEARCH_PROVIDERS", None)
        else:
            os.environ["WEB_SEARCH_PROVIDERS"] = previous_order


async def _check_tavily_cost_controls() -> None:
    import tavily
    import runtime_tools.web_search as search

    captured: dict = {}

    class FakeClient:
        def __init__(self, api_key: str):
            assert api_key == "test-key"

        async def search(self, query: str, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return {"results": [], "usage": {"credits": 2}}

    original_client = tavily.AsyncTavilyClient
    original_get_secret = search.get_secret
    try:
        tavily.AsyncTavilyClient = FakeClient
        search.get_secret = lambda name, default="": "test-key" if name == "TAVILY_API_KEY" else default
        await search._search_tavily(
            "focused question",
            max_results=5,
            search_depth="advanced",
            topic="general",
            time_range=None,
        )
        assert captured["include_answer"] is False
        assert captured["include_raw_content"] is False
        assert captured["auto_parameters"] is False
        assert captured["include_usage"] is True
        assert captured["chunks_per_source"] == 2
    finally:
        tavily.AsyncTavilyClient = original_client
        search.get_secret = original_get_secret


def _check_brave_response_shapes() -> None:
    import runtime_tools.web_search as search

    news = search._brave_results(
        {"type": "news", "results": [{"title": "News result"}]},
        topic="news",
    )
    web = search._brave_results(
        {"type": "search", "web": {"results": [{"title": "Web result"}]}},
        topic="general",
    )
    assert news == [{"title": "News result"}]
    assert web == [{"title": "Web result"}]


async def main() -> None:
    await _check_fallback_and_circuit()
    await _check_empty_result_does_not_double_spend()
    await _check_tavily_cost_controls()
    _check_brave_response_shapes()
    print("web search provider smoke checks passed")


if __name__ == "__main__":
    asyncio.run(main())
