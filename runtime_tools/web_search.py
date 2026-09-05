"""Cost-aware multi-provider web search for runtime agents.

Tavily remains the default because its snippets are tuned for LLM consumption.
Brave Search is an independent fallback (or primary when configured first).
Provider failures open a short process-local circuit so an agent loop does not
pay for repeated calls to a provider that has already reported an auth, quota,
rate-limit, or transport failure.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from provenance.runtime import _wrap_external
from secrets_loader import get_secret
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDERS = ("tavily", "brave")
_VALID_PROVIDERS = frozenset(_DEFAULT_PROVIDERS)
_BRAVE_WEB_URL = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"
_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
_PROVIDER_UNAVAILABLE_UNTIL: dict[str, float] = {}
# Bounded, process-local storage of already formatted public search results.
# Include the complete effective request and provider order in each key.
_SEARCH_CACHE: OrderedDict[tuple, tuple[float, str]] = OrderedDict()
_SEARCH_INFLIGHT: dict[tuple, asyncio.Task[str]] = {}
_CACHE_MAX_ENTRIES = 256


def _cache_ttl(topic: str, time_range: str | None) -> float:
    try:
        ttl = float(os.getenv("WEB_SEARCH_CACHE_TTL_SECONDS", "300"))
        if not math.isfinite(ttl):
            ttl = 300.0
    except ValueError:
        ttl = 300.0
    # Recency-sensitive searches must not inherit the general-purpose TTL.
    cap = 60.0 if topic in {"news", "finance"} or time_range == "day" else 300.0
    return max(0.0, min(ttl, cap))


async def _cached_search(key: tuple, ttl: float, search: Callable[[], Awaitable[str]]) -> str:
    now = time.monotonic()
    for expired in [k for k, (until, _) in _SEARCH_CACHE.items() if until <= now]:
        del _SEARCH_CACHE[expired]
    cached = _SEARCH_CACHE.get(key)
    if cached is not None:
        _SEARCH_CACHE.move_to_end(key)
        logger.info("web_search cache=hit provider_order=%s", key[0])
        return cached[1]

    # Tasks belong to an event loop; never await one from a different loop.
    inflight_key = (asyncio.get_running_loop(), key)
    task = _SEARCH_INFLIGHT.get(inflight_key)
    if task is None:
        async def run() -> str:
            result = await search()
            if not isinstance(result, ToolFailure):
                effective_ttl = min(ttl, 30.0) if result.startswith("No results for:") else ttl
                _SEARCH_CACHE[key] = (time.monotonic() + effective_ttl, result)
                _SEARCH_CACHE.move_to_end(key)
                while len(_SEARCH_CACHE) > _CACHE_MAX_ENTRIES:
                    _SEARCH_CACHE.popitem(last=False)
            return result

        task = asyncio.create_task(run())
        _SEARCH_INFLIGHT[inflight_key] = task

        def finished(done: asyncio.Task[str]) -> None:
            _SEARCH_INFLIGHT.pop(inflight_key, None)
            # Retrieve exceptions even if every original caller was cancelled.
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        logger.info("web_search cache=miss provider_order=%s", key[0])
    else:
        logger.info("web_search cache=coalesced provider_order=%s", key[0])
    # One caller's timeout must not cancel a paid request shared by others.
    return await asyncio.shield(task)


class SearchProviderError(RuntimeError):
    """A sanitized provider error suitable for fallback and user reporting."""

    def __init__(self, provider: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class SearchRequestUnsupported(SearchProviderError):
    """This request cannot be represented without losing search constraints."""


def _provider_order() -> tuple[str, ...]:
    raw = os.getenv("WEB_SEARCH_PROVIDERS", ",".join(_DEFAULT_PROVIDERS))
    ordered: list[str] = []
    for item in raw.split(","):
        provider = item.strip().lower()
        if provider in _VALID_PROVIDERS and provider not in ordered:
            ordered.append(provider)
    return tuple(ordered or _DEFAULT_PROVIDERS)


def _cooldown_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("WEB_SEARCH_PROVIDER_COOLDOWN_SECONDS", "300")))
    except ValueError:
        return 300.0


def _open_circuit(provider: str) -> None:
    _PROVIDER_UNAVAILABLE_UNTIL[provider] = time.monotonic() + _cooldown_seconds()


def _circuit_is_open(provider: str) -> bool:
    until = _PROVIDER_UNAVAILABLE_UNTIL.get(provider, 0.0)
    if until <= time.monotonic():
        _PROVIDER_UNAVAILABLE_UNTIL.pop(provider, None)
        return False
    return True


def _normalize_query(query: str) -> str:
    # Preserve names, quotes, dates and trailing constraints. Never truncate.
    return " ".join(str(query or "").split())


def _normalize_domains(domains: list[str] | None) -> tuple[str, ...]:
    if domains is None:
        return ()
    if not isinstance(domains, list) or len(domains) > 10:
        raise ValueError("Domain filters must be lists of at most 10 bare hostnames.")
    normalized = set()
    for domain in domains:
        if not isinstance(domain, str):
            raise ValueError("Domain filters must contain bare hostnames.")
        try:
            host = domain.strip().rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise ValueError("Invalid domain hostname.") from None
        labels = host.split(".")
        if len(host) > 253 or len(labels) < 2 or any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in labels
        ):
            raise ValueError("Use bare domain names, e.g. docs.python.org; no URLs, paths or wildcards.")
        normalized.add(host)
    return tuple(sorted(normalized))


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _filter_domains(results: list[dict[str, Any]], include: tuple[str, ...], exclude: tuple[str, ...]) -> list[dict[str, Any]]:
    if not include and not exclude:
        return results
    filtered = []
    for result in results:
        try:
            host = (urlsplit(str(result.get("url") or "")).hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
        except (ValueError, UnicodeError):
            continue
        if host and (not include or any(_domain_matches(host, d) for d in include)) and not any(
            _domain_matches(host, d) for d in exclude
        ):
            filtered.append(result)
    return filtered


def _brave_query(query: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> str:
    parts = [query]
    if include:
        parts.append("(" + " OR ".join(f"site:{domain}" for domain in include) + ")")
    parts.extend(f"NOT site:{domain}" for domain in exclude)
    effective = " ".join(parts)
    if len(effective) > 400 or len(effective.split()) > 50:
        raise SearchRequestUnsupported(
            "brave", "Query plus domain filters exceeds Brave's 400-character/50-word limit; "
            "shorten the query or domain lists without dropping the fact being checked.",
        )
    return effective


def _exception_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def _result_content(result: dict[str, Any], *, cap: int) -> str:
    parts = [str(result.get("content") or result.get("description") or "").strip()]
    extras = result.get("extra_snippets") or []
    if isinstance(extras, list):
        parts.extend(str(item).strip() for item in extras if str(item).strip())
    unique = list(dict.fromkeys(part for part in parts if part))
    return "\n".join(unique)[:cap]


def _brave_results(payload: dict[str, Any], *, topic: str) -> list[dict[str, Any]]:
    # Dedicated News Search returns a top-level `results` list; Web Search
    # nests ordinary results under `web.results`.
    raw = (
        payload.get("results")
        if topic == "news"
        else (payload.get("web") or {}).get("results")
    ) or []
    return [item for item in raw if isinstance(item, dict)]


def _format_results(
    provider: str,
    query: str,
    results: list[dict[str, Any]],
    *,
    advanced: bool,
) -> str:
    if not results:
        return f"No results for: {query}"
    snippet_cap = 1000 if advanced else 500
    lines: list[str] = []
    for result in results:
        title = str(result.get("title") or "").strip()
        url = str(result.get("url") or "").strip()
        content = _result_content(result, cap=snippet_cap)
        published = str(
            result.get("published_date")
            or result.get("age")
            or result.get("page_age")
            or ""
        ).strip()
        header = f"### {title}" + (f" ({published})" if published else "")
        lines.append(f"{header}\n{url}\n{content}".rstrip())
    return _wrap_external(
        "\n\n".join(lines),
        f"web_search:{provider}:{query}",
    )


async def _search_tavily(
    query: str,
    *,
    max_results: int,
    search_depth: str,
    topic: str,
    time_range: str | None,
    include_domains: tuple[str, ...] = (),
    exclude_domains: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    api_key = get_secret("TAVILY_API_KEY", "") or ""
    if not api_key:
        raise SearchProviderError("tavily", "TAVILY_API_KEY is not configured")
    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=api_key)
        kwargs: dict[str, Any] = {
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": False,
            "include_raw_content": False,
            "include_usage": True,
            # Tavily may otherwise promote an apparently difficult query to
            # advanced search (2 credits). Explicit depth always wins, but
            # disabling auto parameters makes the cost boundary unambiguous.
            "auto_parameters": False,
        }
        if time_range:
            kwargs["time_range"] = time_range
        if include_domains:
            kwargs["include_domains"] = list(include_domains)
        if exclude_domains:
            kwargs["exclude_domains"] = list(exclude_domains)
        if search_depth == "advanced":
            # Focused chunks reduce downstream LLM context without an extra
            # Extract call. Advanced itself still costs two Tavily credits.
            kwargs["chunks_per_source"] = 2
        response = await client.search(query, **kwargs)
    except SearchProviderError:
        raise
    except Exception as exc:
        raise SearchProviderError(
            "tavily",
            str(exc),
            status_code=_exception_status(exc),
        ) from exc

    usage = response.get("usage") or {}
    credits = usage.get("credits", "unknown") if isinstance(usage, dict) else usage
    logger.info(
        "web_search provider=tavily depth=%s results=%d credits=%s",
        search_depth,
        len(response.get("results") or []),
        credits,
    )
    return [item for item in response.get("results", []) if isinstance(item, dict)]


async def _search_brave(
    query: str,
    *,
    max_results: int,
    search_depth: str,
    topic: str,
    time_range: str | None,
    include_domains: tuple[str, ...] = (),
    exclude_domains: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    query = _brave_query(query, include_domains, exclude_domains)
    api_key = get_secret("BRAVE_SEARCH_API_KEY", "") or ""
    if not api_key:
        raise SearchProviderError("brave", "BRAVE_SEARCH_API_KEY is not configured")

    url = _BRAVE_NEWS_URL if topic == "news" else _BRAVE_WEB_URL
    params: dict[str, Any] = {
        "q": query,
        "count": max_results,
        "safesearch": "moderate",
    }
    if search_depth == "advanced":
        params["extra_snippets"] = "true"
    if time_range:
        params["freshness"] = _FRESHNESS[time_range]
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise SearchProviderError(
            "brave",
            str(exc),
            status_code=_exception_status(exc),
        ) from exc

    results = _brave_results(payload, topic=topic)
    logger.info(
        "web_search provider=brave depth=%s results=%d remaining=%s reset=%s",
        search_depth,
        len(results),
        response.headers.get("X-RateLimit-Remaining", "unknown"),
        response.headers.get("X-RateLimit-Reset", "unknown"),
    )
    return results


_PROVIDER_SEARCH: dict[
    str,
    Callable[..., Awaitable[list[dict[str, Any]]]],
] = {
    "tavily": _search_tavily,
    "brave": _search_brave,
}


async def execute_web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    time_range: str | None = None,
    use_cache: bool = True,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    """Search with the configured providers, falling back once on failures."""
    query = _normalize_query(query)
    if not query:
        return "Error: search query is empty."
    if len(query) > 1500:
        return ToolFailure("Search query exceeds 1500 characters. Rewrite as one focused question, preserving names, dates and the missing fact; do not send the whole task prompt.")
    try:
        include = _normalize_domains(include_domains)
        exclude = _normalize_domains(exclude_domains)
    except ValueError as exc:
        return ToolFailure(str(exc))
    if include and all(any(_domain_matches(d, e) for e in exclude) for d in include):
        return ToolFailure("All included domains are excluded. Correct the domain filters before searching.")
    max_results = max(1, min(int(max_results), 10))
    if search_depth not in {"ultra-fast", "fast", "basic", "advanced"}:
        search_depth = "basic"
    if topic not in {"general", "news", "finance"}:
        topic = "general"
    if time_range not in _FRESHNESS:
        time_range = None

    providers = _provider_order()

    async def search() -> str:
        return await _execute_provider_chain(
            query, providers=providers, max_results=max_results,
            search_depth=search_depth, topic=topic, time_range=time_range,
            include_domains=include, exclude_domains=exclude,
        )

    ttl = _cache_ttl(topic, time_range)
    if not use_cache or ttl <= 0:
        return await search()
    key = (providers, query, max_results, search_depth, topic, time_range, ttl, include, exclude)
    return await _cached_search(key, ttl, search)


async def _execute_provider_chain(
    query: str, *, providers: tuple[str, ...], max_results: int,
    search_depth: str, topic: str, time_range: str | None,
    include_domains: tuple[str, ...] = (), exclude_domains: tuple[str, ...] = (),
) -> str:
    errors: list[str] = []
    for provider in providers:
        if _circuit_is_open(provider):
            errors.append(f"{provider}: temporarily disabled after an earlier failure")
            continue
        try:
            results = await _PROVIDER_SEARCH[provider](
                query,
                max_results=max_results,
                search_depth=search_depth,
                topic=topic,
                time_range=time_range,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
            results = _filter_domains(results, include_domains, exclude_domains)
            # Empty results are a valid, billed response. Do not double-spend
            # by querying the fallback provider for the same empty result.
            return _format_results(
                provider,
                query,
                results,
                advanced=search_depth == "advanced",
            )
        except SearchRequestUnsupported as exc:
            # Input limitations are not provider outages; don't open a circuit.
            errors.append(f"{provider}: {exc}")
        except SearchProviderError as exc:
            _open_circuit(provider)
            status = f" HTTP {exc.status_code}" if exc.status_code else ""
            logger.warning(
                "web_search provider=%s failed%s; trying fallback: %s",
                provider,
                status,
                exc,
            )
            errors.append(f"{provider}{status}: {exc}")

    return ToolFailure("Web search failed across configured providers: " + "; ".join(errors))
