"""Stable runtime tool interface; all paid work is owned by the web gateway."""
from web_gateway.client import search


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
    return await search({
        "query": query, "max_results": max_results, "search_depth": search_depth,
        "topic": topic, "time_range": time_range, "use_cache": use_cache,
        "include_domains": include_domains, "exclude_domains": exclude_domains,
    })
