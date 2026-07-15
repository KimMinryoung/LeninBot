"""Free Wikipedia runtime tools (MediaWiki API — no Tavily credits).

Added to cut Tavily spend: the CommuLingo curator's research is mostly
biographical facts that live on Wikipedia, so give it direct, free access
instead of routing every lookup through the paid web_search tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_UA = "LeninBot-wiki/1.0 (https://cyber-lenin.com; minryoung93@gmail.com)"
_LANG_RE = re.compile(r"^[a-z]{2,12}(-[a-z0-9]{2,12})?$")

WIKI_TOOLS = [
    {
        "name": "wiki_search",
        "description": (
            "Search Wikipedia article titles and text (free MediaWiki API — costs nothing, "
            "unlike web_search). Use FIRST for people, organizations, and historical events. "
            "Set language per subject: 'ru' for Russian/Soviet figures, 'en' as the general "
            "default, 'ko' for Korean topics. Returns titles, snippets, and article URLs; "
            "open a result with wiki_get."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (names in native script work best)."},
                "language": {
                    "type": "string",
                    "description": "Wikipedia language code, e.g. 'ru', 'en', 'ko', 'de', 'fr'. Default 'en'.",
                    "default": "en",
                },
                "limit": {"type": "integer", "description": "Number of results (1-10, default 5).", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "wiki_get",
        "description": (
            "Fetch a Wikipedia article as plain text by exact title (free MediaWiki API — "
            "costs nothing, unlike fetch_url which may bill a paid extractor). Follows "
            "redirects. Returns a character slice; use offset from the next hint to paginate "
            "long articles. Prefer this over fetch_url for any *.wikipedia.org page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact article title, e.g. 'Клаудия Джонс'."},
                "language": {
                    "type": "string",
                    "description": "Wikipedia language code the title belongs to. Default 'en'.",
                    "default": "en",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters of article text to return (1,000-50,000, default 12,000).",
                    "default": 12000,
                },
                "offset": {
                    "type": "integer",
                    "description": "0-indexed character offset into the article text. Default 0.",
                    "default": 0,
                },
            },
            "required": ["title"],
        },
    },
]


def _api_url(language: str) -> tuple[str, str]:
    lang = (language or "en").strip().lower()
    if not _LANG_RE.match(lang):
        lang = "en"
    return f"https://{lang}.wikipedia.org/w/api.php", lang


def _api_get(language: str, params: dict) -> tuple[dict, str]:
    base, lang = _api_url(language)
    qs = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(f"{base}?{qs}", headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8")), lang


_TAG_RE = re.compile(r"<[^>]+>")


async def _exec_wiki_search(query: str, language: str = "en", limit: int = 5) -> str:
    try:
        limit = max(1, min(int(limit), 10))
    except (TypeError, ValueError):
        limit = 5
    try:
        data, lang = await asyncio.to_thread(
            _api_get,
            language,
            {"action": "query", "list": "search", "srsearch": query, "srlimit": limit},
        )
        results = (data.get("query") or {}).get("search") or []
        if not results:
            return f"No {lang}.wikipedia.org results for: {query}"
        from provenance.runtime import _wrap_external

        lines = []
        for r in results:
            title = r.get("title", "")
            snippet = _TAG_RE.sub("", r.get("snippet", ""))
            url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            lines.append(f"### {title}\n{url}\n{snippet}")
        body = _wrap_external("\n\n".join(lines), f"wiki_search:{lang}:{query}")
        return f"[wiki_search] {lang}.wikipedia.org — open a result with wiki_get(title=..., language='{lang}')\n\n{body}"
    except Exception as e:
        logger.error("wiki_search error: %s", e)
        return f"Wikipedia search failed: {e}"


async def _exec_wiki_get(
    title: str, language: str = "en", max_chars: int = 12000, offset: int = 0
) -> str:
    try:
        max_chars = max(1000, min(int(max_chars), 50000))
    except (TypeError, ValueError):
        max_chars = 12000
    try:
        start = max(0, int(offset or 0))
    except (TypeError, ValueError):
        start = 0
    try:
        data, lang = await asyncio.to_thread(
            _api_get,
            language,
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
            },
        )
        pages = (data.get("query") or {}).get("pages") or []
        page = pages[0] if pages else {}
        if not page or page.get("missing"):
            return (
                f"No {lang}.wikipedia.org article titled '{title}'. "
                f"Use wiki_search to find the exact title first."
            )
        text = page.get("extract") or ""
        if not text:
            return f"Article '{page.get('title', title)}' has no extractable text."
        canonical = page.get("title", title)
        url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(canonical.replace(' ', '_'))}"
        if start >= len(text):
            return (
                f"[wiki_get] {url}\n"
                f"Error: offset {start} is beyond the article ({len(text)} chars)."
            )
        end = min(len(text), start + max_chars)
        more = len(text) > end
        next_hint = (
            f"\nnext: wiki_get(title='{canonical}', language='{lang}', offset={end}, max_chars={max_chars})"
            if more
            else ""
        )
        from provenance.runtime import _wrap_external

        header = (
            f"[wiki_get] {url}\n"
            f"chars {start}:{end} of {len(text)} truncated={more}{next_hint}\n\n"
        )
        return header + _wrap_external(text[start:end], f"url:{url}")
    except Exception as e:
        logger.error("wiki_get error: %s", e)
        return f"Wikipedia fetch failed: {e}"


WIKI_TOOL_HANDLERS = {
    "wiki_search": _exec_wiki_search,
    "wiki_get": _exec_wiki_get,
}
