"""vector_search tool: multilingual corpus search with LLM query translation and cross-language merge."""
import asyncio
import json
import logging
import re

from llm.json_utils import extract_json_object as _extract_json_object
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)


def _looks_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7af]", text or ""))


def _looks_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or "")) and not _looks_korean(text)


async def llm_translate_search_query(query: str, target_language: str, layer: str) -> str | None:
    """Best-effort query translation for cross-language corpus recall.

    Model/options: config/llm_call_sites.json ("vector_query_translation")."""
    try:
        from llm.call_registry import generate as _registry_generate

        system = (
            "Translate a vector-search query for Marxist/political document retrieval. "
            "Return only JSON with key translated_query. Preserve names and technical terms."
        )
        user = {
            "query": query,
            "target_language": target_language,
            "target_corpus_layer": layer,
        }
        content = await _registry_generate(
            "vector_query_translation",
            json.dumps(user, ensure_ascii=False),
            system=system,
        )
        if not content:
            return None
        parsed = _extract_json_object(content)
        translated = str((parsed or {}).get("translated_query") or "").strip()
        if translated and translated.lower() != query.lower():
            return translated
    except Exception as e:
        logger.info("vector_search query translation unavailable: %s", e)
    return None


def _doc_dedupe_key(doc) -> tuple[str, str]:
    meta = getattr(doc, "metadata", {}) or {}
    source = str(meta.get("source") or meta.get("public_url") or meta.get("source_url") or meta.get("title") or "")
    chunk = str(meta.get("chunk_index", ""))
    if source:
        return (source, chunk)
    return ("content", str(hash(getattr(doc, "page_content", "") or "")))


def _merge_docs_by_similarity(docs: list, k: int) -> list:
    """Order docs merged from parallel searches by their cosine similarity.

    Scores from the original and translated queries live in the same
    normalized embedding space, so cross-language mismatches (e.g. a Korean
    query against the English corpus) score low and sink naturally.
    """
    def score(doc) -> float:
        try:
            return float((getattr(doc, "metadata", {}) or {}).get("similarity") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return sorted(docs, key=score, reverse=True)[:k]


_AUTHOR_ALIASES = [
    ("joseph stalin", "Stalin"),
    ("j. v. stalin", "Stalin"),
    ("stalin", "Stalin"),
    ("스탈린", "Stalin"),
    ("마오쩌둥", "Mao"),
    ("mao", "Mao"),
    ("마오", "Mao"),
    ("lenin", "Lenin"),
    ("레닌", "Lenin"),
    ("luxemburg", "Rosa Luxemburg"),
    ("룩셈부르크", "Rosa Luxemburg"),
    ("trotsky", "Trotsky"),
    ("트로츠키", "Trotsky"),
    ("gramsci", "Gramsci"),
    ("그람시", "Gramsci"),
]


_TITLE_HINTS = [
    (("national question", "민족 문제", "민족문제"), "National Question"),
    (("chinese revolution", "중국 혁명", "중국혁명"), "Chinese Revolution"),
    (("leninism", "레닌주의"), "Leninism"),
    (("trotskyism", "트로츠키주의"), "Trotskyism"),
]


def _infer_corpus_filters(query: str) -> dict:
    lowered = (query or "").lower()
    filters: dict = {}
    for alias, author in _AUTHOR_ALIASES:
        if _looks_korean(alias):
            found = alias in lowered
        else:
            found = bool(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered))
        if found:
            filters["author"] = author
            break
    year_match = re.search(r"(?<!\d)(18|19|20)\d{2}(?!\d)", query or "")
    if year_match:
        filters["year"] = int(year_match.group(0))
    for hints, title in _TITLE_HINTS:
        if any(hint in lowered for hint in hints):
            filters["title"] = title
            break
    return filters


async def search_corpus_multilingual(
    query: str,
    num_results: int,
    layer: str | None,
    *,
    author: str | None = None,
    title: str | None = None,
    year: int | str | None = None,
    keywords: str | list[str] | None = None,
) -> list:
    from corpus.store import similarity_search

    k = max(1, min(int(num_results or 5), 10))
    filters = _infer_corpus_filters(query)
    relaxable_filters = set(filters)
    if author:
        filters["author"] = author
        relaxable_filters.discard("author")
    if title:
        filters["title"] = title
        relaxable_filters.discard("title")
    if year:
        filters["year"] = year
        relaxable_filters.discard("year")
    if keywords:
        filters["keywords"] = keywords
        relaxable_filters.discard("keywords")
    searches: list[tuple[str, str, str | None]] = [("original", query, layer)]
    if _looks_korean(query) and layer in (None, "core_theory"):
        translated = await llm_translate_search_query(query, "English", "core_theory")
        if translated:
            searches.append(("translated_en", translated, "core_theory"))
    elif _looks_english(query) and layer == "modern_analysis":
        translated = await llm_translate_search_query(query, "Korean", "modern_analysis")
        if translated:
            searches.append(("translated_ko", translated, "modern_analysis"))

    async def run_with(search_filters: dict) -> list:
        if len(searches) == 1:
            return await asyncio.to_thread(
                similarity_search,
                query,
                k,
                layer,
                **search_filters,
            )

        tasks = [
            asyncio.to_thread(
                similarity_search,
                q,
                k * 2,
                search_layer,
                **search_filters,
            )
            for _label, q, search_layer in searches
        ]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        merged = []
        seen: set[tuple[str, str]] = set()
        for batch in batches:
            if isinstance(batch, Exception):
                logger.info("vector_search parallel query failed: %s", batch)
                continue
            for doc in batch:
                key = _doc_dedupe_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
        return _merge_docs_by_similarity(merged, k)

    docs = await run_with(filters)
    if docs:
        return docs

    relaxed = dict(filters)
    for key in ("year", "title"):
        if key not in relaxable_filters:
            continue
        relaxed.pop(key, None)
        docs = await run_with(relaxed)
        if docs:
            logger.info("vector_search relaxed inferred %s filter after empty result", key)
            return docs
    return []


async def exec_vector_search(
    query: str,
    num_results: int = 5,
    layer: str | None = None,
    author: str | None = None,
    title: str | None = None,
    year: int | str | None = None,
    keywords: str | None = None,
) -> str:
    """Execute vector similarity search via chatbot module."""
    try:
        from corpus.store import fetch_corpus_source_context
        docs = await search_corpus_multilingual(
            query,
            num_results,
            layer,
            author=author,
            title=title,
            year=year,
            keywords=keywords,
        )
        if not docs:
            return "No documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[{i}] {meta.get('title', 'Untitled')} — {meta.get('author', 'Unknown')}"
            if meta.get("year"):
                header += f" ({meta['year']})"
            if meta.get("public_url"):
                header += f"\nURL: {meta['public_url']}"
            if meta.get("chunk_count", 1) and int(meta.get("chunk_count", 1)) > 1:
                idx = int(meta.get("chunk_index", 0)) + 1
                header += f"\nChunk: {idx}/{meta.get('chunk_count')}"
            body = doc.page_content
            if (
                meta.get("layer") == "self_produced_analysis"
                and int(meta.get("chunk_count", 1)) > 1
                and meta.get("source")
            ):
                expanded = await asyncio.to_thread(
                    fetch_corpus_source_context,
                    meta.get("source"),
                    center_index=int(meta.get("chunk_index", 0)),
                    window=1,
                    max_chars=9000,
                )
                if expanded and len(expanded) > len(body):
                    body = expanded
                    header += "\nContext: expanded with adjacent chunks from the same public document"
            results.append(f"{header}\n{body}")
        return "\n\n".join(results)
    except Exception as e:
        logger.error("vector_search error: %s", e)
        return ToolFailure(f"Vector search failed: {e}")
