"""Global runtime tool registry and execution handlers."""

import os
import sys
import json
import asyncio
import logging
import re

from runtime_tools.a2a import A2A_TOOL_HANDLERS, A2A_TOOLS
from runtime_tools.fetch import FETCH_TOOL_HANDLERS, FETCH_TOOLS
from runtime_tools.filesystem import FILESYSTEM_TOOL_HANDLERS, FILESYSTEM_TOOLS
from runtime_tools.media import MEDIA_TOOL_HANDLERS, MEDIA_TOOLS
from runtime_tools.social import SOCIAL_TOOL_HANDLERS, SOCIAL_TOOLS
from runtime_tools.web_search import execute_web_search
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)

# restart_service import-preflight targets. Dotted module names — the systemd
# units run dotted package entrypoints; stale flat module names make the
# preflight silently skip services via its isfile guard.
RESTART_PREFLIGHT_ENTRY_POINTS = {
    "telegram": "telegram.bot",
    "api": "services.api",
    "browser": "browser.worker",
}


def _looks_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7af]", text or ""))


def _looks_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or "")) and not _looks_korean(text)


from llm.json_utils import extract_json_object as _extract_json_object


async def _llm_translate_search_query(query: str, target_language: str, layer: str) -> str | None:
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


async def _search_corpus_multilingual(
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
        translated = await _llm_translate_search_query(query, "English", "core_theory")
        if translated:
            searches.append(("translated_en", translated, "core_theory"))
    elif _looks_english(query) and layer == "modern_analysis":
        translated = await _llm_translate_search_query(query, "Korean", "modern_analysis")
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

# ── Tool Definitions (Anthropic API format) ──────────────────────────
TOOLS = [
    {
        "name": "vector_search",
        "description": (
            "Search Marxist-Leninist document DB (pgvector). Returns excerpts with "
            "author/year/title. MATCH YOUR QUERY LANGUAGE TO THE LAYER: "
            "core_theory is English-language classics (Marx, Engels, Lenin, Mao, "
            "Trotsky translations) → query in English. modern_analysis is Korean "
            "analysis/commentary → query in Korean. self_produced_analysis is your "
            "own high-quality saved analysis → query in the language used when saved. "
            "Cross-language queries return near-empty results due to embedding-space separation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Use English for layer=core_theory, Korean for "
                        "layer=modern_analysis. For self_produced_analysis, use the "
                        "same language as the saved analysis. Mismatching language "
                        "to layer degrades recall sharply."
                    ),
                },
                "num_results": {"type": "integer", "description": "Results count (1-10).", "default": 5},
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis", "self_produced_analysis"],
                    "description": (
                        "core_theory: English-language Marxist-Leninist classics. "
                        "modern_analysis: Korean-language contemporary analysis/commentary. "
                        "self_produced_analysis: your own actively saved analytical outputs. "
                        "Omit to search all layers (not recommended — mixes languages)."
                    ),
                },
                "author": {
                    "type": "string",
                    "description": (
                        "Optional metadata filter. Use canonical author names such as "
                        "Stalin, Lenin, Mao, Rosa Luxemburg, Trotsky, Gramsci."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Optional title/source metadata substring filter, e.g. "
                        "'National Question' or 'Chinese Revolution'."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Optional metadata year filter, e.g. 1913.",
                },
                "keywords": {
                    "type": "string",
                    "description": (
                        "Optional exact keyword/phrase filter against chunk text or title. "
                        "Use this when vector similarity alone returns adjacent authors."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "knowledge_graph_search",
        "description": (
            "Search the knowledge graph (Neo4j): people, organizations, events, concepts, "
            "policies and documents across current affairs, the CommuLingo Soviet-history "
            "dictionary (people/terms/events, Korean canonical names) and published research/"
            "archival documents. Facts come back as 'Subject —Predicate→ Object: fact' with "
            "validity dates, trust tier and source. When the query consists of one unambiguous entity name the "
            "result is that entity's full neighbourhood (aliases, external ids, active and "
            "expired facts). Do not invent English names for Korean organizations/publications; "
            "prefer canonical names already used in KG, e.g. '디아마트 (DiaMat)' and "
            "'웹진 반란(Uprising)'. Preserve Korean person names such as '신현준' or '니키타 흐루쇼프' "
            "instead of romanizing them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What entities/relations to find. Preserve proper nouns in "
                        "their known canonical language/name; do not translate or "
                        "romanize Korean organization names unless that spelling is "
                        "part of the canonical name. For Korean people, use the "
                        "Korean name if known."
                    ),
                },
                "num_results": {"type": "integer", "description": "Maximum returned entity/fact items (1-20).", "default": 10},
                "entity": {
                    "type": "string",
                    "description": (
                        "Optional exact entity name or alias (e.g. '니키타 흐루쇼프', 'Nikita Khrushchev'). "
                        "Returns that entity's neighbourhood instead of a semantic search."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "entity", "semantic"],
                    "description": (
                        "auto (default): entity view for an entity name alone, "
                        "else semantic search. entity: force the entity view. semantic: force hybrid search."
                    ),
                    "default": "auto",
                },
            },
            "anyOf": [{"required": ["query"], "properties": {"query": {"pattern": "\\S"}}},
                      {"required": ["entity"], "properties": {"entity": {"pattern": "\\S"}}}],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search missing facts; stop when claims have adequate evidence. Reuse sources for wording/save repairs. "
            "New queries only for gaps, contradictions or plausible changes; no redundant paraphrases. "
            "Known URL: fetch_url. Snippets are leads; fetch before citing figures/quotes. "
            "Use domain parameters, not site operators; queries under 400 chars."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": 1500,
                    "description": (
                        "One missing fact; brief keywords/question with exact names, year/version/location. "
                        "Use source language/original names. No task prompts or answer-format demands."
                    ),
                },
                "max_results": {"type": "integer", "description": "Number of results (1-10).", "default": 5},
                "search_depth": {
                    "type": "string",
                    "enum": ["ultra-fast", "fast", "basic", "advanced"],
                    "description": (
                        "Default basic: 1 Tavily credit. advanced: 2 credits, focused extra context; "
                        "only for difficult unresolved facts. fast/ultra-fast prioritize latency."
                    ),
                    "default": "basic",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "news/finance rank recent coverage higher and return publish dates. Use for current events and market data.",
                    "default": "general",
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Filter by page publication/update recency. Use for current coverage; omit for historical research (put the historical year in query instead).",
                },
                "include_domains": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 10,
                    "description": "Strict domain/subdomain filter; may yield nothing. Prefer known primary sources; omit if unknown. Bare hosts only (docs.python.org), no URLs/paths/wildcards.",
                },
                "exclude_domains": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 10,
                    "description": "Exclude these domains and their subdomains when known to be irrelevant. Bare hostnames only, maximum 10; omit by default.",
                },
                "use_cache": {
                    "type": "boolean",
                    "description": "Reuse identical searches for up to 5 minutes (news/finance/day: 60 seconds). Set false only when a fresh lookup is required; it makes another paid request.",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
    *FILESYSTEM_TOOLS,
    *FETCH_TOOLS,
]


# ── Tool Execution Functions ─────────────────────────────────────────

async def _exec_vector_search(
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
        docs = await _search_corpus_multilingual(
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


async def _exec_kg_search(query: str = "", num_results: int = 10, entity: str | None = None,
                          mode: str = "auto") -> str:
    """Execute knowledge graph search (entity view or semantic) off the event loop."""
    try:
        from kg_runtime.search import search_knowledge_graph
        try:
            num_results = max(1, min(int(num_results), 20))
        except (TypeError, ValueError):
            num_results = 10
        result = await asyncio.to_thread(
            search_knowledge_graph, query, num_results, None,
            entity=(entity or "").strip() or None, mode=mode or "auto",
        )
        if not result:
            from tool_gateway.results import ToolResult
            return ToolResult("No knowledge graph results found.", getattr(result, "result_metadata", None))
        return result
    except Exception as e:
        logger.error("kg_search error: %s", e)
        return ToolFailure(f"Knowledge graph search failed; do not treat this as no KG data: {e}")


# ── Research publish/edit/unpublish tools live in runtime_tools.research ──
# They are registered into TOOLS / TOOL_HANDLERS at the bottom of this file.



# ── Mission Tool ──────────────────────────────────────────────────────

MISSION_TOOL = {
    "name": "mission",
    "description": "Manage the active mission (shared context between chat and tasks). Use 'status' to check current mission, 'close' to end a completed mission.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "close"],
                "description": "status: view active mission + recent events. close: end the mission (use when the goal is fully achieved).",
            },
        },
        "required": ["action"],
    },
}


def build_mission_handler(user_id: int):
    """Create a mission tool handler bound to a specific user_id."""
    async def _handle(action: str, **_kwargs) -> str:
        try:
            from telegram.mission import get_active_mission, get_mission_events, close_mission
            if action == "status":
                mission = get_active_mission(user_id)
                if not mission:
                    return "No active mission."
                events = get_mission_events(mission["id"], limit=10)
                lines = [f"Mission #{mission['id']}: {mission['title']} [{mission['status']}]"]
                lines.append(f"Created: {mission['created_at']}")
                if events:
                    lines.append(f"\nTimeline ({len(events)} events):")
                    for e in events:
                        lines.append(f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {str(e['content'] or '')[:200]}")
                return "\n".join(lines)
            elif action == "close":
                mission = get_active_mission(user_id)
                if not mission:
                    return "No active mission to close."
                return close_mission(mission["id"])
            return f"Unknown mission action: {action}"
        except Exception as e:
            return ToolFailure(f"Mission error: {e}")
    return _handle


# ── Web Search (Tavily / Brave provider chain) ───────────────────────

async def _exec_web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    time_range: str | None = None,
    use_cache: bool = True,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    return await execute_web_search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        time_range=time_range,
        use_cache=use_cache,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )


# ── Restart Service Tool ─────────────────────────────────────────────

RESTART_SERVICE_TOOL = {
    "name": "restart_service",
    "description": (
        "Restart a leninbot service with pre-flight syntax + import checks. "
        "Use instead of execute_python+subprocess. "
        "File→service mapping (and detailed procedure) lives in the programmer agent prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": ["telegram", "api", "browser", "all"],
                "description": "telegram=bot+agents, api=web+a2a, browser=browser worker, all=multi-service code. Default: telegram.",
            },
        },
        "required": [],
    },
}


async def _exec_restart_service(service: str = "telegram") -> str:
    """Safely restart service with pre-flight validation."""
    import ast
    import subprocess

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        from llm.runtime_context import current_task_ctx
        from telegram.tasks import persist_task_restart_state
        ctx = current_task_ctx.get()
        current_task_id = ctx["task_id"] if ctx else None
    except Exception:
        current_task_id = None
        persist_task_restart_state = None

    if service not in ("telegram", "api", "browser", "all"):
        return f"❌ Unknown service: {service}. Use: telegram, api, browser, all"

    # 1. Find .py files with uncommitted changes (staged + unstaged)
    try:
        diff_result = await asyncio.to_thread(
            subprocess.run,
            ["git", "diff", "--name-only", "HEAD", "--diff-filter=ACMR"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        # Also include untracked .py files that might be new
        untracked = await asyncio.to_thread(
            subprocess.run,
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        changed_files = set()
        for line in (diff_result.stdout + "\n" + untracked.stdout).strip().split("\n"):
            line = line.strip()
            if line.endswith(".py"):
                changed_files.add(line)
    except Exception as e:
        return ToolFailure(f"❌ Failed to detect changed files: {e}")

    errors = []

    # 2. Syntax check all changed .py files
    for rel_path in sorted(changed_files):
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=rel_path)
        except SyntaxError as e:
            errors.append(f"SyntaxError in {rel_path}:{e.lineno} — {e.msg}")

    if errors:
        return "❌ Restart blocked — syntax errors found:\n" + "\n".join(errors)

    # 3. Import-level validation: try importing the entry points in a subprocess
    targets = ["telegram", "api", "browser"] if service == "all" else [service]

    for target in targets:
        module = RESTART_PREFLIGHT_ENTRY_POINTS[target]
        module_path = os.path.join(project_root, module.replace(".", os.sep) + ".py")
        if not os.path.isfile(module_path):
            # A missing entry file means the map rotted (this exact guard
            # silently skipped telegram/browser for months when the flat
            # telegram_bot.py/browser_worker.py modules became packages).
            logger.warning(
                "restart_service preflight: entry module %s not found at %s — import check skipped",
                module, module_path,
            )
            continue
        try:
            # Run a quick import check in isolated subprocess
            check_code = (
                f"import sys; sys.path.insert(0, {project_root!r}); "
                f"import importlib; importlib.import_module({module!r})"
            )
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-c", check_code],
                capture_output=True, text=True, timeout=30,
                cwd=project_root,
                env={**os.environ, "PREFLIGHT_CHECK": "1"},
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Extract the last meaningful error line
                err_lines = [l for l in stderr.split("\n") if l.strip()]
                last_err = err_lines[-1] if err_lines else "unknown error"
                errors.append(f"Import check failed for {module}.py: {last_err}")
        except subprocess.TimeoutExpired:
            errors.append(f"Import check timed out for {module}.py (>30s)")
        except Exception as e:
            errors.append(f"Import check error for {module}.py: {e}")

    if errors:
        return "❌ Restart blocked — import errors found:\n" + "\n".join(errors)

    if current_task_id and persist_task_restart_state:
        try:
            persist_task_restart_state(
                current_task_id,
                service=service,
                phase="requested",
                mark_completed=False,
            )
        except Exception as e:
            return ToolFailure(f"❌ Restart blocked — failed to persist durable restart state: {e}")

    # 4. All checks passed — daemon-reload (picks up any unit file changes), then restart
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["sudo", "-n", "systemctl", "daemon-reload"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass  # non-fatal: restart will still use previous unit config

    svc_map = {
        "telegram": ["leninbot-telegram"],
        "api": ["leninbot-api"],
        "browser": ["leninbot-browser"],
        "all": ["leninbot-api", "leninbot-browser", "leninbot-telegram"],  # API first, browser second, telegram last
    }
    results = []
    restart_failed = False
    for svc in svc_map[service]:
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["sudo", "-n", "systemctl", "restart", svc],
                capture_output=True, text=True, timeout=15,
                start_new_session=True,
            )
            if proc.returncode == 0:
                results.append(f"✅ {svc}: restarted")
            else:
                restart_failed = True
                results.append(f"❌ {svc}: {proc.stderr.strip()}")
        except subprocess.TimeoutExpired:
            restart_failed = True
            results.append(f"⏱ {svc}: timeout")
        except Exception as e:
            restart_failed = True
            results.append(f"❌ {svc}: {e}")

    if current_task_id and persist_task_restart_state:
        try:
            persist_task_restart_state(
                current_task_id,
                service=service,
                phase="verification" if not restart_failed else "requested",
                mark_completed=not restart_failed,
                resumed_after_restart=not restart_failed,
                reentry_reason=(
                    "restart completed; next step is post-restart verification"
                    if not restart_failed
                    else "restart command failed; restart branch may retry after fix"
                ),
            )
        except Exception as e:
            results.append(f"⚠️ durable restart completion state update failed: {e}")

    checked_files = ", ".join(sorted(changed_files)[:10]) if changed_files else "(none)"
    return (
        f"Pre-flight checks passed (syntax + import OK, changed: {checked_files})\n"
        + "\n".join(results)
    )


# ── Handler Registry ─────────────────────────────────────────────────

def dedupe_tool_registry(tools: list[dict]) -> list[dict]:
    """Deduplicate tool registry entries by name while preserving first occurrence.

    Root cause for browser task #330: source code had been patched, but a worker can
    still start or keep running with an inconsistent import/lifecycle state. Keeping
    the registry itself unique makes every downstream caller safer, regardless of
    whether agent-level dedupe runs.
    """
    deduped: list[dict] = []
    seen_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            deduped.append(tool)
            continue
        name = str(tool.get("name", "") or "").strip()
        if name and name in seen_names:
            logger.warning("Dropping duplicate tool from base registry: %s", name)
            continue
        if name:
            seen_names.add(name)
        deduped.append(tool)
    return deduped


TOOL_HANDLERS = {
    "vector_search": _exec_vector_search,
    "knowledge_graph_search": _exec_kg_search,
    "web_search": _exec_web_search,
    **FETCH_TOOL_HANDLERS,
    **FILESYSTEM_TOOL_HANDLERS,
    "restart_service": _exec_restart_service,
}

# ── Restart service tool ──────────────────────────────────────────────
TOOLS.append(RESTART_SERVICE_TOOL)

# ── R2 Upload + File Registry ────────────────────────────────────────
UPLOAD_TO_R2_TOOL = {
    "name": "upload_to_r2",
    "description": (
        "Upload a local file to Cloudflare R2 and get a public URL. "
        "Automatically registers the file in the file_registry DB table so other agents can find it. "
        "Use for images, documents, or any file that needs a public URL (e.g. email attachments, web assets)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "local_path": {"type": "string", "description": "Absolute path to the local file."},
            "key": {"type": "string", "description": "Object key/path in R2 bucket (e.g. 'email-assets/logo.png'). Defaults to filename."},
            "description": {"type": "string", "description": "What this file is / what it's for."},
            "category": {
                "type": "string",
                "enum": ["email-asset", "image", "document", "research", "general"],
                "description": "File category for search. Default: general.",
            },
        },
        "required": ["local_path"],
    },
}


async def _exec_upload_to_r2(
    local_path: str, key: str | None = None, description: str = "", category: str = "general",
) -> str:
    from shared import upload_to_r2
    from db import execute as db_execute, query as db_query
    import mimetypes

    path = os.path.abspath(local_path)
    if not os.path.isfile(path):
        return f"File not found: {local_path}"

    filename = os.path.basename(path)
    file_size = os.path.getsize(path)
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    if key is None:
        key = f"{category}/{filename}" if category != "general" else filename

    # Check if already registered by local_path or R2 key
    existing = await asyncio.to_thread(
        db_query,
        "SELECT id, public_url FROM file_registry WHERE local_path = %s OR public_url LIKE %s LIMIT 1",
        (path, f"%/{key}"),
    )
    if existing:
        return f"Already registered: {existing[0]['public_url']}\n(file_registry id: {existing[0]['id']})"

    url = await asyncio.to_thread(upload_to_r2, path, key, content_type)
    if not url:
        return "R2 upload failed. Check R2 env config."

    # Get current task context for tracking
    task_id = None
    agent_type = None
    try:
        from llm.runtime_context import current_task_ctx
        ctx = current_task_ctx.get()
        task_id = ctx["task_id"] if ctx else None
    except Exception:
        pass

    # Register in file_registry
    registry_id = None
    try:
        reg_rows = await asyncio.to_thread(
            db_query,
            "INSERT INTO file_registry (local_path, public_url, filename, content_type, description, category, file_size, created_by_task_id, created_by_agent) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (path, url, filename, content_type, description or filename, category, file_size, task_id, agent_type),
        )
        registry_id = reg_rows[0]["id"] if reg_rows else None
    except Exception as e:
        logger.warning("file_registry insert failed: %s", e)

    reg_line = f"\nfile_registry id: {registry_id}" if registry_id else "\n(file_registry registration failed)"
    return f"Uploaded: {url}\nLocal: {path}\nSize: {file_size} bytes\nCategory: {category}{reg_line}"


TOOLS.append(UPLOAD_TO_R2_TOOL)
TOOL_HANDLERS["upload_to_r2"] = _exec_upload_to_r2

# ── Send Email Tool (implementation: mail_runtime/tools.py) ─────────
from mail_runtime.tools import SEND_EMAIL_TOOL, exec_send_email

TOOLS.append(SEND_EMAIL_TOOL)
TOOL_HANDLERS["send_email"] = exec_send_email

# ── Self-awareness tools (shared memory access) ─────────────────────
from self_runtime.tools import SELF_TOOLS, SELF_TOOL_HANDLERS

TOOLS.extend(SELF_TOOLS)
TOOLS.append(MISSION_TOOL)
TOOL_HANDLERS.update(SELF_TOOL_HANDLERS)

# ── Finance data tool (real-time market prices) ──────────────────────
from runtime_tools.finance import FINANCE_TOOL, FINANCE_TOOL_HANDLER

TOOLS.append(FINANCE_TOOL)
TOOL_HANDLERS["get_finance_data"] = FINANCE_TOOL_HANDLER

# ── X/Twitter post lookup ────────────────────────────────────────────
from runtime_tools.x import X_TOOLS, X_TOOL_HANDLERS

TOOLS.extend(X_TOOLS)
TOOL_HANDLERS.update(X_TOOL_HANDLERS)

# ── Site publishing tools (hub curations + static pages for cyber-lenin.com) ──
from runtime_tools.site_publishing import SITE_PUBLISHING_TOOLS, SITE_PUBLISHING_TOOL_HANDLERS

TOOLS.extend(SITE_PUBLISHING_TOOLS)
TOOL_HANDLERS.update(SITE_PUBLISHING_TOOL_HANDLERS)

# ── Direct SQL tool (programmer only; analyst etc. keep read_self/kg_search) ──
from runtime_tools.db import DB_TOOLS, DB_TOOL_HANDLERS

TOOLS.extend(DB_TOOLS)
TOOL_HANDLERS.update(DB_TOOL_HANDLERS)

# ── Public-post editor (UPDATE + Redis cache purge in one step) ──
from runtime_tools.post_edit import POST_EDIT_TOOLS, POST_EDIT_TOOL_HANDLERS

TOOLS.extend(POST_EDIT_TOOLS)
TOOL_HANDLERS.update(POST_EDIT_TOOL_HANDLERS)

# ── Research publish/edit/unpublish (atomic write + cache purge) ──
from runtime_tools.research import RESEARCH_TOOLS, RESEARCH_TOOL_HANDLERS

TOOLS.extend(RESEARCH_TOOLS)
TOOL_HANDLERS.update(RESEARCH_TOOL_HANDLERS)

# ── Wikipedia (free MediaWiki API — no Tavily credits) ──
from runtime_tools.wiki import WIKI_TOOLS, WIKI_TOOL_HANDLERS

TOOLS.extend(WIKI_TOOLS)
TOOL_HANDLERS.update(WIKI_TOOL_HANDLERS)

# ── CommuLingo people dictionary (read + staged edit suggestions) ──
from commulingo.people import COMMULINGO_TOOLS, COMMULINGO_TOOL_HANDLERS

TOOLS.extend(COMMULINGO_TOOLS)
TOOL_HANDLERS.update(COMMULINGO_TOOL_HANDLERS)

# ── Admin-only private research documents (not exposed to public web chat) ──
from runtime_tools.private_reports import PRIVATE_REPORT_TOOLS, PRIVATE_REPORT_TOOL_HANDLERS

TOOLS.extend(PRIVATE_REPORT_TOOLS)
TOOL_HANDLERS.update(PRIVATE_REPORT_TOOL_HANDLERS)

# ── Crypto wallet tools (address + balance + swap + transfer + x402 pay) ───
from crypto_wallet import (
    WALLET_TOOL, WALLET_TOOL_HANDLER,
    SWAP_TOOL, SWAP_TOOL_HANDLER,
    TRANSFER_TOOL, TRANSFER_TOOL_HANDLER,
    PAY_AND_FETCH_TOOL, PAY_AND_FETCH_TOOL_HANDLER,
)

TOOLS.append(WALLET_TOOL)
TOOL_HANDLERS["check_wallet"] = WALLET_TOOL_HANDLER
TOOLS.append(SWAP_TOOL)
TOOL_HANDLERS["swap_eth_to_usdc"] = SWAP_TOOL_HANDLER
TOOLS.append(TRANSFER_TOOL)
TOOL_HANDLERS["transfer_usdc"] = TRANSFER_TOOL_HANDLER
TOOLS.append(PAY_AND_FETCH_TOOL)
TOOL_HANDLERS["pay_and_fetch"] = PAY_AND_FETCH_TOOL_HANDLER

# ── Telegram channel broadcast tool ─────────────────────────────────
from runtime_tools.broadcast import BROADCAST_TO_CHANNEL_TOOL, broadcast_to_channel

TOOLS.append(BROADCAST_TO_CHANNEL_TOOL)
TOOL_HANDLERS["broadcast_to_channel"] = broadcast_to_channel

TOOLS.extend(MEDIA_TOOLS)
TOOL_HANDLERS.update(MEDIA_TOOL_HANDLERS)


# ── Mailbox tools (implementation: mail_runtime/tools.py) ──────────
from mail_runtime.tools import (
    ALLOWLIST_SENDER_TOOL, CHECK_INBOX_TOOL, exec_allowlist_sender, exec_check_inbox,
)

TOOLS.append(CHECK_INBOX_TOOL)
TOOL_HANDLERS["check_inbox"] = exec_check_inbox
TOOLS.append(ALLOWLIST_SENDER_TOOL)
TOOL_HANDLERS["allowlist_sender"] = exec_allowlist_sender

# ── Diary Writer Tool ─────────────────────────────────────────────────
SAVE_DIARY_TOOL = {
    "name": "save_diary",
    "description": "Save a diary entry to the ai_diary table. Used by the diary agent to persist generated diary entries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "One-line title/summary of the diary entry (Korean)."},
            "content": {"type": "string", "description": "Full diary body text (Korean, 2+ paragraphs)."},
        },
        "required": ["title", "content"],
    },
}


def _check_diary_publication_risks(title: str, content: str) -> list[str]:
    """Return advisory risk reasons for diary content.

    This intentionally checks only secret-like technical material. Editorial,
    political, reputational, or current-usefulness judgment belongs to the LLM
    review path, not substring or regex policy.
    """
    text = f"{title or ''}\n{content or ''}"
    reasons: list[str] = []

    secret_patterns = [
        (r"sk-[A-Za-z0-9_-]{20,}", "possible API key"),
        (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
        (r"\b(seed phrase|mnemonic|private key|api key|access token|refresh token)\b", "secret-bearing phrase"),
        (r"\b[A-Za-z0-9+/]{40,}={0,2}\b", "long token-like string"),
    ]
    for pattern, label in secret_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reasons.append(label)

    return reasons


async def _exec_save_diary(title: str, content: str) -> str:
    from db import query_one as db_query_one
    try:
        risk_reasons = _check_diary_publication_risks(title, content)
        if risk_reasons:
            logger.warning(
                "save_diary publication risk advisory: %s",
                "; ".join(dict.fromkeys(risk_reasons)),
            )
        row = await asyncio.to_thread(
            db_query_one,
            "INSERT INTO ai_diary (title, content) VALUES (%s, %s) RETURNING id",
            (title, content),
        )
        diary_id = row.get("id") if row else None
        broadcast_note = ""
        try:
            from telegram.channel_broadcast import should_broadcast_diary, send_broadcast
            if should_broadcast_diary():
                preview = re.sub(r"\s+", " ", (content or "").strip())
                if len(preview) > 500:
                    cut = preview[:501]
                    split_at = max(cut.rfind(" "), cut.rfind("."), cut.rfind("。"), cut.rfind("!"), cut.rfind("?"))
                    if split_at < 250:
                        split_at = 500
                    preview = cut[:split_at].rstrip(" ,;:") + "..."
                public_url = f"https://cyber-lenin.com/ai-diary/{diary_id}" if diary_id else "https://cyber-lenin.com/ai-diary"
                result = await send_broadcast(
                    title=f"사이버-레닌 일기: {title}",
                    summary=preview,
                    url=public_url,
                )
                broadcast_note = f" / Telegram channel: {'sent' if result.ok else result.message}"
        except Exception as e:
            broadcast_note = f" / Telegram channel failed: {e}"
        risk_note = ""
        if risk_reasons:
            risk_note = " / publication guard: advisory warning logged"
        return f"Diary saved: {title}{broadcast_note}{risk_note}"
    except Exception as e:
        return ToolFailure(f"Failed to save diary: {e}")


TOOLS.append(SAVE_DIARY_TOOL)
TOOL_HANDLERS["save_diary"] = _exec_save_diary

TOOLS.extend(SOCIAL_TOOLS)
TOOL_HANDLERS.update(SOCIAL_TOOL_HANDLERS)

TOOLS.extend(A2A_TOOLS)
TOOL_HANDLERS.update(A2A_TOOL_HANDLERS)




from mail_runtime.inbox import PREPARE_MAIL_BRIEFING_TOOL, prepare_mail_briefing
TOOLS.append(PREPARE_MAIL_BRIEFING_TOOL)
TOOL_HANDLERS["prepare_mail_briefing"] = prepare_mail_briefing

# Private standalone roleplay notes; handler also enforces caller isolation.
from roleplay.memory import ROLEPLAY_MEMORY_TOOL, roleplay_memory
TOOLS.append(ROLEPLAY_MEMORY_TOOL)
TOOL_HANDLERS["roleplay_memory"] = roleplay_memory

from roleplay.memory import ROLEPLAY_STATE_TOOL, roleplay_state
TOOLS.append(ROLEPLAY_STATE_TOOL)
TOOL_HANDLERS["roleplay_state"] = roleplay_state

from roleplay.memory import ROLEPLAY_PERSON_TOOL, roleplay_person
TOOLS.append(ROLEPLAY_PERSON_TOOL)
TOOL_HANDLERS["roleplay_person"] = roleplay_person


# ── Schema normalization ─────────────────────────────────────────────
#
# Runs last — after every module-level TOOLS.append/extend above and after
# the name dedupe — so every registered tool acquires
# ``additionalProperties: false`` unless it deliberately opts out. Effects per provider:
#   * llama-server: constrains grammar-based tool-call decoding so Qwen
#     can't emit parameter names outside the declared schema.
#   * Anthropic: treats it as advisory (no behavioral change).
#   * OpenAI: strict mode is enabled only when the schema is also
#     "strict-safe" (see openai_tool_loop._convert_tool_anthropic_to_openai).

def _normalize_tool_schemas_inplace(tools: list[dict]) -> None:
    for t in tools:
        schema = t.get("input_schema")
        if not isinstance(schema, dict):
            continue
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False


TOOLS = dedupe_tool_registry(TOOLS)
_normalize_tool_schemas_inplace(TOOLS)
