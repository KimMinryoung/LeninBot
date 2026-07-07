"""Writer tool surface: schemas and per-project handlers.

The allowed surface is declared centrally as the ``system.writer`` tool
profile in tool_gateway.profiles (the same registry every other agent surface
uses); build_writer_tools enforces it, and security_gateway.policy assigns the
risk classes. Handlers are closures over project_id so the model never
supplies it."""

from __future__ import annotations

import asyncio
import logging

from tool_gateway.profiles import WRITER_PROFILE, profile_tool_names

from writer.config import WRITER_CRITIC_TOOL_NAMES, WRITER_WEB_SEARCH_ENABLED
from writer.documents import get_document, list_documents, save_document, search_documents
from writer.runs import record_run_edit
from writer.store import (
    append_manuscript,
    read_manuscript_slice,
    replace_manuscript_text,
    search_manuscript,
)

logger = logging.getLogger(__name__)

_SEARCH_MANUSCRIPT_TOOL = {
    "name": "search_manuscript",
    "description": (
        "Exact-substring search over the SAVED manuscript (whitespace/quote-insensitive fallback for multi-word "
        "queries). Returns matching passages with character offsets. Use SHORT distinctive queries — a name, or a "
        "2-6 word phrase; long quoted paragraphs usually fail. For continuity checks on earlier scenes beyond the "
        "context tail. Never search for prose you wrote this same turn: the tail in context and tool confirmations "
        "already reflect the saved draft. To pull broader context around a hit, follow up with read_manuscript."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short substring or phrase to find (a name, 2-6 words)."},
            "limit": {"type": "integer", "description": "Max passages to return (1-20).", "default": 8},
        },
        "required": ["query"],
    },
}

_READ_MANUSCRIPT_TOOL = {
    "name": "read_manuscript",
    "description": (
        "Read an exact slice of the saved manuscript by character offsets (offsets appear in search_manuscript "
        "results and the manuscript context header). Use it to pull full surrounding context before revising a "
        "passage, or to re-read an earlier scene. With no arguments it returns the last 5000 characters. "
        "Max 20000 characters per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start": {"type": "integer", "description": "Start character offset (0-based). Omit to read the tail."},
            "end": {"type": "integer", "description": "End character offset (exclusive). Defaults to start + 5000."},
        },
    },
}

_READ_DOCUMENT_TOOL = {
    "name": "read_document",
    "description": (
        "Read one background document (worldbuilding, character sheets, outline, research notes) in full by its "
        "title. The available documents are listed in your context. Use these as the authoritative reference for "
        "setting, character, and plot facts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Exact document title (case-insensitive)."},
        },
        "required": ["title"],
    },
}

_SEARCH_DOCUMENTS_TOOL = {
    "name": "search_documents",
    "description": (
        "Substring search across all background documents of this project (titles and contents). Returns document "
        "titles with matching snippets. Use short distinctive queries; then read_document for the full text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short substring to find (a name, a term)."},
            "limit": {"type": "integer", "description": "Max documents to return (1-20).", "default": 8},
        },
        "required": ["query"],
    },
}

_SAVE_DOCUMENT_TOOL = {
    "name": "save_document",
    "description": (
        "Create or fully overwrite a background document by title (worldbuilding notes, character sheet, outline, "
        "timeline). This does NOT touch the manuscript. Use it when the writer asks you to record or update notes, "
        "or to keep an agreed story bible current after major developments. Overwrites the whole document — read it "
        "first if you are updating."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Document title. Reusing an existing title overwrites that document."},
            "content": {"type": "string", "description": "Full document content (replaces any previous content)."},
            "kind": {"type": "string", "description": "Category label, e.g. character/setting/outline/research/note.", "default": "note"},
        },
        "required": ["title", "content"],
    },
}

_APPEND_MANUSCRIPT_TOOL = {
    "name": "append_to_manuscript",
    "description": (
        "Append new prose to the END of the manuscript — use this to continue the story. "
        "The text is added after the current ending; do not repeat existing text. "
        "This edits the saved manuscript directly."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Manuscript-ready prose to append."},
        },
        "required": ["text"],
    },
}

_REPLACE_MANUSCRIPT_TOOL = {
    "name": "replace_in_manuscript",
    "description": (
        "Revise a specific part of the manuscript: find an existing passage and replace it with new prose. "
        "This edits the saved manuscript directly. Copy 'find' from the saved manuscript (whitespace and quote-style "
        "differences are tolerated, wording is not) and make it unique — if you don't have the real text at hand, "
        "read_manuscript or search_manuscript gives it to you. Fails safely if 'find' is missing or ambiguous; "
        "then include more surrounding text and retry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "find": {"type": "string", "description": "The exact existing passage to replace (verbatim, unique in the manuscript)."},
            "replacement": {"type": "string", "description": "The new prose to put in its place."},
        },
        "required": ["find", "replacement"],
    },
}


_writer_tools_cache: dict[int, tuple[list[dict], dict]] = {}


def invalidate_project_tools(project_id: int) -> None:
    _writer_tools_cache.pop(project_id, None)


def build_writer_tools(project_id: int) -> tuple[list[dict], dict]:
    """Build the (tools, handlers) pair for a writer turn, bound to one project.

    - Handlers are closures over project_id so the model never supplies it.
    - Memoized per project: the tool dispatcher caches handler signatures by
      object identity, and per-request closures whose ids get recycled poisoned
      that cache (observed as spurious 'unexpected keyword argument' tool
      failures). Keeping one long-lived handler set per project avoids id reuse.
    - The final surface is filtered against the system.writer gateway profile,
      the single declaration of what the writer may call.
    """
    cached = _writer_tools_cache.get(project_id)
    if cached is not None:
        return cached

    async def _handle_search_manuscript(query: str, limit: int = 8) -> str:
        try:
            n = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            n = 8
        rows = await asyncio.to_thread(search_manuscript, project_id, query, n)
        if not rows:
            return (
                f"No manuscript matches for: {query}\n"
                "Search is exact-substring (with a whitespace/quote-insensitive fallback). "
                "Try a SHORTER distinctive phrase (a name, 2-4 words), or read_manuscript to read a region directly. "
                "Do not search for text you wrote this turn — it is already saved as confirmed by the tool results."
            )
        blocks = []
        for r in rows:
            heading = str(r.get("heading") or "").strip()
            label = f"[{heading}] " if heading else ""
            snippet = str(r.get("snippet") or "").strip()
            blocks.append(
                f"{label}chars {r.get('start_offset')}–{r.get('end_offset')} "
                f"(match {r.get('match_start')}:{r.get('match_end')}):\n…{snippet}…"
            )
        return "\n\n".join(blocks)

    async def _handle_read_manuscript(start: int | None = None, end: int | None = None) -> str:
        return await asyncio.to_thread(read_manuscript_slice, project_id, start, end)

    async def _handle_read_document(title: str) -> str:
        doc = await asyncio.to_thread(get_document, project_id, None, title)
        if not doc:
            titles = [str(d.get("title")) for d in await asyncio.to_thread(list_documents, project_id)]
            listing = "; ".join(titles) if titles else "(no documents exist yet)"
            return f"No document titled {title!r}. Available documents: {listing}"
        content = str(doc.get("content") or "")
        suffix = ""
        if len(content) > 30000:
            content = content[:30000]
            suffix = "\n[truncated at 30000 chars]"
        return f"Document {doc.get('title')!r} (kind: {doc.get('kind')}, {doc.get('char_count')} chars):\n{content}{suffix}"

    async def _handle_search_documents(query: str, limit: int = 8) -> str:
        try:
            n = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            n = 8
        rows = await asyncio.to_thread(search_documents, project_id, query, n)
        if not rows:
            return f"No document matches for: {query}. Use short distinctive terms, or read_document by title."
        blocks = [
            f"{r.get('title')!r} (kind: {r.get('kind')}):\n…{str(r.get('snippet') or '').strip()}…"
            for r in rows
        ]
        return "\n\n".join(blocks)

    async def _handle_save_document(title: str, content: str, kind: str = "note") -> str:
        if not title or not title.strip():
            return "No title provided; nothing saved."
        doc = await asyncio.to_thread(save_document, project_id, title, content or "", kind or "note")
        if not doc:
            return "Save failed: project not found."
        return f"Saved document {doc.get('title')!r} (kind: {doc.get('kind')}, {doc.get('char_count')} chars)."

    async def _handle_append(text: str) -> str:
        if not text or not text.strip():
            return "No text provided; nothing appended."
        result = await asyncio.to_thread(append_manuscript, project_id, text, "tool append")
        if not result:
            return "Append failed: manuscript not found."
        record_run_edit(project_id, "append", result.get("edit_start"), result.get("edit_end"), 0)
        return f"Appended. Manuscript is now {result.get('char_count')} characters."

    async def _handle_replace(find: str, replacement: str) -> str:
        result = await asyncio.to_thread(replace_manuscript_text, project_id, find, replacement, "tool replace")
        if result.get("ok"):
            record_run_edit(project_id, "replace", result.get("edit_start"), result.get("edit_end"), result.get("delta"))
            return f"Replaced. Manuscript is now {result.get('char_count')} characters."
        return "Replace failed: " + result.get("message", "unknown error")

    tools: list[dict] = [
        _SEARCH_MANUSCRIPT_TOOL,
        _READ_MANUSCRIPT_TOOL,
        _APPEND_MANUSCRIPT_TOOL,
        _REPLACE_MANUSCRIPT_TOOL,
        _READ_DOCUMENT_TOOL,
        _SEARCH_DOCUMENTS_TOOL,
        _SAVE_DOCUMENT_TOOL,
    ]
    handlers: dict = {
        "search_manuscript": _handle_search_manuscript,
        "read_manuscript": _handle_read_manuscript,
        "append_to_manuscript": _handle_append,
        "replace_in_manuscript": _handle_replace,
        "read_document": _handle_read_document,
        "search_documents": _handle_search_documents,
        "save_document": _handle_save_document,
    }

    # Web research (see WRITER_WEB_SEARCH_ENABLED): the main model gets the
    # research_web delegation tool, not raw web_search — a light DeepSeek
    # sub-agent runs the searches and only its distilled brief enters
    # heavy-model context (writer.research).
    if WRITER_WEB_SEARCH_ENABLED:
        try:
            from writer.models import resolve_writer_model
            from writer.research import RESEARCH_TOOL_SPEC, build_research_handler

            tools.append(RESEARCH_TOOL_SPEC)
            handlers["research_web"] = build_research_handler(
                project_id, lambda: resolve_writer_model(None)
            )
        except Exception:
            logger.exception("writer: research_web tool unavailable; continuing with manuscript search only")

    # Enforce the gateway profile: the declared allow-list is authoritative.
    allowed = profile_tool_names(WRITER_PROFILE)
    dropped = [t["name"] for t in tools if t["name"] not in allowed]
    if dropped:
        logger.warning("writer tools outside %s profile dropped: %s", WRITER_PROFILE, dropped)
    tools = [t for t in tools if t["name"] in allowed]
    handlers = {name: fn for name, fn in handlers.items() if name in allowed}

    _writer_tools_cache[project_id] = (tools, handlers)
    return tools, handlers


def build_critic_tools(project_id: int) -> tuple[list[dict], dict]:
    """Read/replace-only tool surface for the optional line-edit pass.

    Filters the memoized build_writer_tools output so handler identities are
    shared with the main pass (keeps the dispatcher's id-cache warm) and the
    subset stays inside the system.writer gateway profile."""
    tools, handlers = build_writer_tools(project_id)
    return (
        [t for t in tools if t["name"] in WRITER_CRITIC_TOOL_NAMES],
        {name: fn for name, fn in handlers.items() if name in WRITER_CRITIC_TOOL_NAMES},
    )
