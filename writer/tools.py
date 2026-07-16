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

from writer.config import (
    WRITER_CRITIC_TOOL_NAMES,
    WRITER_DIAGNOSIS_TOOL_NAMES,
    WRITER_WEB_SEARCH_ENABLED,
)
from writer.documents import get_document, list_documents, save_document, search_documents
from writer.runs import record_run_edit
from writer.prompts import scene_locator_entries
from writer.store import (
    append_manuscript,
    get_manuscript,
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
        "Read the saved manuscript by stable chapter anchors (preferred) or temporary character offsets. "
        "For durable checkpoints pass chapter plus its opening and closing sentence from the scene index; "
        "the server resolves their CURRENT offsets after earlier edits. Duplicate or missing anchors fail safely. "
        "With no arguments it returns the last 5000 characters. Max 20000 characters per call; long chapters "
        "return a next_start_anchor for the following page."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chapter": {"type": "string", "description": "Chapter heading or number, e.g. # 12 or 12."},
            "start_anchor": {"type": "string", "description": "Opening sentence copied from the scene index, or a continuation anchor."},
            "end_anchor": {"type": "string", "description": "Closing sentence copied from the scene index."},
            "start": {"type": "integer", "description": "Temporary start character offset (0-based)."},
            "end": {"type": "integer", "description": "Temporary end character offset (exclusive)."},
        },
    },
}

_READ_DOCUMENT_TOOL = {
    "name": "read_document",
    "description": (
        "Read one background document (worldbuilding, character sheets, outline, research notes) in full by its "
        "title. The available documents — project-specific and shared (common to all projects) — are listed in "
        "your context. Use these as the authoritative reference for setting, character, and plot facts. A project "
        "document shadows a shared one with the same title."
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
        "Substring search across this project's background documents AND the shared documents (titles and "
        "contents). Returns document titles with matching snippets. Use short distinctive queries; then "
        "read_document for the full text."
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
        "Create or fully overwrite a background document of THIS project by title (worldbuilding notes, character "
        "sheet, outline, timeline). This does NOT touch the manuscript, and it can never modify a shared document — "
        "reusing a shared document's title creates a project-local override of it. Use it when the writer asks you "
        "to record or update notes, or to keep an agreed story bible current after major developments. Overwrites "
        "the whole document — read it first if you are updating."
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


# CommuLingo people dictionary, read side only (the edit tool is not in the
# system.writer profile). The handler is project-independent and module-level,
# so its identity is stable for the dispatcher's signature cache. The spec
# reuses the canonical schema but swaps the description: the canonical one
# instructs the reader to follow up with commulingo_edit, which the writer
# does not have.
_COMMULINGO_PEOPLE_WRITER_TOOL: dict | None = None
_COMMULINGO_PEOPLE_HANDLER = None
try:
    from runtime_tools.commulingo_people import COMMULINGO_TOOL_HANDLERS, COMMULINGO_PEOPLE_TOOL

    _COMMULINGO_PEOPLE_WRITER_TOOL = {
        **COMMULINGO_PEOPLE_TOOL,
        "description": (
            "Read-only reference: the CommuLingo people dictionary "
            "(cyber-lenin.com/commulingo/people) — Soviet-history figures with bios, "
            "career timelines, and institution (office) leadership timelines, bilingual "
            "ko/en. Use it to get historical facts right (who held which post when, "
            "life dates, fates, name spellings ko/en/cyrillic) when the story touches "
            "real Soviet figures; background documents still rule for this story's own "
            "fictional canon. Actions: `search_people` (q matches id/name/cyrillic; "
            "optional group_id), `get_person` (full record), `get_sections` (a person's "
            "long-form detail sections), `list_groups` (era groups), `list_offices` / "
            "`get_office` (institution leadership timelines), `list_categories`, "
            "`list_events` / `get_event` (historical events and who was involved)."
        ),
    }
    _COMMULINGO_PEOPLE_HANDLER = COMMULINGO_TOOL_HANDLERS["commulingo_people"]
except Exception:
    logger.exception("writer: commulingo_people tool unavailable; continuing without it")


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

    async def _handle_read_manuscript(
        start: int | None = None,
        end: int | None = None,
        chapter: str | None = None,
        start_anchor: str | None = None,
        end_anchor: str | None = None,
    ) -> str:
        if not chapter:
            return await asyncio.to_thread(read_manuscript_slice, project_id, start, end)
        manuscript = await asyncio.to_thread(get_manuscript, project_id)
        body = str((manuscript or {}).get("body") or "")
        requested = str(chapter).strip().lstrip("#").strip().casefold()
        matches = [
            entry for entry in scene_locator_entries(body)
            if str(entry["heading"]).strip().lstrip("#").strip().casefold() == requested
        ]
        if len(matches) != 1:
            return f"Chapter locator must match exactly once; {chapter!r} matched {len(matches)} chapters."
        entry = matches[0]
        scope_start, scope_end = int(entry["start"]), int(entry["end"])
        scope = body[scope_start:scope_end]

        def resolve_anchor(anchor: str | None, default: int, *, after: bool = False) -> int | str:
            value = str(anchor or "").strip()
            if not value:
                return default
            count = scope.count(value)
            if count != 1:
                return f"Anchor must match exactly once inside {entry['heading']}; {value!r} matched {count} times."
            pos = scope.find(value)
            return scope_start + pos + (len(value) if after else 0)

        lo = resolve_anchor(start_anchor, scope_start)
        if isinstance(lo, str):
            return lo
        hi = resolve_anchor(end_anchor, scope_end, after=True)
        if isinstance(hi, str):
            return hi
        if hi < lo:
            return "The end anchor occurs before the start anchor inside the chapter."
        result = await asyncio.to_thread(read_manuscript_slice, project_id, lo, hi)
        if hi - lo > 20000:
            rest = body[lo + 20000:hi]
            next_anchor = next((line.strip() for line in rest.splitlines() if line.strip()), "")[:240]
            if next_anchor:
                result += (
                    f"\n[chapter continues; next call: read_manuscript(chapter={chapter!r}, "
                    f"start_anchor={next_anchor!r}, end_anchor={str(end_anchor or entry['end_anchor'])!r})]"
                )
        return result

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

    if _COMMULINGO_PEOPLE_WRITER_TOOL is not None and _COMMULINGO_PEOPLE_HANDLER is not None:
        tools.append(_COMMULINGO_PEOPLE_WRITER_TOOL)
        handlers["commulingo_people"] = _COMMULINGO_PEOPLE_HANDLER

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


def _tool_subset(project_id: int, names: frozenset[str]) -> tuple[list[dict], dict]:
    """Filter the memoized build_writer_tools output so handler identities are
    shared with the main pass (keeps the dispatcher's id-cache warm) and the
    subset stays inside the system.writer gateway profile."""
    tools, handlers = build_writer_tools(project_id)
    return (
        [t for t in tools if t["name"] in names],
        {name: fn for name, fn in handlers.items() if name in names},
    )


def build_critic_tools(project_id: int) -> tuple[list[dict], dict]:
    """Read/replace-only surface: the legacy line-edit pass and the author-
    revision stage of the diagnose→revise 퇴고 (refine, never append)."""
    return _tool_subset(project_id, WRITER_CRITIC_TOOL_NAMES)


def build_diagnosis_tools(project_id: int) -> tuple[list[dict], dict]:
    """Read-only surface for the diagnosis stage: it reports, it never edits.

    The author-revision stage of diagnose_revise has NO subset builder on
    purpose: it uses build_writer_tools verbatim so its request shares the
    exact (tools, system) prefix with the main pass and reads the prompt cache
    instead of re-writing it. Appending is forbidden by the revision request."""
    return _tool_subset(project_id, WRITER_DIAGNOSIS_TOOL_NAMES)
