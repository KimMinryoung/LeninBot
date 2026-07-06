"""Writer prompt assembly: system prompt, manuscript context, message history,
and response parsing."""

from __future__ import annotations

import re

from writer.config import (
    CONTEXT_CHAR_BUDGET,
    MANUSCRIPT_SELECTION_LIMIT,
    MANUSCRIPT_TAIL_CHARS,
    MAX_CONTEXT_MESSAGES,
    MIN_CONTEXT_MESSAGES,
    PINNED_DOC_CHAR_LIMIT,
    PINNED_DOC_KIND,
    PINNED_DOC_MAX_COUNT,
    WRITER_CACHE_CONTROL_1H,
    WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
    WRITER_WEB_SEARCH_ENABLED,
)
from writer.documents import get_document, list_documents
from writer.store import get_manuscript, recent_messages


def _tools_prompt_section() -> str:
    """The '# Tools' prompt block, gated to the tools that are actually wired
    (see writer.tools.build_writer_tools). Advertising a tool the model can't
    call would just make it hallucinate calls."""
    lines = [
        "# Tools — you edit the manuscript yourself\n",
        "You act on the manuscript directly through tools; the writer never copies text by hand. "
        "Use tools silently and on your own initiative, then give a short commentary.\n",
        "- search_manuscript(query): exact-substring search of the FULL saved manuscript. Use SHORT distinctive "
        "queries (a name, a 2-6 word phrase) to check continuity or locate an earlier scene. Long quoted "
        "paragraphs rarely match.\n",
        "- read_manuscript(start, end): read any region of the saved manuscript by character offsets "
        "(no arguments = the last 5000 chars). Prefer this over repeated searching when you need context.\n",
        "- append_to_manuscript(text): add prose to the END of the manuscript — for continuing the story.\n",
        "- replace_in_manuscript(find, replacement): revise a specific part — 'find' must match saved text "
        "(whitespace/quote drift tolerated) and be unique. Take 'find' from read_manuscript or search_manuscript "
        "results, or from the context tail.\n",
        "- read_document(title) / search_documents(query): consult the project's background documents "
        "(character sheets, worldbuilding, outline, research). They are the authoritative reference for "
        "setting and plot facts — check them before inventing or contradicting a detail.\n",
        "- save_document(title, content, kind): create or fully overwrite a background document. Use when the "
        "writer asks you to record notes, or to keep an agreed story bible current. Never store manuscript "
        "prose in documents.\n",
        "- Maintain a document titled 'Story so far' with kind 'pinned': a compact synopsis (under 4000 chars) "
        "of events, character states, open threads, and planted setups. Pinned documents are placed in your "
        "context automatically every turn, so keeping it current replaces expensive re-reading of the "
        "manuscript. Update it after any scene that changes the situation; keep it factual and dense.\n",
    ]
    if WRITER_WEB_SEARCH_ENABLED:
        lines.append(
            "- web_search: look up real-world facts (history, geography, technical or domain detail) when accuracy would ground the fiction. "
            "Do not over-research or let it flatten the prose into an encyclopedia; use it only to get concrete details right.\n"
        )
    lines.append(
        "Every manuscript change MUST go through append_to_manuscript or replace_in_manuscript — never paste manuscript "
        "prose into your text reply. Tool calls are invisible to the reader; never narrate that you searched or edited.\n"
        "Tool economy: the manuscript tail in your context IS the current saved draft — do not re-search or re-read text "
        "you can already see, and never search for prose you wrote earlier in this same turn (a successful tool result "
        "already confirms it was saved). One or two well-chosen lookups beat a chain of guesses.\n\n"
    )
    return "".join(lines)


def build_base_system_prompt(project: dict) -> str:
    premise = str(project.get("premise") or "").strip() or "(No premise recorded yet.)"
    style_notes = str(project.get("style_notes") or "").strip() or "(No style notes recorded yet.)"
    title = str(project.get("title") or "Untitled").strip()
    return (
        "You are an elite fiction-writing collaborator on one writer's personal novel. "
        "You are not a chatbot or an assistant persona — you are a craftsperson serving this manuscript, "
        "and prose quality is the only thing that matters.\n\n"
        f"Project title: {title}\n"
        f"Premise:\n{premise}\n\n"
        f"Style and continuity notes:\n{style_notes}\n\n"
        "# Craft standards\n"
        "- Write in the language, voice, point of view, and tense already established in the manuscript context. "
        "Match its prose rhythm and diction; never reset to a generic default voice.\n"
        "- Dramatize through concrete action, sensory detail, and subtext. Do not summarize emotion, "
        "explain the subtext, or state the theme outright — trust the reader.\n"
        "- Vary sentence length and structure. Favor strong verbs and specific nouns over adverbs and abstraction.\n"
        "- Write dialogue that carries character, tension, and information indirectly; avoid on-the-nose exposition.\n"
        "- Cut clichés, filler, and AI tells (e.g. 'little did they know', 'a testament to', "
        "reflexive over-explaining, neatly moralized endings, purple padding).\n"
        "- Honor continuity absolutely: names, timeline, established facts, and what each character can plausibly know. "
        "If the request conflicts with the established draft, follow continuity and flag the conflict in commentary.\n\n"
        "# Working method\n"
        "- Before drafting a substantial scene, plan it privately: what the scene must accomplish, where the tension "
        "lives, what image or gesture carries the subtext, where it should end. Never put the plan in your reply.\n"
        "- Draft a scene as one coherent whole (one append), not as fragments.\n"
        "- After appending a major scene, reread it once with fresh eyes (read_manuscript on the new range) and repair "
        "the weakest lines with replace_in_manuscript — flat verbs, over-explained emotion, rhythm that sags. "
        "One targeted polish pass, not endless fiddling.\n\n"
        + _tools_prompt_section()
        + "# Editing discipline\n"
        "- The saved manuscript is the authoritative draft; your tools edit it in place (every change is reversible).\n"
        "- Continue the story with append_to_manuscript; revise a specific part with replace_in_manuscript.\n"
        "- Make each edit flow seamlessly with the surrounding prose; never duplicate text that already exists.\n"
        "- If the writer selected a range to revise, replace exactly that range (use its text as 'find').\n"
        "- If the writer asks a question, asks for diagnosis, asks for options, or brainstorms without requesting a manuscript edit, "
        "make NO edit and answer directly in commentary.\n"
        "- If the request is ambiguous or would break continuity, make NO edit and ask in commentary instead.\n\n"
        "# Response format\n"
        "After applying your edits with the tools, reply with ONLY a commentary block:\n"
        "<commentary>\n"
        "A brief note: what you changed (appended vs revised, and which part), key choices, continuity assumptions, "
        "and any genuine question for the writer. Do NOT paste the manuscript prose here. "
        "No praise, no filler, no boilerplate caveats.\n"
        "</commentary>"
    )


def manuscript_context(project_id: int, selection_start: int | None, selection_end: int | None) -> str:
    manuscript = get_manuscript(project_id) or {}
    body = str(manuscript.get("body") or "")
    parts = [f"Manuscript character count: {len(body)}"]
    if body:
        tail = body[-MANUSCRIPT_TAIL_CHARS:]
        parts.append(
            f"Recent manuscript tail (chars {len(body) - len(tail)}–{len(body)}, already saved):\n" + tail
        )
    if selection_start is not None and selection_end is not None and body:
        start = max(0, min(selection_start, len(body)))
        end = max(start, min(selection_end, len(body)))
        selected = body[start:end]
        if selected:
            if len(selected) > MANUSCRIPT_SELECTION_LIMIT:
                selected = selected[:MANUSCRIPT_SELECTION_LIMIT] + "\n[selection truncated]"
            parts.append(f"Selected manuscript range {start}:{end}:\n{selected}")
    documents = list_documents(project_id)
    if documents:
        listing = "\n".join(
            f"- {str(d.get('title'))!r} (kind: {d.get('kind')}, {d.get('char_count')} chars)"
            for d in documents
        )
        parts.append("Background documents (read with read_document(title)):\n" + listing)
        # Pinned documents (the agent-maintained story synopsis and similar)
        # ride along in full so long-novel continuity doesn't depend on the
        # model choosing to re-read the manuscript every turn.
        pinned = [d for d in documents if str(d.get("kind") or "") == PINNED_DOC_KIND]
        for d in pinned[:PINNED_DOC_MAX_COUNT]:
            doc = get_document(project_id, int(d["id"]))
            content = str((doc or {}).get("content") or "").strip()
            if not content:
                continue
            if len(content) > PINNED_DOC_CHAR_LIMIT:
                content = content[:PINNED_DOC_CHAR_LIMIT] + "\n[pinned document truncated]"
            parts.append(f"Pinned document {str(d.get('title'))!r}:\n{content}")
    return "\n\n".join(parts)


def build_system_blocks(
    project: dict,
    project_id: int,
    selection_start: int | None,
    selection_end: int | None,
) -> list[dict]:
    return [
        {
            "type": "text",
            "text": build_base_system_prompt(project),
            "cache_control": WRITER_CACHE_CONTROL_1H,
        },
        {
            "type": "text",
            "text": "<manuscript_context>\n"
            + manuscript_context(project_id, selection_start, selection_end)
            + "\n</manuscript_context>",
            "cache_control": WRITER_CACHE_CONTROL_1H,
        },
    ]


def messages_for_model(
    project_id: int,
    user_prompt: str,
    selection_start: int | None = None,
    selection_end: int | None = None,
) -> list[dict]:
    rows = recent_messages(project_id, MAX_CONTEXT_MESSAGES)
    # rows are newest-first: apply the char budget from the newest backwards,
    # then restore chronological order.
    kept: list[dict] = []
    spent = 0
    for row in rows:
        if row.get("role") not in {"user", "assistant"}:
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        if len(kept) >= MIN_CONTEXT_MESSAGES and spent + len(content) > CONTEXT_CHAR_BUDGET:
            break
        kept.append({"role": row["role"], "content": content})
        spent += len(content)
    messages = list(reversed(kept))
    current_turn = "<user_request>\n" + user_prompt.strip() + "\n</user_request>"
    messages.append({"role": "user", "content": current_turn})
    return messages


def writer_error_message(provider_display: str, raw_error: str) -> str:
    text = str(raw_error or "").strip()
    lowered = text.lower()
    if "cancelled by server shutdown" in lowered:
        return (
            "The server restarted while this request was running, so the model run was cancelled. "
            "Manuscript edits the tools had already applied before the restart are saved; "
            "send the request again to continue from there."
        )
    if "provider stream produced no text/final event" in lowered:
        return (
            f"{provider_display} opened the request but then produced no stream data for "
            f"{WRITER_PROVIDER_IDLE_TIMEOUT_SEC}s. The provider connection stalled before "
            "any answer or tool call completed; nothing was written to the manuscript."
        )
    if "provider stream did not finalize" in lowered:
        return (
            f"{provider_display} stream sent partial data but did not finalize cleanly. "
            "The request was stopped before a complete answer could be saved."
        )
    if any(token in lowered for token in ("timeout", "connection", "network", "remoteprotocol", "readerror", "apierror")):
        return (
            f"{provider_display} connection failed while generating. This was a provider/network "
            "stream failure, not a manuscript tool failure; no assistant result was saved."
        )
    return f"{provider_display} request failed before completion: {text}"


_MANUSCRIPT_DELTA_RE = re.compile(
    r"<manuscript_delta>\s*(.*?)\s*</manuscript_delta>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENTARY_RE = re.compile(
    r"<commentary>\s*(.*?)\s*</commentary>",
    re.IGNORECASE | re.DOTALL,
)


def parse_writer_response(text: str) -> dict[str, str]:
    manuscript_match = _MANUSCRIPT_DELTA_RE.search(text)
    commentary_match = _COMMENTARY_RE.search(text)
    manuscript_text = manuscript_match.group(1).strip() if manuscript_match else ""
    commentary_text = commentary_match.group(1).strip() if commentary_match else ""
    if manuscript_match or commentary_match:
        remaining = _MANUSCRIPT_DELTA_RE.sub("", text)
        remaining = _COMMENTARY_RE.sub("", remaining).strip()
        if remaining and not commentary_text:
            commentary_text = remaining
    else:
        manuscript_text = text.strip()
    display_parts = []
    if manuscript_text:
        display_parts.append("Manuscript\n" + manuscript_text)
    if commentary_text:
        display_parts.append("Notes\n" + commentary_text)
    return {
        "manuscript_text": manuscript_text,
        "commentary_text": commentary_text,
        "display_text": "\n\n".join(display_parts) or text.strip(),
    }
