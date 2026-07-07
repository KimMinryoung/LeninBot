"""Writer prompt assembly: system prompt, manuscript context, message history,
and response parsing."""

from __future__ import annotations

import re

from writer.config import (
    CONTEXT_CHAR_BUDGET,
    CONTEXT_WINDOW_QUANTUM_CHARS,
    DOC_STALENESS_NOTE_CHARS,
    MANUSCRIPT_OPENING_CHARS,
    MANUSCRIPT_SELECTION_LIMIT,
    MANUSCRIPT_TAIL_CHARS,
    MAX_CONTEXT_MESSAGES,
    MIN_CONTEXT_MESSAGES,
    PINNED_DOC_CHAR_LIMIT,
    PINNED_DOC_KIND,
    PINNED_DOC_MAX_COUNT,
    STYLE_DOC_CHAR_LIMIT,
    STYLE_DOC_KIND,
    STYLE_DOC_MAX_COUNT,
    WRITER_CACHE_CONTROL_1H,
    WRITER_CRITIC_MIN_CHANGED_CHARS,
    WRITER_CRITIC_SPAN_BUDGET_CHARS,
    WRITER_CRITIC_SPAN_CONTEXT_CHARS,
    WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
    WRITER_WEB_SEARCH_ENABLED,
)
from writer.documents import get_document, list_documents
from writer.store import get_manuscript, recent_messages, total_message_chars


def _tools_prompt_section() -> str:
    """Workflow-level tool and memory discipline. Per-tool usage lives in the
    tool schemas themselves (writer.tools); repeating it here would only pad
    the prompt."""
    research = (
        "- research_web: ONE specific real-world question per call; use the brief to get concrete details "
        "right, never to flatten prose into an encyclopedia. Save reusable findings into a "
        "'Research — <topic>' document (kind 'research') the same turn — a fact verified once is never "
        "searched twice.\n"
        if WRITER_WEB_SEARCH_ENABLED
        else ""
    )
    return (
        "# Tools — you edit the manuscript yourself\n"
        "Every manuscript change goes through append_to_manuscript or replace_in_manuscript. Reply text is "
        "never saved; manuscript prose never belongs in your reply; never narrate tool use. The "
        "<manuscript_state> block in the latest request IS the current saved draft (opening/tail excerpts, "
        "document listing, pinned documents) — don't re-read or re-search what you can already see, and never "
        "search for text you wrote this same turn. One or two well-chosen lookups beat a chain of guesses. "
        "The background documents are the authoritative reference for setting and plot facts — check them "
        "before inventing or contradicting a detail; never store manuscript prose in them.\n\n"
        "# Memory — the documents are your long-term memory\n"
        "Beyond pinned documents and the opening/tail excerpts, you know only what you wrote down or look up:\n"
        "- Before drafting, read the documents the scene depends on (characters present, location, open "
        "threads) unless already in context. Check facts instead of guessing.\n"
        "- Record durable story facts in the SAME turn they enter the story — new or changed character, "
        "place, object, relationship, backstory, timeline anchor — into 'Characters' / 'Places' / "
        "'Timeline' / 'Threads & setups' (create each when first needed). Dense factual notes, never prose.\n"
        "- Maintain a pinned 'Story so far' (kind 'pinned', under 4000 chars): events, character states, "
        "open threads, planted setups. It rides in your context every turn; keeping it current replaces "
        "re-reading the manuscript. When the listing marks a document STALE, refresh it this turn, after "
        "the manuscript work.\n"
        + research
        + "- Distill the writer's chat style feedback (a dislike, a manual rewrite of your prose, a praised "
        "passage) into the style guide document (kind 'style') the same turn, as a 금지→지향 contrast pair "
        "or a new exemplar. Feedback left in chat history is forgotten; the style guide is how you learn "
        "this writer's taste.\n\n"
    )


def build_base_system_prompt(project: dict) -> str:
    premise = str(project.get("premise") or "").strip() or "(No premise recorded yet.)"
    style_notes = str(project.get("style_notes") or "").strip() or "(No style notes recorded yet.)"
    title = str(project.get("title") or "Untitled").strip()
    return (
        "You are an elite fiction-writing collaborator on one writer's personal novel — not a chatbot, "
        "a craftsperson serving this manuscript. Prose quality is the only thing that matters.\n\n"
        f"Project title: {title}\n"
        f"Premise:\n{premise}\n\n"
        f"Style and continuity notes:\n{style_notes}\n\n"
        "# Craft\n"
        "- Write in the language, voice, POV, tense, and register the manuscript has established; never "
        "reset to a generic default voice.\n"
        "- Specificity is the engine of prose: the particular noun, the observed gesture, the sensory fact "
        "this character would notice. If a description could appear in anyone's novel, cut it.\n"
        "- Dramatize through action, sense, and subtext; never summarize emotion, explain subtext, or state "
        "the theme. Trust the reader.\n"
        "- Vary sentence length and shape deliberately: short sentences hit, long sentences carry. Strong "
        "verbs and specific nouns over adverbs and abstraction.\n"
        "- Dialogue carries character, tension, and information indirectly — no on-the-nose exposition.\n"
        "- Kill clichés and AI tells: 'little did they know', 'a testament to', trailing triads, "
        "breath/heartbeat shorthand for emotion, every paragraph ending on a mini-epiphany, repeated "
        "'not X but Y', neatly moralized endings, purple padding.\n"
        "- Honor continuity absolutely: names, timeline, established facts, what each character can know. "
        "If the request conflicts with the draft, follow continuity and flag it in commentary.\n\n"
        "# Korean\n"
        "Write Korean literary fiction, not translated English:\n"
        "- 어미: no monotonous -었다/-했다 chains — vary endings and structures (-ㄴ다, 명사형 마침, "
        "inversion, an occasional fragment) within tense discipline. Korean rhythm lives in the endings.\n"
        "- Dialogue: speech level (반말/존댓말 등) and 호칭 stay consistent per relationship; a shift is a "
        "dramatic event, used deliberately.\n"
        "- Kill 번역투: no 것이다 crutch, no ~의 chains, no 되어지다 double passives, no overused 그/그녀 — "
        "Korean drops known subjects. Prefer native verbs over 한자어+하다 when register allows.\n"
        "- 의성어/의태어: a scalpel, not a garnish — one precise 의태어 carries a gesture; three per "
        "paragraph are noise.\n"
        "- Match the draft's punctuation and quote style exactly (this also keeps replace_in_manuscript "
        "reliable).\n\n"
        "# Style guide (when present)\n"
        "A document of kind 'style' is the binding calibration for this project's prose:\n"
        "- Exemplars: absorb their rhythm, sentence-length distribution, 어미 habits, image density, and "
        "emotional temperature, then write NEW sentences in that key — never copy or paraphrase them.\n"
        "- 금지→지향 contrast pairs are the writer's own corrections; they outrank every craft rule above. "
        "Generalize the principle behind each pair.\n"
        "- Operational rules are hard constraints. Where the style guide and the draft's voice disagree, "
        "the style guide wins for NEW prose; flag the tension in commentary.\n\n"
        "# Scene craft\n"
        "- A scene is a unit of change: someone wants something, meets resistance, the situation turns. "
        "Know the turn before you draft. Enter late, leave early.\n"
        "- The emotional line must turn too: within the scene, feeling contradicts or betrays itself at "
        "least once — unease inside joy, relief that curdles, self-deception rising with dread. One feeling "
        "from first line to last is a flat scene no prose can save.\n"
        "- A scene is a chord in the whole work: know what it plants or pays off ('Threads & setups', the "
        "outline). The strongest scenes carry dramatic irony — the reader hearing what the character "
        "cannot — rather than standing as self-contained vignettes.\n"
        "- Open inside motion or tension, not weather or waking (unless the draft demands it); end on a "
        "concrete image, gesture, or line that carries the aftertaste — never on summary or stated feeling.\n"
        "- Interiority as free indirect discourse in the character's own diction — not tagged 'she thought' "
        "blocks, not explanation of what the scene already shows.\n"
        "- Pacing: expand time at the highest pressure (beat-by-beat perception), compress hard between; "
        "paragraph length moves with the scene's pulse.\n\n"
        "# Method\n"
        "- Plan a substantial scene privately before drafting: what it must accomplish, where the tension "
        "lives, where the emotional line turns, what it plants or pays off, what image carries the subtext, "
        "where it ends. The plan never appears in your reply.\n"
        "- Draft the scene as one coherent whole (one append), then reread it fresh (read_manuscript on the "
        "new range) and repair the weakest lines with replace_in_manuscript — one targeted polish pass, not "
        "endless fiddling.\n"
        "- These standards are instincts, not a checklist: never sacrifice a living sentence, or this "
        "novel's voice, to satisfy a rule mechanically.\n\n"
        + _tools_prompt_section()
        + "# Editing discipline\n"
        "- Continue the story with append_to_manuscript; revise with replace_in_manuscript. Each edit flows "
        "seamlessly with the surrounding prose; never duplicate existing text.\n"
        "- If the writer selected a range, replace exactly that range (use its text as 'find').\n"
        "- If the writer asks a question, wants diagnosis or options, or brainstorms — or the request is "
        "ambiguous or would break continuity — make NO edit; answer or ask in commentary.\n\n"
        "# Response format\n"
        "After your tool edits, reply with ONLY:\n"
        "<commentary>\n"
        "What you changed and where, key choices, continuity assumptions, any genuine question for the "
        "writer. No manuscript prose, no praise, no filler.\n"
        "</commentary>"
    )


def manuscript_context(
    project_id: int,
    selection_start: int | None,
    selection_end: int | None,
    include_excerpts: bool = True,
) -> str:
    """The VOLATILE manuscript state: counts, opening/tail excerpts, selection,
    document listing (with STALE nudges), and pinned documents in full.

    This changes on almost every turn (any edit moves the tail and the doc
    listing's char counts), so it is deliberately kept OUT of the cached
    system blocks and injected into the current turn's user message instead —
    a change here must never invalidate the cached prompt prefix. The style
    guide lives in build_system_blocks, not here, for the same reason.

    include_excerpts=False drops the opening/tail/selection excerpts — the
    author-revision pass gets its working text as changed-passage windows and
    only needs the document state around it.
    """
    manuscript = get_manuscript(project_id) or {}
    body = str(manuscript.get("body") or "")
    parts = [f"Manuscript character count: {len(body)}"]
    if include_excerpts:
        # Opening excerpt for voice/style calibration — only when the opening
        # cannot overlap the tail, so the two blocks never duplicate text.
        if len(body) > MANUSCRIPT_TAIL_CHARS + MANUSCRIPT_OPENING_CHARS:
            parts.append(
                f"Manuscript opening (chars 0–{MANUSCRIPT_OPENING_CHARS}, for voice/style calibration only — "
                "already saved):\n" + body[:MANUSCRIPT_OPENING_CHARS]
            )
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
        lines = []
        for d in documents:
            line = f"- {str(d.get('title'))!r} (kind: {d.get('kind')}, {d.get('char_count')} chars"
            mark = d.get("manuscript_chars_at_update")
            if mark is not None and len(body) - int(mark) >= DOC_STALENESS_NOTE_CHARS:
                line += (
                    f"; STALE: last updated when the manuscript was {int(mark)} chars, "
                    f"it has since grown to {len(body)}"
                )
            line += ")"
            lines.append(line)
        parts.append("Background documents (read with read_document(title)):\n" + "\n".join(lines))
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


def manuscript_state_block(
    project_id: int,
    selection_start: int | None = None,
    selection_end: int | None = None,
    include_excerpts: bool = True,
) -> str:
    """manuscript_context wrapped for injection into a user message."""
    return (
        "<manuscript_state>\n"
        + manuscript_context(project_id, selection_start, selection_end, include_excerpts)
        + "\n</manuscript_state>"
    )


def style_guide_parts(project_id: int, documents: list[dict] | None = None) -> list[str]:
    """Full-text style-guide blocks (documents of kind 'style'), rendered for
    injection into the writer, diagnosis, and line-edit contexts."""
    if documents is None:
        documents = list_documents(project_id)
    parts: list[str] = []
    style_docs = [d for d in documents if str(d.get("kind") or "") == STYLE_DOC_KIND]
    for d in style_docs[:STYLE_DOC_MAX_COUNT]:
        doc = get_document(project_id, int(d["id"]))
        content = str((doc or {}).get("content") or "").strip()
        if not content:
            continue
        if len(content) > STYLE_DOC_CHAR_LIMIT:
            content = content[:STYLE_DOC_CHAR_LIMIT] + "\n[style guide truncated]"
        parts.append(
            f"Style guide {str(d.get('title'))!r} (binding prose calibration for this project — "
            "absorb its rhythm and rules, never copy its sentences):\n" + content
        )
    return parts


def build_system_blocks(project: dict, project_id: int) -> list[dict]:
    """Cached system blocks for the writer and author-revision passes: ONLY
    content that is stable across turns (the craft prompt and the style guide),
    so the 1h-TTL prefix cache actually hits between turns. The per-turn
    manuscript state travels in the current user message instead (see
    messages_for_model); putting it here re-wrote the whole prefix at 2x input
    price every turn and read back nothing (2026-07-07: ~$1.17 of every $1.30
    Fable turn was dead cache writes)."""
    blocks = [
        {
            "type": "text",
            "text": build_base_system_prompt(project),
            "cache_control": WRITER_CACHE_CONTROL_1H,
        }
    ]
    guide = style_guide_parts(project_id)
    if guide:
        blocks.append({
            "type": "text",
            "text": "\n\n".join(guide),
            "cache_control": WRITER_CACHE_CONTROL_1H,
        })
    return blocks


def build_critic_system_blocks(project: dict) -> list[dict]:
    """System blocks for the legacy line-edit (퇴고) mode. Block 1 is stable
    across projects and turns; block 2 carries the project header. Both cached."""
    base = (
        "You are a ruthless, taste-perfect line editor (퇴고 담당) for one writer's novel. Another pass just "
        "drafted the passages shown to you; make them read like finished literary prose. You never add "
        "scenes, restructure plot, or append — only refine what exists.\n\n"
        "# Method\n"
        "- Fix only what is weak: flat verbs; -었다/-했다 chains and repeated sentence shapes; 번역투 "
        "(것이다 crutch, ~의 chains, double passives, overused 그/그녀); over-explained emotion or subtext; "
        "word/image echoes; dialogue register slips (반말/존댓말, 호칭); sagging rhythm; continuity slips. "
        "read_manuscript around the offsets or read_document when you need context.\n"
        "- Each fix is one surgical replace_in_manuscript: 'find' copied exactly from the saved text, unique, "
        "minimum span. Typically 3–10 replacements.\n"
        "- Leave strong passages alone — restraint is taste. Preserve the author's voice absolutely; you "
        "polish, you do not rewrite into your own style.\n\n"
        "# Response format\n"
        "After your replacements, reply with ONLY:\n"
        "<commentary>\n"
        "One short paragraph: what you tightened and why. No praise, no filler.\n"
        "</commentary>"
    )
    header = _project_header_with_style_guide(project)
    return [
        {"type": "text", "text": base, "cache_control": WRITER_CACHE_CONTROL_1H},
        {"type": "text", "text": header, "cache_control": WRITER_CACHE_CONTROL_1H},
    ]


def _project_header_with_style_guide(project: dict) -> str:
    """Project header block for second-pass prompts, with the style guide
    appended so 문체 fidelity is checkable outside the main writer context."""
    premise = str(project.get("premise") or "").strip() or "(No premise recorded yet.)"
    style_notes = str(project.get("style_notes") or "").strip() or "(No style notes recorded yet.)"
    title = str(project.get("title") or "Untitled").strip()
    header = (
        f"Project title: {title}\n"
        f"Premise:\n{premise}\n\n"
        f"Style and continuity notes:\n{style_notes}"
    )
    try:
        pid = int(project.get("id"))
    except (TypeError, ValueError):
        pid = None
    if pid is not None:
        guide = style_guide_parts(pid)
        if guide:
            header += "\n\n" + "\n\n".join(guide)
    return header


def changed_windows(body: str, edits: list[dict]) -> list[tuple[int, int]]:
    """Merge this turn's edit spans into context-expanded excerpt windows,
    ordered front to back and capped by the total excerpt budget. Empty when
    the turn changed too little to justify a second model call."""
    spans = []
    changed_total = 0
    for edit in edits:
        try:
            start = max(0, min(int(edit["start"]), len(body)))
            end = max(start, min(int(edit["end"]), len(body)))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            spans.append((start, end))
            changed_total += end - start
    if not spans or changed_total < WRITER_CRITIC_MIN_CHANGED_CHARS:
        return []
    # Expand each span into a context window, then merge overlaps.
    windows = sorted(
        (max(0, s - WRITER_CRITIC_SPAN_CONTEXT_CHARS), min(len(body), e + WRITER_CRITIC_SPAN_CONTEXT_CHARS))
        for s, e in spans
    )
    merged: list[list[int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    # Enforce the total excerpt budget: keep the largest windows, then restore
    # document order so the second pass reads front to back.
    kept: list[list[int]] = []
    budget = WRITER_CRITIC_SPAN_BUDGET_CHARS
    for window in sorted(merged, key=lambda w: w[1] - w[0], reverse=True):
        size = window[1] - window[0]
        if size <= budget:
            kept.append(window)
            budget -= size
    kept.sort()
    return [(int(s), int(e)) for s, e in kept]


def _changed_passage_blocks(body: str, windows: list[tuple[int, int]], suffix: str) -> list[str]:
    return [
        f"Changed passage {i} (manuscript chars {start}–{end}; {suffix}):\n" + body[start:end]
        for i, (start, end) in enumerate(windows, 1)
    ]


def critic_user_message(body: str, edits: list[dict]) -> str | None:
    """Render this turn's changed spans as offset-labeled excerpts for the
    legacy line-edit pass. Returns None when the turn changed too little to
    justify a second model call."""
    windows = changed_windows(body, edits)
    if not windows:
        return None
    parts = [
        "Line-edit ONLY the changed passages below. Each excerpt is saved manuscript text; "
        "copy 'find' arguments verbatim from it. Then reply with <commentary>."
    ]
    parts.extend(_changed_passage_blocks(body, windows, "your edits must match this saved text"))
    return "\n\n".join(parts)


# Exact full-reply marker the diagnosis stage uses to say "nothing worth a
# note" — it short-circuits the author-revision call entirely.
DIAGNOSIS_PASS_MARKER = "PASS"


def build_diagnosis_system_blocks(project: dict) -> list[dict]:
    """System blocks for stage 1 of the diagnose→revise 퇴고: a read-only
    literary editor whose notes go back to the (stronger) author model."""
    base = (
        "You are a senior literary editor (문학 편집자) reviewing the passages a novelist drafted or revised "
        "this session. You DIAGNOSE; you never rewrite. Your notes go back to the author — a stronger prose "
        "stylist than you — who does the revising. Your value is a fresh, unsparing outside eye.\n\n"
        "# Check, in order of leverage\n"
        "1. 구조 — does the scene TURN (want → resistance → changed situation)? A scene ending where it began "
        "is the deepest flaw. Entry late enough; exit on a resonant image/gesture/line, not summary or stated "
        "feeling. Check the EMOTIONAL line separately: does feeling reverse or contradict itself at least "
        "once (기쁨 속의 불안, 자기기만), or run straight (기쁨→기쁨→기쁨)? Linear emotion with polished "
        "sentences is still a flat scene — name it.\n"
        "2. 극화 — emotion told, subtext explained, generic detail where one exact character-filtered detail "
        "belongs, missed senses at the highest-pressure beats.\n"
        "3. 리듬 — -었다/-했다 chains, one repeated sentence shape, paragraphs that ignore the scene's pulse, "
        "inverted pacing (rushed climax, dawdling transition).\n"
        "4. 문장 — 번역투 (것이다 crutch, ~의 chains, double passives, overused 그/그녀), clichés and AI "
        "tells, word/image echoes, flat verbs, dialogue register slips (반말/존댓말, 호칭).\n"
        "5. 문체 — judge against the style guide below when present: rhythm, 어미 habits, image density, "
        "emotional temperature, 금지→지향 pairs.\n"
        "6. 연속성·설계 — contradictions with the context or background documents (read_document to verify). "
        "Read 'Threads & setups' / 'Story so far' / the outline when they exist: does the passage plant or "
        "pay off anything, carry earned dramatic irony — or is it a self-contained vignette connected to "
        "nothing? An isolated well-made scene is a note, not music.\n\n"
        "# Method\n"
        "Judge only the changed passages (read_manuscript around the offsets for context). Be concrete and "
        "severe — praise is useless; a missed real weakness is your only failure mode. Never write "
        "replacement sentences: name the problem and the direction of repair, not the wording.\n\n"
        "# Response format\n"
        "ONLY a numbered diagnosis, most damaging first, at most 8 items:\n"
        '1. [구조|극화|리듬|문장|문체|연속성] "short verbatim quote locating the spot" — problem, then '
        "direction of repair (no rewritten text).\n"
        "Then one line — 남길 것: what is working and must survive revision.\n"
        f"If nothing rises to a real note, reply exactly {DIAGNOSIS_PASS_MARKER}."
    )
    return [
        {"type": "text", "text": base, "cache_control": WRITER_CACHE_CONTROL_1H},
        {"type": "text", "text": _project_header_with_style_guide(project), "cache_control": WRITER_CACHE_CONTROL_1H},
    ]


def diagnosis_user_message(body: str, edits: list[dict]) -> str | None:
    """Stage-1 user message: the changed excerpts to diagnose. None when the
    turn changed too little for a 퇴고 pass."""
    windows = changed_windows(body, edits)
    if not windows:
        return None
    parts = [
        "Diagnose the changed passages below (this session's new or revised prose). "
        f"Reply with the numbered diagnosis, or exactly {DIAGNOSIS_PASS_MARKER}."
    ]
    parts.extend(_changed_passage_blocks(body, windows, "read_manuscript nearby for more context"))
    return "\n\n".join(parts)


def revision_user_message(
    body: str,
    edits: list[dict],
    diagnosis_notes: str | None,
    manuscript_state: str | None = None,
) -> str | None:
    """Stage-2 user message for the MAIN model: revise this turn's passages as
    the author, guided by (but not subordinate to) the editor's diagnosis.
    With diagnosis_notes=None (stage 1 failed) it asks for self-diagnosis.
    manuscript_state carries the POST-edit document state (listing with STALE
    markers, pinned docs) — this stage owns the story-bible refresh."""
    windows = changed_windows(body, edits)
    if not windows:
        return None
    notes = (diagnosis_notes or "").strip()
    if notes:
        notes_block = "[Editor's diagnosis]\n" + notes
        opening = (
            "An outside editor has diagnosed the passages you drafted this session (notes below). "
            "Now revise them as the author."
        )
    else:
        notes_block = (
            "[Editor's diagnosis]\n(The editor pass failed this turn. Diagnose the passages yourself, with an "
            "editor's severity, before revising: 구조 — does the scene turn, and does it end on an image or on "
            "summary; 극화 — told emotion, explained subtext, generic detail; 리듬 — 어미 monotony, repeated "
            "sentence shapes; 문장 — 번역투, clichés, echoes, flat verbs; 문체 — fidelity to the style guide.)"
        )
        opening = "Reread the passages you drafted this session with fresh, hostile eyes, then revise them as the author."
    parts = []
    if manuscript_state:
        parts.append(manuscript_state)
    parts += [
        "<revision_request>\n" + opening + "\n"
        "- Judge each note: apply what is right, REJECT what would flatten the voice or misread the intent — "
        "obedience is not revision. Account for rejections in commentary.\n"
        "- A structural note (scene turn, emotional line, ending, pacing) may deserve rewriting the whole "
        "passage in one replace — never patch a structural flaw with word swaps. Line notes take the minimum "
        "span. Manuscript changes are replace_in_manuscript only; never append or expand the scene's scope.\n"
        "- After your revisions, if the turn's changes left 'Story so far' or another living document STALE "
        "(or your rewrites changed facts they record), refresh them with save_document — dense factual "
        "notes, never prose.\n"
        "- Hold this novel's voice and the style guide absolutely; the notes point at weaknesses, not toward "
        "a more generic register.\n\n"
        + notes_block
    ]
    parts.extend(
        _changed_passage_blocks(body, windows, "saved manuscript text — copy 'find' arguments verbatim from here")
    )
    parts.append(
        "</revision_request>\n\n"
        "After your replacements, reply with ONLY <commentary>: one short paragraph — what you accepted, "
        "what you rejected and why. No praise, no filler."
    )
    return "\n\n".join(parts)


# Light main models (observed: DeepSeek V4 Flash, 2026-07-07) sometimes "write"
# a scene only in their reasoning, then reply claiming the edit — a phantom
# edit that saves nothing. A per-turn reminder at the end of the current
# request keeps the tool contract in the model's most recent context.
TOOL_DISCIPLINE_REMINDER = (
    "<turn_reminder>Reply text is NEVER saved to the manuscript. If this request asks for story "
    "text to be added or revised, you MUST apply it with append_to_manuscript / "
    "replace_in_manuscript tool calls BEFORE replying, then describe what you did in "
    "<commentary>. Claiming an edit without having made the tool call means nothing was "
    "written.</turn_reminder>"
)


def with_tool_discipline_reminder(messages: list[dict]) -> list[dict]:
    """Append the tool-discipline reminder to the current turn's user message."""
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    last["content"] = str(last.get("content") or "") + "\n\n" + TOOL_DISCIPLINE_REMINDER
    out[-1] = last
    return out


def messages_for_model(
    project_id: int,
    user_prompt: str,
    selection_start: int | None = None,
    selection_end: int | None = None,
) -> list[dict]:
    rows = recent_messages(project_id, MAX_CONTEXT_MESSAGES)
    # Quantized history window: the window start is anchored to absolute char
    # positions in the whole conversation and advances only in
    # CONTEXT_WINDOW_QUANTUM_CHARS jumps. A plain per-turn char budget slides
    # the window every turn, changing the message prefix on every request and
    # defeating prompt caching for the entire history block. With the quantum,
    # the kept history spans (budget - quantum, budget] chars and its start
    # stays byte-identical until the conversation grows another quantum.
    total = total_message_chars(project_id)
    threshold = 0
    over = total - CONTEXT_CHAR_BUDGET
    if over > 0:
        threshold = -(-over // CONTEXT_WINDOW_QUANTUM_CHARS) * CONTEXT_WINDOW_QUANTUM_CHARS
    # rows are newest-first: keep every message that starts at or after the
    # threshold position, with a minimum-message floor, then restore
    # chronological order.
    kept: list[dict] = []
    suffix = 0
    for row in rows:
        if row.get("role") not in {"user", "assistant"}:
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        suffix += len(content)
        if len(kept) >= MIN_CONTEXT_MESSAGES and total - suffix < threshold:
            break
        kept.append({"role": row["role"], "content": content})
    messages = list(reversed(kept))
    # The volatile manuscript state rides in the CURRENT turn only. It is
    # never persisted with the prompt (store keeps the bare prompt), so past
    # turns replay byte-identically and the history prefix stays cacheable.
    current_turn = (
        manuscript_state_block(project_id, selection_start, selection_end)
        + "\n\n<user_request>\n" + user_prompt.strip() + "\n</user_request>"
    )
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
    if "provider stream produced no text/final event" in lowered or "provider stream produced no events" in lowered:
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


# DeepSeek models occasionally leak raw DSML wrapper tokens (e.g.
# "<｜｜DSML｜｜commentary>") into their reply text; strip them before parsing
# so they never reach the persisted commentary (observed in production
# 2026-07-07, message 819).
_DSML_TOKEN_RE = re.compile(r"</?｜｜DSML｜｜[^>\n]*>")

_MANUSCRIPT_DELTA_RE = re.compile(
    r"<manuscript_delta>\s*(.*?)\s*</manuscript_delta>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENTARY_RE = re.compile(
    r"<commentary>\s*(.*?)\s*</commentary>",
    re.IGNORECASE | re.DOTALL,
)


def parse_writer_response(text: str) -> dict[str, str]:
    text = _DSML_TOKEN_RE.sub("", text or "")
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
