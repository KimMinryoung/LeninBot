"""Writer runtime constants shared across the package."""

from __future__ import annotations

WRITER_DEFAULT_MAX_TOKENS = 20000
# Tool-use rounds: allow extended continuity searches and edit retries before
# the model writes. 1 = no tools (legacy). Each round is a priced model call.
WRITER_MAX_ROUNDS = 16
WRITER_IDLE_TIMEOUT_SEC = 240
# Max zero-event silence on a provider stream before the call is treated as
# stalled and retried. Fable's adaptive thinking emits NO stream events while
# it thinks (measured 2026-07-07: a 29.7s silent gap on a trivial 173-token
# prompt; heavy writer turns scale with thinking depth) — at 70s the watchdog
# deterministically killed hard effort-high turns three retries in a row.
# True dead connections still fail via the SDK's own network timeout.
WRITER_PROVIDER_IDLE_TIMEOUT_SEC = 240
# Web research: enabled so the agent can actively research real-world
# specifics (period detail, geography, terminology) and distill the findings
# into background documents instead of guessing or re-searching. The main
# model does not call Tavily directly — it delegates through the research_web
# tool to a light DeepSeek sub-agent that searches and returns a digest
# (raw search snippets never enter heavy-model context).
WRITER_WEB_SEARCH_ENABLED = True

# Delegated research sub-agent (research_web): light model, few rounds, and a
# hard wall-clock cap so a stuck sub-run can never hang the main writer turn.
WRITER_RESEARCH_MAX_ROUNDS = 6
WRITER_RESEARCH_MAX_TOKENS = 3000
WRITER_RESEARCH_BUDGET_USD = 1.0
WRITER_RESEARCH_TIMEOUT_SEC = 240
WRITER_CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}

MAX_CONTEXT_MESSAGES = 80
# Character budget for conversation history sent to the model (~10k tokens).
# 80 messages of long legacy replies could reach 6-figure token counts; with
# Fable-tier input pricing the history is the dominant per-turn cost, and old
# commentary adds little craft value. Newest messages win; at least the last
# MIN_CONTEXT_MESSAGES survive the budget so short-term context never drops.
CONTEXT_CHAR_BUDGET = 30000
MIN_CONTEXT_MESSAGES = 8

MANUSCRIPT_TAIL_CHARS = 16000
MANUSCRIPT_SELECTION_LIMIT = 20000
# Opening excerpt for voice/style calibration, injected only when the
# manuscript is long enough that it cannot overlap the tail.
MANUSCRIPT_OPENING_CHARS = 3500

# The document listing flags a document as lagging when the manuscript has
# grown at least this many chars since the document was last saved — the
# in-context nudge that keeps the agent's story bible current.
DOC_STALENESS_NOTE_CHARS = 2000

# Documents with kind 'pinned' (e.g. an agent-maintained 'Story so far'
# synopsis) are injected in full into the manuscript context block.
PINNED_DOC_KIND = "pinned"
PINNED_DOC_CHAR_LIMIT = 8000
PINNED_DOC_MAX_COUNT = 3

# Style guide documents (kind 'style'): the project's prose calibration
# target — exemplar passages, contrast pairs (금지→지향), operational rules.
# Injected in full into the writer, diagnosis, and line-edit contexts and
# treated as binding calibration (absorb rhythm/diction, never copy).
STYLE_DOC_KIND = "style"
STYLE_DOC_CHAR_LIMIT = 12000
STYLE_DOC_MAX_COUNT = 2

# Optional 퇴고 pass, opt-in per request. Two modes:
# - 'diagnose_revise' (default): a light editor model DIAGNOSES the changed
#   passages (scene structure, dramatization, rhythm, language, style
#   fidelity) without touching the text, then the MAIN model revises as the
#   author — free to reject notes that would hurt the voice. Critique flows
#   to the strongest model instead of a weaker model rewriting its prose.
# - 'line_edit': legacy single light-model line edit of the changed spans.
WRITER_REVISION_MODE = "diagnose_revise"
WRITER_DIAGNOSIS_MAX_TOKENS = 4000
WRITER_DIAGNOSIS_MAX_ROUNDS = 5
# Read-only surface for the diagnosis stage: it reports, it never edits.
WRITER_DIAGNOSIS_TOOL_NAMES = frozenset({
    "read_manuscript",
    "search_manuscript",
    "read_document",
})
WRITER_REVISION_MAX_TOKENS = 16000
WRITER_REVISION_MAX_ROUNDS = 10

WRITER_CRITIC_MAX_TOKENS = 8000
WRITER_CRITIC_MAX_ROUNDS = 8
# Skip the critic when the main pass changed fewer characters than this.
WRITER_CRITIC_MIN_CHANGED_CHARS = 600
# Context included around each changed span in the critic's excerpts.
WRITER_CRITIC_SPAN_CONTEXT_CHARS = 700
# Total excerpt budget across all changed-span windows.
WRITER_CRITIC_SPAN_BUDGET_CHARS = 24000
# Strict subset of the system.writer gateway profile — the critic can read
# and surgically replace, never append or write documents.
WRITER_CRITIC_TOOL_NAMES = frozenset({
    "read_manuscript",
    "search_manuscript",
    "replace_in_manuscript",
    "read_document",
})
