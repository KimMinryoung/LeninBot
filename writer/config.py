"""Writer runtime constants shared across the package."""

from __future__ import annotations

WRITER_DEFAULT_MAX_TOKENS = 20000
# Tool-use rounds: allow extended continuity searches and edit retries before
# the model writes. 1 = no tools (legacy). Each round is a priced model call.
WRITER_MAX_ROUNDS = 16
WRITER_IDLE_TIMEOUT_SEC = 240
WRITER_PROVIDER_IDLE_TIMEOUT_SEC = 70
# Tavily-backed web_search: enabled so the agent can actively research
# real-world specifics (period detail, geography, terminology) and distill the
# findings into background documents instead of guessing or re-searching.
WRITER_WEB_SEARCH_ENABLED = True
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

# Optional critic/line-edit pass (퇴고): a second model call after the main
# pass that line-edits only the spans this turn changed. Opt-in per request.
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
