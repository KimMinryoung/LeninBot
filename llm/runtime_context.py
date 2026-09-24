"""Per-turn runtime context shared by the Telegram orchestrator and workers.

Lives outside ``telegram.bot`` so other processes (browser worker, A2A API)
can render the same runtime prelude and read the per-coroutine task context
without importing the aiogram bot module and its import-time side effects.
``telegram.bot`` re-exports these names under their historical private
aliases.
"""

import contextvars
from datetime import datetime

from shared import KST

# Per-coroutine task context — allows concurrent tasks to know their own task_id
current_task_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_task_ctx", default=None)


def current_datetime_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


def format_current_model_context(kind: str = "chat", provider: str = "claude") -> str:
    """Format runtime-selected model info for prompt/context injection.

    Leads with the human-readable product name ("Claude Opus 5", "GPT-5.6 Sol")
    so self-identification works cleanly; the raw API id and tier stay available
    as secondary metadata. `provider` controls BOTH the surface form (XML for
    Claude, Markdown elsewhere) AND which tier map the model is resolved from —
    so an agent pinned to Claude while config.provider="openai" still surfaces
    the real Claude model it's running on, not the chat-side GPT.
    """
    from bot_config import get_current_model_selection

    sel = get_current_model_selection(kind, provider_override=provider)
    name = sel["display_name"]
    model_id = sel["model_id"]
    tier = sel["tier"]
    if provider == "claude":
        return (
            f"<current-model tier=\"{tier}\" id=\"{model_id}\">{name}</current-model>"
        )
    return f"- **Current Model**: {name} (id: `{model_id}`, tier: {tier})"


def build_runtime_prelude(provider: str = "claude", kind: str = "chat") -> str:
    """Render the volatile runtime header (time + active model).

    Goes at the top of extra context so the system prompt itself stays
    byte-identical across turns (prompt-cache friendly). Returned without any
    leading/trailing whitespace — separator insertion is the caller's job
    (`join_context_blocks`).
    """
    current_time = current_datetime_str()
    current_model = format_current_model_context(kind, provider)
    if provider == "claude":
        return (
            f"<runtime>\n<current-time>{current_time}</current-time>\n"
            f"{current_model}\n</runtime>"
        )
    return (
        f"### Runtime\n"
        f"- **Current Time**: {current_time}\n"
        f"{current_model}"
    )


def join_context_blocks(*blocks: str) -> str:
    """Concatenate non-empty context blocks with a blank-line separator.

    Every block (XML tag group or Markdown section) gets bounded by an actual
    blank line in the output, which both CommonMark/GFM parsers and LLM
    attention treat as a real section break. Empty or whitespace-only blocks
    are skipped, so callers can unconditionally pass optional context slots.
    """
    cleaned = [b.strip() for b in blocks if b and b.strip()]
    return "\n\n".join(cleaned)


def merge_runtime_context_into_last_user(
    messages: list[dict], runtime_context: str
) -> list[dict]:
    """Attach per-turn runtime metadata beside the trailing user request.

    Placing volatile context (time, mission, alerts, …) immediately before the
    current user query — rather than at the start of the message array — keeps
    the history prefix byte-stable across turns so prompt caching (Claude
    ephemeral / OpenAI automatic) keeps hitting. Returns a new list; the caller's
    list and its inner dicts are left untouched.
    """
    if not runtime_context or not runtime_context.strip():
        return list(messages)

    from llm.execution_context import attach_context, context_record
    return attach_context(messages, [context_record(
        "runtime_state", "telegram_runtime", runtime_context.strip(),
        temporal_scope="current turn",
    )])
