"""Delegated web research: the research_web tool.

The heavy writer model never reads raw Tavily output. It calls
research_web(question) and a light DeepSeek sub-agent runs the actual
web_search rounds, then returns a distilled factual brief — the only text
that enters heavy-model context. Falls back to the main writer model when
DeepSeek is unconfigured (see writer.models.resolve_light_model)."""

from __future__ import annotations

import asyncio
import logging

from writer.config import (
    WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
    WRITER_RESEARCH_BUDGET_USD,
    WRITER_RESEARCH_MAX_ROUNDS,
    WRITER_RESEARCH_MAX_TOKENS,
    WRITER_RESEARCH_TIMEOUT_SEC,
)
from writer.models import WRITER_RESEARCH_CHOICE, resolve_writer_model
from writer.runs import broadcast_run_event, record_run_cost

logger = logging.getLogger(__name__)

RESEARCH_TOOL_SPEC = {
    "name": "research_web",
    "description": (
        "Delegate real-world research to a fast assistant that searches the web and returns a distilled "
        "factual brief (facts, names, numbers, terminology, source URLs). Ask ONE specific question per call "
        "and say what the story needs it for, e.g. 'What did a 1920s Moscow tram interior look like? Needed "
        "for sensory detail in a night scene.' Use it whenever accuracy would ground the fiction; you never "
        "search the web yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "One specific research question, including what the fiction needs from the answer.",
            },
        },
        "required": ["question"],
    },
}

_RESEARCH_SYSTEM_PROMPT = (
    "You are a fast research assistant embedded in a fiction-writing workspace. The novelist's writing "
    "model delegates one real-world question to you per run.\n"
    "- Use web_search to answer it: start with one well-chosen query, refine or add queries only if the "
    "first results don't settle the question. Prefer search_depth='advanced' when digging into one "
    "specific point.\n"
    "- Then reply with a compact research brief: dense factual notes (bullet style) carrying the concrete "
    "specifics a novelist can use — exact names, dates, numbers, period terminology, sensory-relevant "
    "detail. No prose, no filler, no restating the question.\n"
    "- Write the brief in the language the question was asked in.\n"
    "- Include the source URL after each fact or fact cluster.\n"
    "- Mark uncertainty explicitly ('불확실:' / 'uncertain:') instead of guessing; say so plainly when the "
    "searches did not answer the question."
)


def _web_search_tooling() -> tuple[dict, dict] | None:
    """The main runtime's Tavily web_search (schema + handler), or None."""
    try:
        from runtime_tools.registry import TOOLS as _RT_TOOLS, TOOL_HANDLERS as _RT_HANDLERS

        spec = next((t for t in _RT_TOOLS if t.get("name") == "web_search"), None)
        handler = _RT_HANDLERS.get("web_search")
        if spec and handler:
            return spec, handler
    except Exception:
        logger.exception("writer research: web_search tooling unavailable")
    return None


def build_research_handler(project_id: int, resolve_fallback):
    """Async research_web handler bound to one project.

    resolve_fallback() lazily resolves the main writer model tuple used when
    DeepSeek is unavailable; deferred so the handler (memoized per project)
    never pins a client resolved at build time.
    """

    async def _handle_research(question: str) -> str:
        question = (question or "").strip()
        if not question:
            return "No question provided; nothing researched."
        tooling = _web_search_tooling()
        if tooling is None:
            return "Research failed: web search is not available on this server."
        web_spec, web_handler = tooling
        # Light tier first; the main writer model only as a resolved-lazily
        # fallback so an unconfigured provider degrades to an error string,
        # never an exception out of the tool handler.
        try:
            client, model, display, extra = resolve_writer_model(WRITER_RESEARCH_CHOICE)
        except (ValueError, RuntimeError):
            try:
                client, model, display, extra = resolve_fallback()
            except Exception as exc:
                logger.exception("writer research: no model provider available project_id=%s", project_id)
                return f"Research failed: model provider unavailable ({exc})."

        async def on_progress(event: str, detail: str):
            if event in ("tool_call", "tool_result") and detail:
                await broadcast_run_event(project_id, {"type": "tool", "content": f"[리서치] {detail}"})
            else:
                # Any sub-agent progress keeps the main run's idle clock alive.
                await broadcast_run_event(project_id, {"type": "ping"})

        budget_tracker: dict = {}
        try:
            from claude_loop import chat_with_tools

            brief = await asyncio.wait_for(
                chat_with_tools(
                    [{"role": "user", "content": question}],
                    client=client,
                    model=model,
                    tools=[web_spec],
                    tool_handlers={"web_search": web_handler},
                    system_prompt=_RESEARCH_SYSTEM_PROMPT,
                    max_rounds=WRITER_RESEARCH_MAX_ROUNDS,
                    max_tokens=WRITER_RESEARCH_MAX_TOKENS,
                    budget_usd=WRITER_RESEARCH_BUDGET_USD,
                    budget_tracker=budget_tracker,
                    on_progress=on_progress,
                    agent_name="writer_research",
                    provider_idle_timeout_sec=WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
                    **extra,
                ),
                timeout=WRITER_RESEARCH_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            return (
                f"Research timed out after {WRITER_RESEARCH_TIMEOUT_SEC}s. "
                "Proceed without it or retry with a narrower question."
            )
        except Exception as exc:
            logger.exception("writer research sub-run failed project_id=%s", project_id)
            return f"Research failed ({exc}). Proceed without it or retry with a narrower question."
        finally:
            record_run_cost(project_id, "research_web", budget_tracker.get("total_cost"))
        brief = (brief or "").strip()
        if not brief:
            return "Research returned no usable brief. Proceed without it or retry with a narrower question."
        # The brief distills untrusted web content: keep the external-provenance
        # envelope (and its tag neutralization) when it reaches the main model.
        from provenance.runtime import _wrap_external

        return f"Research brief from {display}:\n" + _wrap_external(brief, f"research_web:{question[:120]}")

    return _handle_research
