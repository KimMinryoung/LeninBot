"""Provider-aware system-prompt rendering.

System prompts are authored as a semantic intermediate representation
(``SystemPrompt``) and compiled to the target format at dispatch time.
Claude receives XML tags (Anthropic's native structure); OpenAI/Qwen
receive Markdown headers (the structure those families are trained on).

Rationale: the harness already uses thin-adapter dispatch on the *API*
layer (Claude SDK vs OpenAI-compatible); this module applies the same
split to the *prompt* layer so one semantic source of truth can serve
every provider without the lossy auto-conversion hazard of regex-based
tag rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Intermediate representation ──────────────────────────────────────

@dataclass
class SystemPrompt:
    """Semantic structure of a system prompt.

    Fields:
      identity   — persona / ground-truth prose, emitted verbatim.
      preamble   — short framing prose between identity and sections.
      sections   — ordered list of (kebab-name, body) tuples. Rendered
                   as ``<tag>body</tag>`` for Claude, ``## Title``
                   headers for Markdown providers.
      context    — ordered list of (key, value) for the context footer.
                   Values may contain ``{placeholder}`` slots to be
                   resolved later via ``str.format()``.
    """

    identity: str = ""
    preamble: str = ""
    sections: list[tuple[str, str]] = field(default_factory=list)
    context: list[tuple[str, str]] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────

def _title(kebab: str) -> str:
    """kebab-case-or-snake_case → Title Case."""
    normalized = kebab.replace("_", "-")
    return " ".join(part.capitalize() for part in normalized.split("-") if part)


# ── Renderers ────────────────────────────────────────────────────────

class ClaudeRenderer:
    """XML-tag rendering — Anthropic's recommended structure."""

    def render(self, p: SystemPrompt) -> str:
        parts: list[str] = []
        if p.identity:
            parts.append(p.identity.rstrip())
        if p.preamble:
            parts.append(p.preamble.rstrip())
        for name, body in p.sections:
            parts.append(f"<{name}>\n{body.strip()}\n</{name}>")
        if p.context:
            inner = [f"<{k}>{v}</{k}>" for k, v in p.context if v]
            if inner:
                parts.append("<context>\n" + "\n".join(inner) + "\n</context>")
        return "\n\n".join(parts)


class MarkdownRenderer:
    """Markdown-header rendering — GPT and Qwen parse ``## Header`` as
    top-level structure in their instruction-tuning data."""

    def render(self, p: SystemPrompt) -> str:
        parts: list[str] = []
        if p.identity:
            parts.append(p.identity.rstrip())
        if p.preamble:
            parts.append(p.preamble.rstrip())
        for name, body in p.sections:
            parts.append(f"## {_title(name)}\n{body.strip()}")
        if p.context:
            ctx_lines = [
                f"- **{_title(k)}**: {v}" for k, v in p.context if v
            ]
            if ctx_lines:
                parts.append("## Context\n" + "\n".join(ctx_lines))
        return "\n\n".join(parts)


# ── Dispatch ─────────────────────────────────────────────────────────

_RENDERERS: dict[str, object] = {
    "claude": ClaudeRenderer(),
    "openai": MarkdownRenderer(),
    "local": MarkdownRenderer(),  # Qwen — same Markdown style as GPT
}


def render(prompt: SystemPrompt, provider: str) -> str:
    """Render ``prompt`` in the structure native to ``provider``.

    Unknown providers fall back to Markdown (the safer common denominator).
    """
    renderer = _RENDERERS.get(provider, MarkdownRenderer())
    return renderer.render(prompt)  # type: ignore[union-attr]


# ── Provider-aware dynamic wrappers ──────────────────────────────────
#
# These helpers format the runtime-injected dynamic blocks (current-model,
# system-alerts) in the structure expected by each provider. They exist so
# that the dynamic tail of the system prompt stays coherent with the static
# structure chosen by the renderer.

def format_current_model(
    provider: str, *, model_id: str, attrs: dict[str, str]
) -> str:
    """Format the current-model context block for the given provider."""
    if provider == "claude":
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        return f"<current-model {attr_str}>{model_id}</current-model>"
    # Markdown: bullet with inline metadata
    meta = ", ".join(f"{k}={v}" for k, v in attrs.items())
    return f"- **Current Model**: {model_id} ({meta})"


def format_system_alerts(provider: str, *, alerts: list[str]) -> str:
    """Format the system-alerts block. Empty list → empty string so the
    caller can omit the slot cleanly."""
    if not alerts:
        return ""
    if provider == "claude":
        items = "\n".join(f"- {a}" for a in alerts)
        return f"\n<system-alerts>\n{items}\n</system-alerts>"
    items = "\n".join(f"- {a}" for a in alerts)
    return f"\n### System Alerts\n{items}"
