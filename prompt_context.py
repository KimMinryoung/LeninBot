"""prompt_context.py - Provider-native context block rendering helpers."""

from __future__ import annotations

import html


def prompt_format_for_provider(provider: str | None) -> str:
    return "xml" if (provider or "claude") == "claude" else "markdown"


def uses_xml(provider: str | None) -> bool:
    return prompt_format_for_provider(provider) == "xml"


def xml_attrs(attrs: dict | None = None) -> str:
    if not attrs:
        return ""
    parts = []
    for key, value in attrs.items():
        if value is None:
            continue
        safe_key = str(key).replace(" ", "-")
        safe_val = html.escape(str(value), quote=True)
        parts.append(f'{safe_key}="{safe_val}"')
    return " " + " ".join(parts) if parts else ""


def wrap_context_block(
    name: str,
    body: str,
    provider: str | None,
    *,
    heading: str | None = None,
    attrs: dict | None = None,
) -> str:
    """Wrap a context body as XML for Claude, Markdown for other providers."""
    body = (body or "").strip()
    if uses_xml(provider):
        return f"<{name}{xml_attrs(attrs)}>\n{body}\n</{name}>"
    return f"### {heading or name.replace('-', ' ').title()}\n\n{body}"


def wrap_task_content(content: str, provider: str | None) -> str:
    return wrap_context_block("task", content, provider, heading="Task")


def fenced_text(text: str) -> str:
    return f"```text\n{text}\n```"
