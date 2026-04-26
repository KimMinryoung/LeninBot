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


def format_mission_context(
    mission_id: int,
    mission_title: str,
    events: list[dict],
    provider: str | None,
) -> str:
    if not events:
        return ""
    if uses_xml(provider):
        lines = [f"<mission-context{xml_attrs({'id': mission_id, 'title': mission_title})}>"]
        for event in events:
            lines.append(
                f"  [{event['created_at']}] ({event['source']}) "
                f"{event['event_type']}: {str(event.get('content') or '')[:500]}"
            )
        lines.append("</mission-context>")
    else:
        lines = [f"### Mission Context (#{mission_id}: {mission_title})"]
        for event in events:
            lines.append(
                f"- [{event['created_at']}] ({event['source']}) "
                f"{event['event_type']}: {str(event.get('content') or '')[:500]}"
            )
    return "\n".join(lines)


def format_subtask_results(sibling_results: list[dict], provider: str | None) -> str:
    if not sibling_results:
        return ""

    result_blocks = []
    for row in sibling_results:
        agent = row.get("agent_type") or "unknown"
        result = str(row.get("result") or "")[:5000]
        task_brief = str(row.get("content") or "")[:300]
        if uses_xml(provider):
            result_blocks.append(
                f"  <subtask id=\"{row['id']}\" agent=\"{agent}\">\n"
                f"    <task-brief>{task_brief}</task-brief>\n"
                f"    <result>\n{result}\n    </result>\n"
                f"  </subtask>"
            )
        else:
            result_blocks.append(
                f"#### Subtask #{row['id']} [{agent}]\n"
                f"**Task brief:** {task_brief}\n\n"
                f"**Result:**\n\n{result}"
            )

    if uses_xml(provider):
        return "<subtask-results>\n" + "\n".join(result_blocks) + "\n</subtask-results>"
    return "### Subtask Results\n\n" + "\n\n".join(result_blocks)


def format_agent_execution_history(
    *,
    agent_type: str,
    previous_task_id: int,
    completed_at: str,
    summary: str,
    tool_log: str = "",
    provider: str | None,
) -> str:
    if uses_xml(provider):
        block = f"  <prev-task id=\"{previous_task_id}\" completed=\"{completed_at}\">\n"
        block += f"    <summary>{summary}</summary>\n"
        if tool_log:
            block += f"    <tool-log>\n{tool_log}\n    </tool-log>\n"
        block += f"  </prev-task>"
        return (
            f"<agent-execution-history agent=\"{agent_type}\">\n"
            + block
            + "\n</agent-execution-history>"
        )

    lines = [
        f"### Agent Execution History ({agent_type})",
        f"Previous task: #{previous_task_id} completed {completed_at}",
        "",
        f"**Summary:** {summary}",
    ]
    if tool_log:
        lines.extend(["", "**Tool log:**", fenced_text(tool_log)])
    return "\n".join(lines)
