"""prompt_context.py - Provider-native context block rendering helpers."""

from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone


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


def format_agent_board(messages: list[dict], provider: str | None) -> str:
    if not messages:
        return ""

    kst = timezone(timedelta(hours=9))
    lines = []
    for message in messages:
        ts = message.get("ts", 0)
        time_str = datetime.fromtimestamp(ts, tz=kst).strftime("%H:%M") if ts else "?"
        lines.append(
            f"  [{time_str}] [{message.get('agent', '?')} #{message.get('task_id', '?')}] "
            f"{message.get('message', '')}"
        )

    description = "Messages left by other agents participating in the same mission."
    if not uses_xml(provider):
        return (
            "### Agent Board\n"
            f"{description}\n"
            + "\n".join(f"-{line[1:]}" if line.startswith("  ") else f"- {line}" for line in lines)
        )
    return (
        "<agent-board>\n"
        f"Below are {description[0].lower()}{description[1:]}\n"
        + "\n".join(lines)
        + "\n</agent-board>"
    )


def format_task_chain(chain: list[dict], provider: str | None) -> str:
    if not chain:
        return ""

    if not uses_xml(provider):
        parts = [
            f"### Task Chain (depth {len(chain)})",
            "Parent task chain. Understand prior work and avoid duplicate work.",
        ]
        for entry in chain:
            tid = entry.get("task_id", "?")
            agent = entry.get("agent_type", "?")
            content = entry.get("content", "")
            result = entry.get("result", "")
            tool_log = entry.get("tool_log", "")
            parts.append("")
            parts.append(f"#### Ancestor #{tid} [{agent}]")
            parts.append(f"**Task content:** {content}")
            if result:
                parts.append(f"**Result:** {result}")
            if tool_log:
                parts.append("**Tool log:**")
                parts.append(fenced_text(tool_log))
        return "\n".join(parts)

    parts = []
    for entry in chain:
        tid = entry.get("task_id", "?")
        agent = entry.get("agent_type", "?")
        content = entry.get("content", "")
        result = entry.get("result", "")
        tool_log = entry.get("tool_log", "")
        block = f"  <ancestor task_id=\"{tid}\" agent=\"{agent}\">\n"
        block += f"    <task-content>{content}</task-content>\n"
        if result:
            block += f"    <result>{result}</result>\n"
        if tool_log:
            block += f"    <tool-log>{tool_log}</tool-log>\n"
        block += f"  </ancestor>"
        parts.append(block)
    return (
        f"<task-chain depth=\"{len(chain)}\">\n"
        "Below is the parent chain of the current task. Understand what prior tasks did and avoid duplicate work.\n"
        + "\n".join(parts)
        + "\n</task-chain>"
    )
