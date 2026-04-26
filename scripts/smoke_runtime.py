#!/usr/bin/env python3
"""Smoke checks for runtime profile and provider-native prompt formatting."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_context import (
    format_agent_execution_history,
    format_mission_context,
    format_subtask_results,
    prompt_format_for_provider,
    uses_xml,
    wrap_context_block,
    wrap_task_content,
    xml_attrs,
)
from runtime_profile import resolve_runtime_profile


def _assert_prompt_context() -> None:
    assert prompt_format_for_provider("claude") == "xml"
    assert prompt_format_for_provider("openai") == "markdown"
    assert prompt_format_for_provider("deepseek") == "markdown"
    assert uses_xml("claude")
    assert not uses_xml("openai")

    assert xml_attrs({"id": 7, "title": 'A "quoted" task', "skip": None}) == (
        ' id="7" title="A &quot;quoted&quot; task"'
    )

    assert wrap_task_content("Do the work", "claude") == "<task>\nDo the work\n</task>"
    assert wrap_task_content("Do the work", "openai") == "### Task\n\nDo the work"
    assert wrap_context_block(
        "mission-context",
        "body",
        "deepseek",
        heading="Mission Context",
        attrs={"id": 1},
    ) == "### Mission Context\n\nbody"

    events = [
        {"created_at": "2026-04-26", "source": "task#1", "event_type": "task_created", "content": "started"},
    ]
    assert format_mission_context(3, "Mission", events, "claude") == (
        '<mission-context id="3" title="Mission">\n'
        "  [2026-04-26] (task#1) task_created: started\n"
        "</mission-context>"
    )
    assert format_mission_context(3, "Mission", events, "openai") == (
        "### Mission Context (#3: Mission)\n"
        "- [2026-04-26] (task#1) task_created: started"
    )

    subtasks = [{"id": 11, "agent_type": "scout", "content": "Find facts", "result": "Facts"}]
    assert format_subtask_results(subtasks, "claude") == (
        "<subtask-results>\n"
        '  <subtask id="11" agent="scout">\n'
        "    <task-brief>Find facts</task-brief>\n"
        "    <result>\nFacts\n    </result>\n"
        "  </subtask>\n"
        "</subtask-results>"
    )
    assert format_subtask_results(subtasks, "deepseek") == (
        "### Subtask Results\n\n"
        "#### Subtask #11 [scout]\n"
        "**Task brief:** Find facts\n\n"
        "**Result:**\n\nFacts"
    )

    assert format_agent_execution_history(
        agent_type="analyst",
        previous_task_id=5,
        completed_at="2026-04-26 01:02",
        summary="summary",
        tool_log="tool log",
        provider="openai",
    ) == (
        "### Agent Execution History (analyst)\n"
        "Previous task: #5 completed 2026-04-26 01:02\n\n"
        "**Summary:** summary\n\n"
        "**Tool log:**\n"
        "```text\ntool log\n```"
    )


async def _assert_runtime_profiles() -> None:
    deepseek_chat = await resolve_runtime_profile("chat", provider_override="deepseek")
    assert deepseek_chat.provider == "deepseek"
    assert deepseek_chat.prompt_format == "markdown"
    assert deepseek_chat.model_id == "deepseek-v4-pro"

    openai_task = await resolve_runtime_profile("task", provider_override="openai")
    assert openai_task.provider == "openai"
    assert openai_task.prompt_format == "markdown"
    assert openai_task.model_id == "gpt-5.5"

    autonomous = await resolve_runtime_profile(
        "autonomous",
        provider_override="openai",
        max_rounds_override=6,
        max_tokens_override=16384,
        budget_override=0.60,
    )
    assert autonomous.provider == "openai"
    assert autonomous.model_id == "gpt-5.5"
    assert autonomous.max_rounds == 6
    assert autonomous.max_tokens == 16384
    assert autonomous.budget_usd == 0.60

    claude_webchat = await resolve_runtime_profile(
        "webchat",
        provider_override="claude",
        model_override="claude-test-model",
        tier_override="medium",
    )
    assert claude_webchat.provider == "claude"
    assert claude_webchat.prompt_format == "xml"
    assert claude_webchat.tier == "medium"
    assert claude_webchat.model_id == "claude-test-model"


async def main() -> None:
    _assert_prompt_context()
    await _assert_runtime_profiles()
    print("runtime smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
