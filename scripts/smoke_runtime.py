#!/usr/bin/env python3
"""Smoke checks for runtime profile and provider-native prompt formatting."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_context import (
    format_agent_execution_history,
    format_agent_board,
    format_mission_context,
    format_subtask_results,
    format_task_chain,
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

    board = [{"ts": 0, "agent": "scout", "task_id": 12, "message": "lead found"}]
    assert format_agent_board(board, "claude") == (
        "<agent-board>\n"
        "Below are messages left by other agents participating in the same mission.\n"
        "  [?] [scout #12] lead found\n"
        "</agent-board>"
    )
    assert format_agent_board(board, "openai") == (
        "### Agent Board\n"
        "Messages left by other agents participating in the same mission.\n"
        "- [?] [scout #12] lead found"
    )

    chain = [{
        "task_id": "3",
        "agent_type": "analyst",
        "content": "Analyze",
        "result": "Done",
        "tool_log": "search",
    }]
    assert format_task_chain(chain, "claude") == (
        '<task-chain depth="1">\n'
        "Below is the parent chain of the current task. Understand what prior tasks did and avoid duplicate work.\n"
        '  <ancestor task_id="3" agent="analyst">\n'
        "    <task-content>Analyze</task-content>\n"
        "    <result>Done</result>\n"
        "    <tool-log>search</tool-log>\n"
        "  </ancestor>\n"
        "</task-chain>"
    )
    assert format_task_chain(chain, "deepseek") == (
        "### Task Chain (depth 1)\n"
        "Parent task chain. Understand prior work and avoid duplicate work.\n\n"
        "#### Ancestor #3 [analyst]\n"
        "**Task content:** Analyze\n"
        "**Result:** Done\n"
        "**Tool log:**\n"
        "```text\nsearch\n```"
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


def _assert_agent_runtime_config() -> None:
    from agents import get_agent

    diary = get_agent("diary")
    assert diary.provider == "deepseek"
    assert diary.model == "deepseek_pro"
    assert diary.terminal_tools == ["save_diary"]
    assert diary.skip_orchestrator_report is True
    assert diary.max_rounds == 30

    programmer = get_agent("programmer")
    assert programmer.provider == "codex"
    assert programmer.budget_usd == 0.01

    scout = get_agent("scout")
    assert scout.provider == "deepseek"
    assert scout.model == "deepseek_flash"
    assert scout.max_rounds == 30

    browser = get_agent("browser")
    assert browser.budget_usd == 1.5

    autonomous = get_agent("autonomous_project")
    assert autonomous.budget_usd == 0.60
    assert "add_research_note" in autonomous.finalization_tools
    assert "publish_static_page" in autonomous.finalization_tools


def _assert_agent_runtime_dynamic_reload() -> None:
    from agents.base import AgentSpec
    import agents.runtime_config as runtime_config

    original_path = runtime_config._CONFIG_PATH
    original_mtime = runtime_config._last_mtime_ns
    original_base = dict(runtime_config._base_runtime)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "agent_runtime.json"
        registry = {
            "dummy": AgentSpec(
                name="dummy",
                description="dummy agent",
                system_prompt_template="dummy",
            )
        }
        try:
            runtime_config._CONFIG_PATH = path
            runtime_config._last_mtime_ns = None
            runtime_config._base_runtime = {}

            path.write_text(
                json.dumps({
                    "dummy": {
                        "provider": "claude",
                        "model": "sonnet",
                        "budget_usd": 2.0,
                        "max_rounds": 12,
                    }
                }),
                encoding="utf-8",
            )
            runtime_config.apply_agent_runtime_config(registry)
            assert registry["dummy"].provider == "claude"
            assert registry["dummy"].model == "sonnet"
            assert registry["dummy"].budget_usd == 2.0
            assert registry["dummy"].max_rounds == 12

            path.write_text(
                json.dumps({
                    "dummy": {
                        "provider": "deepseek",
                        "model": "deepseek_flash",
                        "budget_usd": 3.0,
                        "max_rounds": 8,
                    }
                }),
                encoding="utf-8",
            )
            runtime_config._last_mtime_ns = -1
            assert runtime_config.reload_agent_runtime_config_if_changed(registry)
            assert registry["dummy"].provider == "deepseek"
            assert registry["dummy"].model == "deepseek_flash"
            assert registry["dummy"].budget_usd == 3.0
            assert registry["dummy"].max_rounds == 8

            path.write_text(json.dumps({"dummy": {}}), encoding="utf-8")
            runtime_config._last_mtime_ns = -1
            assert runtime_config.reload_agent_runtime_config_if_changed(registry)
            assert registry["dummy"].provider is None
            assert registry["dummy"].model is None
            assert registry["dummy"].budget_usd == 1.0
            assert registry["dummy"].max_rounds == 50

            registry["dummy"].tools = ["allowed_tool"]
            path.write_text(
                json.dumps({"dummy": {"terminal_tools": ["missing_tool"]}}),
                encoding="utf-8",
            )
            runtime_config._last_mtime_ns = -1
            assert not runtime_config.reload_agent_runtime_config_if_changed(registry)
            assert registry["dummy"].provider is None
            assert registry["dummy"].terminal_tools == []
        finally:
            runtime_config._CONFIG_PATH = original_path
            runtime_config._last_mtime_ns = original_mtime
            runtime_config._base_runtime = original_base


def _assert_web_political_line_dynamic_reload() -> None:
    import agents.base as agent_base
    from web_chat import _build_web_system_prompt

    original_path = agent_base._POLITICAL_LINE_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "political_line.md"
        try:
            agent_base._POLITICAL_LINE_PATH = path

            path.write_text("first web political line", encoding="utf-8")
            first_xml = _build_web_system_prompt("claude")
            first_markdown = _build_web_system_prompt("openai")
            assert "first web political line" in first_xml
            assert "first web political line" in first_markdown

            path.write_text("second web political line", encoding="utf-8")
            second_xml = _build_web_system_prompt("claude")
            second_markdown = _build_web_system_prompt("openai")
            assert "second web political line" in second_xml
            assert "second web political line" in second_markdown
            assert "first web political line" not in second_xml
            assert "first web political line" not in second_markdown
        finally:
            agent_base._POLITICAL_LINE_PATH = original_path


def _assert_diary_web_context_injection() -> None:
    import telegram.tasks as tasks

    original_formatter = tasks._format_diary_web_chat_context
    try:
        tasks._format_diary_web_chat_context = (
            lambda provider: "### Diary Web Chat Preflight\n\nWeb user: 일기에서 이 수정 지시를 반영해"
        )
        content = tasks._build_task_context_content(
            {"id": 900, "user_id": 0, "agent_type": "diary", "content": "write diary"},
            "write diary",
            context_provider="deepseek",
        )
        assert "### Diary Web Chat Preflight" in content
        assert "Web user: 일기에서 이 수정 지시를 반영해" in content
        assert content.index("### Diary Web Chat Preflight") < content.index("### Task")

        non_diary = tasks._build_task_context_content(
            {"id": 901, "user_id": 0, "agent_type": "analyst", "content": "analyze"},
            "analyze",
            context_provider="deepseek",
        )
        assert "Diary Web Chat Preflight" not in non_diary
    finally:
        tasks._format_diary_web_chat_context = original_formatter


async def _assert_diary_runtime_route_fallbacks() -> None:
    import self_runtime.tools as tools

    original_classifier = tools._classify_route_with_llm
    try:
        async def unavailable_classifier(_task: str, _candidates=None):
            return None

        tools._classify_route_with_llm = unavailable_classifier
        content_edit = json.loads(await tools._exec_route_task("일기 123번 오타 수정해줘", include_store_guide=False))
        assert content_edit["recommendation"]["recommended_agent"] == "diary"

        pipeline_fix = json.loads(await tools._exec_route_task("프로그래머한테 일기 쓰기 실패 오류 고치라고 시켜", include_store_guide=False))
        assert pipeline_fix["recommendation"]["recommended_agent"] == "programmer"

        runtime_fix = json.loads(await tools._exec_route_task("diary agent 파이프라인 결함 수정", include_store_guide=False))
        assert runtime_fix["recommendation"]["recommended_agent"] == "programmer"
    finally:
        tools._classify_route_with_llm = original_classifier


async def _assert_guarded_diary_save_handler_accepts_tool_payloads() -> None:
    import telegram.bot as bot
    import telegram.diary_publication as publication
    from tool_loop_common import execute_tool

    original_review = bot._run_stasova_diary_review
    original_apply = bot._apply_stasova_diary_review
    original_publish = publication.publish_reviewed_diary_entry
    seen: list[tuple[str, str]] = []

    async def fake_review(_task, title, content):
        seen.append((title, content))
        return "위험도 총평: 낮음\n경고 항목: 없음"

    async def fake_apply(_task, *, title, content, review_report):
        assert "위험도 총평" in review_report
        return title, content

    async def fake_publish(**kwargs):
        assert kwargs["final_title"] == kwargs["title"]
        assert kwargs["final_content"] == kwargs["content"]
        return 321, " / Telegram channel: skipped", None

    try:
        bot._run_stasova_diary_review = fake_review
        bot._apply_stasova_diary_review = fake_apply
        publication.publish_reviewed_diary_entry = fake_publish

        handler = bot._make_guarded_diary_save_handler(None, {"id": 900})
        result, is_error = await execute_tool(
            "save_diary",
            {"title": "제목", "content": "본문"},
            {"save_diary": handler},
            log_event=None,
        )
        assert not is_error
        assert "Diary reviewed by Stasova" in result
        assert seen[-1] == ("제목", "본문")

        positional_result = await handler({"title": "딕셔너리 제목", "content": "딕셔너리 본문"})
        assert "Diary reviewed by Stasova" in positional_result
        assert seen[-1] == ("딕셔너리 제목", "딕셔너리 본문")

        missing_title = await handler(content="본문")
        assert "missing required argument: title" in missing_title
    finally:
        bot._run_stasova_diary_review = original_review
        bot._apply_stasova_diary_review = original_apply
        publication.publish_reviewed_diary_entry = original_publish


async def main() -> None:
    _assert_prompt_context()
    await _assert_runtime_profiles()
    _assert_agent_runtime_config()
    _assert_agent_runtime_dynamic_reload()
    _assert_web_political_line_dynamic_reload()
    _assert_diary_web_context_injection()
    await _assert_diary_runtime_route_fallbacks()
    await _assert_guarded_diary_save_handler_accepts_tool_payloads()
    print("runtime smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
