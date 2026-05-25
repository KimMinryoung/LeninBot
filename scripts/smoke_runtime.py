#!/usr/bin/env python3
"""Smoke checks for runtime profile and provider-native prompt formatting."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
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
    import os

    import bot_config

    deepseek_chat = await resolve_runtime_profile("chat", provider_override="deepseek")
    deepseek_expected = bot_config.get_current_model_selection("chat", provider_override="deepseek")
    assert deepseek_chat.provider == "deepseek"
    assert deepseek_chat.prompt_format == "markdown"
    assert deepseek_chat.tier == str(bot_config._config.get("chat_model", "high"))
    assert deepseek_chat.alias == deepseek_expected["alias"]
    assert deepseek_chat.model_id == deepseek_expected["model_id"]
    assert deepseek_chat.max_rounds == int(bot_config._config.get("max_rounds_chat", 50))
    assert deepseek_chat.max_tokens == bot_config._CLAUDE_MAX_TOKENS
    assert deepseek_chat.budget_usd == float(bot_config._config.get("chat_budget", 0.30))

    openai_task = await resolve_runtime_profile("task", provider_override="openai")
    openai_task_expected = bot_config.get_current_model_selection("task", provider_override="openai")
    assert openai_task.provider == "openai"
    assert openai_task.prompt_format == "markdown"
    assert openai_task.tier == str(bot_config._config.get("task_model", "high"))
    assert openai_task.alias == openai_task_expected["alias"]
    assert openai_task.model_id == openai_task_expected["model_id"]
    assert openai_task.max_rounds == int(bot_config._config.get("max_rounds_task", 50))
    assert openai_task.max_tokens == bot_config._CLAUDE_MAX_TOKENS_TASK
    assert openai_task.budget_usd == float(bot_config._config.get("task_budget", 1.00))

    autonomous = await resolve_runtime_profile(
        "autonomous",
        provider_override="openai",
        max_rounds_override=6,
        max_tokens_override=16384,
        budget_override=0.60,
    )
    autonomous_expected = bot_config.get_current_model_selection("autonomous", provider_override="openai")
    assert autonomous.provider == "openai"
    assert autonomous.prompt_format == "markdown"
    assert autonomous.tier == str(bot_config._config.get("autonomous_model", "high"))
    assert autonomous.alias == autonomous_expected["alias"]
    assert autonomous.model_id == autonomous_expected["model_id"]
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
    assert claude_webchat.max_rounds == 20
    try:
        expected_webchat_tokens = int(os.getenv("WEBCHAT_MAX_TOKENS") or bot_config._WEBCHAT_MAX_TOKENS)
    except ValueError:
        expected_webchat_tokens = bot_config._WEBCHAT_MAX_TOKENS
    if expected_webchat_tokens <= 0:
        expected_webchat_tokens = bot_config._WEBCHAT_MAX_TOKENS
    assert claude_webchat.max_tokens == expected_webchat_tokens
    assert claude_webchat.budget_usd == float(
        bot_config._config.get("webchat_budget", bot_config._config.get("chat_budget", 0.30))
    )


def _assert_agent_runtime_config() -> None:
    import agents.runtime_config as runtime_config
    from agents import list_agents

    config = runtime_config._load_runtime_config()
    specs = {spec.name: spec for spec in list_agents()}
    base_runtime = runtime_config._base_runtime

    assert set(config) <= set(specs)
    assert set(specs) <= set(base_runtime)

    for name, spec in specs.items():
        cfg = config.get(name, {})
        base = base_runtime[name]
        expected = {
            "provider": cfg.get("provider", base["provider"]),
            "model": cfg.get("model", base["model"]),
            "budget_usd": float(cfg.get("budget_usd", base["budget_usd"])),
            "max_rounds": int(cfg.get("max_rounds", base["max_rounds"])),
            "finalization_tools": list(cfg.get("finalization_tools", base["finalization_tools"])),
            "terminal_tools": list(cfg.get("terminal_tools", base["terminal_tools"])),
            "skip_orchestrator_report": bool(
                cfg.get("skip_orchestrator_report", base["skip_orchestrator_report"])
            ),
        }

        assert spec.provider == expected["provider"], name
        assert spec.model == expected["model"], name
        assert spec.budget_usd == expected["budget_usd"], name
        assert spec.max_rounds == expected["max_rounds"], name
        assert spec.finalization_tools == expected["finalization_tools"], name
        assert spec.terminal_tools == expected["terminal_tools"], name
        assert spec.skip_orchestrator_report is expected["skip_orchestrator_report"], name


def _assert_autonomous_base_finalization_tools() -> None:
    import agents.runtime_config as runtime_config

    base = runtime_config._base_runtime["autonomous_project"]
    required = {
        "add_research_note",
        "revise_plan",
        "set_project_state",
        "research_document",
        "publish_hub_curation",
        "edit_content",
        "publish_static_page",
    }
    assert required <= set(base["finalization_tools"])


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


def _assert_autonomous_prompt_surfaces_staged_drafts() -> None:
    import autonomous_project as ap

    original_notes = ap._recent_notes
    original_last_log = ap._fetch_last_tick_tool_log
    original_staged = ap._recent_staged_research_drafts
    original_attention = ap._recent_tick_attention_events
    try:
        ap._recent_notes = lambda _project: []
        ap._fetch_last_tick_tool_log = lambda _project_id: None
        ap._recent_tick_attention_events = lambda _project_id: [
            {
                "event_type": "tick_no_durable_action",
                "content": "tick completed without durable action",
                "created_at": None,
            },
            {
                "event_type": "advisories_retained_no_durable_action",
                "content": "1 advisories retained because tick saved no durable project action",
                "created_at": None,
            },
        ]
        ap._recent_staged_research_drafts = lambda _project_id: [{
            "filename": "draft-report.md",
            "slug": "draft-report",
            "title": "Draft Report",
            "summary": "Pending fact-check before publication.",
            "updated_at": "2026-05-25 11:00:00+00",
            "project_match": True,
        }]
        project = {
            "id": 7,
            "title": "Autonomous Test",
            "topic": "staged drafts",
            "goal": "Continue publication work.",
            "state": "researching",
            "plan": {"goals": ["publish checked draft"], "steps": ["fact-check"]},
            "turn_count": 2,
            "last_run_at": None,
        }

        markdown_prompt = ap._build_task_prompt(project, turn_budget=3, provider="deepseek")
        assert "### Staged Research Drafts" in markdown_prompt
        assert "[this-project] slug=draft-report" in markdown_prompt
        assert "status='staged'" in markdown_prompt
        assert "### Recent Tick Warnings" in markdown_prompt
        assert "tick_no_durable_action" in markdown_prompt
        assert "advisories_retained_no_durable_action" in markdown_prompt

        xml_prompt = ap._build_task_prompt(project, turn_budget=3, provider="claude")
        assert "<staged-research-drafts>" in xml_prompt
        assert "slug=draft-report" in xml_prompt
        assert "read_self(content_type='research_document'" in xml_prompt
        assert "<recent-tick-warnings>" in xml_prompt
        assert "tick completed without durable action" in xml_prompt
        assert "advisories retained" in xml_prompt
    finally:
        ap._recent_notes = original_notes
        ap._fetch_last_tick_tool_log = original_last_log
        ap._recent_staged_research_drafts = original_staged
        ap._recent_tick_attention_events = original_attention


def _assert_staged_drafts_prioritize_project_events() -> None:
    import autonomous_project as ap

    original_query = ap.db_query
    calls: list[str] = []

    def fake_query(sql, params=()):
        calls.append(sql)
        if "FROM autonomous_project_events" in sql:
            assert params == (7, 8)
            assert "ROW_NUMBER() OVER" in sql
            assert "ORDER BY staged_at DESC" in sql
            return [{
                "filename": "project-draft.md",
                "slug": "project-draft",
                "title": "Project Draft",
                "summary": "Project-specific staged draft.",
                "updated_at": None,
                "project_match": True,
            }]
        if "FROM research_documents" in sql:
            return [
                {
                    "filename": "project-draft.md",
                    "slug": "project-draft",
                    "title": "Duplicate Global",
                    "summary": "Should be skipped.",
                    "updated_at": None,
                    "project_match": False,
                },
                {
                    "filename": "other-draft.md",
                    "slug": "other-draft",
                    "title": "Other Draft",
                    "summary": "Fallback staged draft.",
                    "updated_at": None,
                    "project_match": False,
                },
            ]
        raise AssertionError(f"unexpected query: {sql}")

    try:
        ap.db_query = fake_query
        rows = ap._recent_staged_research_drafts(7)
        assert [row["slug"] for row in rows] == ["project-draft", "other-draft"]
        rendered = ap._format_staged_research_drafts(rows)
        assert "[this-project] slug=project-draft" in rendered
        assert "[other-staged] slug=other-draft" in rendered
        assert len(calls) == 2
    finally:
        ap.db_query = original_query


async def _assert_read_self_autonomous_project_uses_note_table() -> None:
    import db
    import self_runtime.tools as tools

    original_query = db.query
    original_query_one = db.query_one

    def fake_query_one(sql, params=()):
        if "FROM autonomous_projects WHERE id" in sql:
            return {
                "id": 7,
                "title": "Autonomous Test",
                "topic": "note table",
                "goal": "Expose durable notes.",
                "state": "researching",
                "plan": {"goals": ["observe"], "steps": ["read notes"]},
                "research_notes": [{"turn": 0, "text": "legacy note", "sources": []}],
                "turn_count": 3,
                "last_run_at": None,
                "created_at": None,
            }
        if "COUNT(*) AS count FROM autonomous_project_notes" in sql:
            return {"count": 2}
        raise AssertionError(f"unexpected query_one: {sql}")

    def fake_query(sql, params=()):
        if "FROM autonomous_project_notes" in sql:
            return [
                {"turn": 2, "text": "newer durable note", "sources": ["https://source/2"], "created_at": None},
                {"turn": 1, "text": "older durable note", "sources": ["https://source/1"], "created_at": None},
            ]
        if "event_type = 'tick_tool_log'" in sql:
            return [{
                "content": "  [1] web_search({\"query\": \"already checked\"}) -> result",
                "meta": {"turn": 3, "rounds_used": 2, "tool_calls": 1, "cost_usd": 0.0123},
                "created_at": None,
            }]
        if "event_type = 'tick_error'" in sql:
            return [{"content": "RuntimeError: failed autonomous fetch", "created_at": None}]
        if "event_type = 'tick_no_durable_action'" in sql:
            return [{"content": "tick completed without durable action", "created_at": None}]
        if "FROM autonomous_project_events" in sql:
            assert "ORDER BY created_at DESC, id DESC" in sql
            return []
        if "FROM autonomous_project_advisories" in sql:
            return []
        raise AssertionError(f"unexpected query: {sql}")

    try:
        db.query = fake_query
        db.query_one = fake_query_one
        output = await tools._exec_read_autonomous_project(project_id=7, limit=2)
        assert "source=autonomous_project_notes" in output
        assert "2 shown / 2 total" in output
        assert "older durable note" in output
        assert "newer durable note" in output
        assert "-- last tick error (?) --" in output
        assert "failed autonomous fetch" in output
        assert "-- last tick no durable action (?) --" in output
        assert "without durable action" in output
        assert "-- last tick tool log (turn=3, rounds=2, tools=1, cost=$0.012) --" in output
        assert "already checked" in output
        assert "legacy note" not in output
    finally:
        db.query = original_query
        db.query_one = original_query_one


def _assert_autonomous_project_selection_prioritizes_advice() -> None:
    import autonomous_project as ap

    original_query_one = ap.db_query_one
    captured: dict[str, object] = {}

    def fake_query_one(sql, params=()):
        captured["sql"] = " ".join(sql.split())
        captured["params"] = params
        return {"id": 7}

    try:
        ap.db_query_one = fake_query_one
        row = ap._pick_next_project()
        assert row == {"id": 7}
        sql = str(captured["sql"])
        assert "autonomous_project_advisories" in sql
        assert "consumed_at IS NULL" in sql
        assert "pending_advisories" in sql
        assert "ORDER BY COALESCE(a.pending_advisories, 0) DESC" in sql
        assert "last_run_at ASC NULLS FIRST" in sql
        assert list(captured["params"][0]) == list(ap.ACTIVE_STATES)
    finally:
        ap.db_query_one = original_query_one


async def _assert_autonomous_tick_failure_updates_cooldown() -> None:
    import autonomous_project as ap

    original_get_agent = __import__("agents").get_agent
    original_chat = __import__("telegram.bot").bot._chat_with_tools
    original_execute = ap.db_execute
    original_log_event = ap._log_event
    original_notify = ap._notify_telegram
    original_fetch_advisories = ap._fetch_pending_advisories
    original_recent_notes = ap._recent_notes
    original_recent_staged = ap._recent_staged_research_drafts
    original_last_log = ap._fetch_last_tick_tool_log
    original_logger_disabled = ap.logger.disabled

    updates: list[tuple[str, tuple]] = []
    events: list[tuple[str, str]] = []

    class DummySpec:
        max_rounds = 2
        model = None
        budget_usd = 0.1
        finalization_tools = []
        terminal_tools = []

        def effective_provider(self, _configured_provider):
            return "openai"

        def filter_tools(self, _tools, _handlers):
            return [], {}

        def render_prompt(self, *, provider="claude", **_kwargs):
            return "system"

    async def failing_chat(*_args, **_kwargs):
        raise RuntimeError("simulated autonomous failure")

    def fake_execute(sql, params=()):
        updates.append((" ".join(sql.split()), tuple(params)))

    def fake_log_event(_project_id, event_type, content="", meta=None, **_kwargs):
        events.append((event_type, content))

    async def fake_notify(*_args, **_kwargs):
        raise AssertionError("notify should not run on failed tick")

    import agents
    import telegram.bot as bot

    try:
        agents.get_agent = lambda _name: DummySpec()
        bot._chat_with_tools = failing_chat
        ap.db_execute = fake_execute
        ap._log_event = fake_log_event
        ap._notify_telegram = fake_notify
        ap._fetch_pending_advisories = lambda _project_id: []
        ap._recent_notes = lambda _project: []
        ap._recent_staged_research_drafts = lambda _project_id: []
        ap._fetch_last_tick_tool_log = lambda _project_id: None
        ap.logger.disabled = True

        project = {
            "id": 42,
            "title": "Failure Cooldown",
            "topic": "runtime",
            "goal": "Do not starve other projects.",
            "state": "researching",
            "plan": {"goals": [], "steps": []},
            "research_notes": [],
            "turn_count": 5,
            "last_run_at": None,
            "created_at": None,
        }
        try:
            await ap._run_one_tick(project)
        except RuntimeError as exc:
            assert "simulated autonomous failure" in str(exc)
        else:
            raise AssertionError("_run_one_tick should have raised")

        assert any(event_type == "tick_error" for event_type, _content in events)
        assert any("SET last_run_at = NOW()" in sql and "turn_count" not in sql for sql, _params in updates)
        assert not any("turn_count = turn_count + 1" in sql for sql, _params in updates)
    finally:
        agents.get_agent = original_get_agent
        bot._chat_with_tools = original_chat
        ap.db_execute = original_execute
        ap._log_event = original_log_event
        ap._notify_telegram = original_notify
        ap._fetch_pending_advisories = original_fetch_advisories
        ap._recent_notes = original_recent_notes
        ap._recent_staged_research_drafts = original_recent_staged
        ap._fetch_last_tick_tool_log = original_last_log
        ap.logger.disabled = original_logger_disabled


def _assert_collect_tick_actions_includes_publications() -> None:
    import autonomous_project as ap

    original_query = ap.db_query

    def fake_query(sql, params=()):
        assert "FROM autonomous_project_events" in sql
        return [
            {"event_type": "note_added", "content": "saved note", "meta": {}},
            {"event_type": "publication_created", "content": "research: Public Report\n/reports/public", "meta": {}},
            {"event_type": "plan_revised", "content": "next plan", "meta": {}},
            {"event_type": "state_transition", "content": "finished", "meta": {"from": "researching", "to": "paused"}},
        ]

    try:
        ap.db_query = fake_query
        actions = ap._collect_tick_actions(7, "2026-05-25T10:00:00+00:00")
        assert actions["notes"] == ["saved note"]
        assert actions["publications"] == ["research: Public Report\n/reports/public"]
        assert actions["plan_rationale"] == "next plan"
        assert actions["state_change"] == ("researching", "paused", "finished")
    finally:
        ap.db_query = original_query


async def _assert_successful_noop_tick_logs_no_durable_action() -> None:
    import autonomous_project as ap

    original_get_agent = __import__("agents").get_agent
    original_chat = __import__("telegram.bot").bot._chat_with_tools
    original_execute = ap.db_execute
    original_log_event = ap._log_event
    original_notify = ap._notify_telegram
    original_collect = ap._collect_tick_actions
    original_fetch_advisories = ap._fetch_pending_advisories
    original_mark_advisories = ap._mark_advisories_consumed
    original_recent_notes = ap._recent_notes
    original_recent_staged = ap._recent_staged_research_drafts
    original_last_log = ap._fetch_last_tick_tool_log

    events: list[tuple[str, str, dict]] = []
    consumed_advisories: list[tuple[int, list[int]]] = []
    notifications: list[dict] = []

    class DummySpec:
        max_rounds = 2
        model = None
        budget_usd = 0.1
        finalization_tools = []
        terminal_tools = []

        def effective_provider(self, _configured_provider):
            return "openai"

        def filter_tools(self, _tools, _handlers):
            return [], {}

        def render_prompt(self, *, provider="claude", **_kwargs):
            return "system"

    async def quiet_chat(*_args, **kwargs):
        tracker = kwargs.get("budget_tracker")
        if isinstance(tracker, dict):
            tracker["rounds_used"] = 1
            tracker["total_cost"] = 0.001
        return "No durable change this tick.\n\nSelf-critique: no project state was saved."

    def fake_log_event(_project_id, event_type, content="", meta=None, **_kwargs):
        events.append((event_type, content, meta or {}))

    async def fake_notify(_project, _result_text, actions, runtime):
        notifications.append({"actions": actions, "runtime": runtime})

    import agents
    import telegram.bot as bot

    try:
        agents.get_agent = lambda _name: DummySpec()
        bot._chat_with_tools = quiet_chat
        ap.db_execute = lambda *_args, **_kwargs: None
        ap._log_event = fake_log_event
        ap._notify_telegram = fake_notify
        ap._collect_tick_actions = lambda _project_id, _since: {
            "notes": [],
            "publications": [],
            "plan_rationale": None,
            "state_change": None,
        }
        ap._fetch_pending_advisories = lambda _project_id: [{"id": 91, "content": "do the advised work", "created_at": None}]
        ap._mark_advisories_consumed = lambda project_id, ids: consumed_advisories.append((project_id, ids))
        ap._recent_notes = lambda _project: []
        ap._recent_staged_research_drafts = lambda _project_id: []
        ap._fetch_last_tick_tool_log = lambda _project_id: None

        project = {
            "id": 43,
            "title": "No Durable Action",
            "topic": "runtime",
            "goal": "Surface no-op ticks.",
            "state": "researching",
            "plan": {"goals": [], "steps": []},
            "research_notes": [],
            "turn_count": 6,
            "last_run_at": None,
            "created_at": None,
        }
        result = await ap._run_one_tick(project)
        assert result["project_id"] == 43
        assert not consumed_advisories
        assert any(event_type == "advisories_retained_no_durable_action" for event_type, _content, _meta in events)
        no_action_events = [meta for event_type, _content, meta in events if event_type == "tick_no_durable_action"]
        assert no_action_events and no_action_events[-1].get("retained_advisory_ids") == [91]
        assert notifications and notifications[-1]["actions"]["publications"] == []
    finally:
        agents.get_agent = original_get_agent
        bot._chat_with_tools = original_chat
        ap.db_execute = original_execute
        ap._log_event = original_log_event
        ap._notify_telegram = original_notify
        ap._collect_tick_actions = original_collect
        ap._fetch_pending_advisories = original_fetch_advisories
        ap._mark_advisories_consumed = original_mark_advisories
        ap._recent_notes = original_recent_notes
        ap._recent_staged_research_drafts = original_recent_staged
        ap._fetch_last_tick_tool_log = original_last_log


def _assert_autonomous_cli_events_orders_by_id() -> None:
    import argparse
    import contextlib
    import io
    import scripts.autonomous_cli as cli

    original_ensure = cli._ensure_tables
    original_query = cli.db_query

    def fake_query(sql, params=()):
        assert "ORDER BY created_at DESC, id DESC" in sql
        assert params == (7, 5)
        return []

    try:
        cli._ensure_tables = lambda: None
        cli.db_query = fake_query
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli._cmd_events(argparse.Namespace(project_id=7, limit=5))
        assert rc == 0
        assert "(no events)" in buf.getvalue()
    finally:
        cli._ensure_tables = original_ensure
        cli.db_query = original_query


def _assert_autonomous_cli_list_includes_operational_signals() -> None:
    import contextlib
    import io
    import scripts.autonomous_cli as cli

    original_ensure = cli._ensure_tables
    original_query = cli.db_query

    def fake_query(sql, params=()):
        assert "pending_advisories" in sql
        assert "last_event_type" in sql
        assert "ORDER BY ev.created_at DESC, ev.id DESC" in sql
        return [{
            "id": 7,
            "title": "CLI List Signals",
            "state": "researching",
            "turn_count": 5,
            "last_run_at": datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc),
            "created_at": None,
            "pending_advisories": 2,
            "last_event_type": "tick_no_durable_action",
            "last_event_at": datetime(2026, 5, 25, 10, 5, tzinfo=timezone.utc),
        }]

    try:
        cli._ensure_tables = lambda: None
        cli.db_query = fake_query
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli._cmd_list(None)
        output = buf.getvalue()
        assert rc == 0
        assert "advice=2" in output
        assert "event=tick_no_durable_action" in output
        assert "CLI List Signals" in output
    finally:
        cli._ensure_tables = original_ensure
        cli.db_query = original_query


def _assert_autonomous_cli_show_uses_note_table() -> None:
    import argparse
    import scripts.autonomous_cli as cli

    original_ensure = cli._ensure_tables
    original_query = cli.db_query
    original_query_one = cli.db_query_one

    def fake_query_one(sql, params=()):
        if "FROM autonomous_projects WHERE id" in sql:
            return {
                "id": 7,
                "title": "CLI Test",
                "topic": "operator view",
                "goal": "Show durable notes.",
                "state": "researching",
                "plan": {"goals": ["observe"], "steps": ["show"]},
                "research_notes": [{"turn": 0, "text": "legacy note", "sources": []}],
                "turn_count": 4,
                "last_run_at": None,
            }
        if "COUNT(*) AS count FROM autonomous_project_notes" in sql:
            return {"count": 2}
        raise AssertionError(f"unexpected query_one: {sql}")

    def fake_query(sql, params=()):
        if "FROM autonomous_project_notes" in sql:
            return [
                {"turn": 2, "text": "newer CLI durable note", "sources": '["https://source/2"]', "created_at": None},
                {"turn": 1, "text": "older CLI durable note", "sources": ["https://source/1"], "created_at": None},
            ]
        if "FROM autonomous_project_advisories" in sql:
            return [
                {"id": 3, "content": "pending CLI advice", "created_at": None, "consumed_at": None},
                {"id": 2, "content": "consumed CLI advice", "created_at": None, "consumed_at": datetime(2026, 5, 25, 9, 0, tzinfo=timezone.utc)},
            ]
        if params and len(params) >= 2 and params[1] == "tick_tool_log":
            return [{
                "content": "  [1] vector_search({\"query\": \"done already\"}) -> result",
                "meta": {"turn": 4, "rounds_used": 3, "tool_calls": 1, "cost_usd": 0.04},
                "created_at": None,
            }]
        if params and len(params) >= 2 and params[1] == "tick_error":
            return [{"content": "ValueError: broken CLI tick", "meta": {}, "created_at": None}]
        if params and len(params) >= 2 and params[1] == "tick_no_durable_action":
            return [{"content": "tick completed without durable CLI action", "meta": {}, "created_at": None}]
        raise AssertionError(f"unexpected query: {sql}")

    try:
        cli._ensure_tables = lambda: None
        cli.db_query = fake_query
        cli.db_query_one = fake_query_one
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli._cmd_show(argparse.Namespace(project_id=7))
        output = buf.getvalue()
        assert rc == 0
        assert "operator_advisories: pending=1 recent_consumed=1" in output
        assert "pending CLI advice" in output
        assert "consumed CLI advice" in output
        assert "source=autonomous_project_notes" in output
        assert "newer CLI durable note" in output
        assert "older CLI durable note" in output
        assert "last_tick_error: ?" in output
        assert "broken CLI tick" in output
        assert "last_tick_no_durable_action: ?" in output
        assert "without durable CLI action" in output
        assert "last_tick_tool_log: turn=4, rounds=3, tools=1, cost=$0.040" in output
        assert "done already" in output
        assert "legacy note" not in output
    finally:
        cli._ensure_tables = original_ensure
        cli.db_query = original_query
        cli.db_query_one = original_query_one


def _assert_orchestrator_autonomous_status_includes_operational_signals() -> None:
    import db
    from telegram.bot import _format_autonomous_status

    original_query = db.query

    def fake_query(sql, params=()):
        assert "pending_advisories" in sql
        assert "last_event_type" in sql
        return [{
            "id": 7,
            "title": "Operational Signals",
            "state": "researching",
            "turn_count": 4,
            "last_run_at": datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc),
            "pending_advisories": 2,
            "last_event_type": "tick_error",
            "last_event_at": datetime(2026, 5, 25, 10, 5, tzinfo=timezone.utc),
        }]

    try:
        db.query = fake_query
        markdown = _format_autonomous_status("deepseek")
        assert "pending advice 2" in markdown
        assert "last event tick_error" in markdown
        assert "Operational Signals" in markdown

        xml = _format_autonomous_status("claude")
        assert xml.startswith("<autonomous-agent-status>")
        assert "pending advice 2" in xml
    finally:
        db.query = original_query


async def _assert_telegram_projects_lists_operational_signals() -> None:
    import telegram.commands as commands

    original_query = commands._query
    original_ctx = getattr(commands, "_ctx", None)

    class User:
        id = 1

    class Message:
        from_user = User()

        def __init__(self):
            self.replies: list[str] = []

        async def answer(self, text, **_kwargs):
            self.replies.append(text)

    def fake_query(sql, params=()):
        assert "pending_advisories" in sql
        assert "last_event_type" in sql
        return [{
            "id": 7,
            "title": "Project List Signals",
            "state": "researching",
            "turn_count": 3,
            "last_run_at": datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc),
            "pending_advisories": 1,
            "last_event_type": "tick_error",
            "last_event_at": datetime(2026, 5, 25, 10, 5, tzinfo=timezone.utc),
        }]

    try:
        commands._query = fake_query
        commands._ctx = {"is_allowed": lambda _user_id: True}
        message = Message()
        await commands.cmd_projects(message)
        assert message.replies
        output = message.replies[-1]
        assert "advice=1" in output
        assert "event=tick_error" in output
        assert "Project List Signals" in output
    finally:
        commands._query = original_query
        if original_ctx is None:
            try:
                delattr(commands, "_ctx")
            except AttributeError:
                pass
        else:
            commands._ctx = original_ctx


async def _assert_telegram_project_show_includes_tick_signals() -> None:
    import telegram.commands as commands

    original_query = commands._query
    original_query_one = commands._query_one
    original_ctx = getattr(commands, "_ctx", None)

    class User:
        id = 1

    class Message:
        from_user = User()
        text = "/project 7 show"

        def __init__(self):
            self.replies: list[str] = []

        async def answer(self, text, **_kwargs):
            self.replies.append(text)

    def fake_query_one(sql, params=()):
        if "FROM autonomous_projects" in sql:
            return {
                "id": 7,
                "title": "Project Detail Signals",
                "topic": "operator view",
                "goal": "Expose recent failure state.",
                "state": "researching",
                "plan": {},
                "turn_count": 4,
                "last_run_at": datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc),
                "max_publications_per_day": 2,
                "cooldown_after_publish_minutes": 30,
            }
        if "COUNT(*)::int AS n FROM autonomous_project_notes" in sql:
            return {"n": 2}
        if "event_type = 'publication_created'" in sql:
            return {"n": 1}
        if "event_type = 'tick_error'" in sql:
            return {
                "content": "RuntimeError: detail smoke failure",
                "meta": {},
                "created_at": datetime(2026, 5, 25, 10, 5, tzinfo=timezone.utc),
            }
        if "event_type = 'tick_no_durable_action'" in sql:
            return {
                "content": "tick completed without durable telegram action",
                "meta": {},
                "created_at": datetime(2026, 5, 25, 10, 6, tzinfo=timezone.utc),
            }
        if "event_type = 'tick_tool_log'" in sql:
            return {
                "content": "  [1] read_self({\"content_type\": \"autonomous_project\"}) -> ok",
                "meta": {"turn": 4, "rounds_used": 2, "tool_calls": 1, "cost_usd": 0.012},
                "created_at": datetime(2026, 5, 25, 10, 4, tzinfo=timezone.utc),
            }
        raise AssertionError(f"unexpected query_one: {sql}")

    def fake_query(sql, params=()):
        if "FROM autonomous_project_notes" in sql:
            return [{
                "turn": 4,
                "text": "durable telegram note",
                "sources": [],
                "created_at": datetime(2026, 5, 25, 10, 3, tzinfo=timezone.utc),
            }]
        if "FROM autonomous_project_advisories" in sql:
            return [{
                "id": 9,
                "content": "Prioritize the waiting staged draft before new research.",
                "created_at": datetime(2026, 5, 25, 10, 2, tzinfo=timezone.utc),
            }]
        raise AssertionError(f"unexpected query: {sql}")

    try:
        commands._query = fake_query
        commands._query_one = fake_query_one
        commands._ctx = {"is_allowed": lambda _user_id: True}
        message = Message()
        await commands.cmd_project(message)
        assert message.replies
        output = message.replies[-1]
        assert "Project Detail Signals" in output
        assert "pending_advice: 1" in output
        assert "Prioritize the waiting staged draft" in output
        assert "durable telegram note" in output
        assert "last_tick_error:" in output
        assert "detail smoke failure" in output
        assert "last_tick_no_durable_action:" in output
        assert "without durable telegram action" in output
        assert "last_tick_tool_log: turn=4, rounds=2, tools=1, cost=$0.012" in output
        assert "read_self" in output
    finally:
        commands._query = original_query
        commands._query_one = original_query_one
        if original_ctx is None:
            try:
                delattr(commands, "_ctx")
            except AttributeError:
                pass
        else:
            commands._ctx = original_ctx


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


async def _assert_web_autonomous_summary_includes_publication_events() -> None:
    import web_chat

    original_query = web_chat.db_query

    def fake_query(sql, params=()):
        if "FROM autonomous_projects" in sql:
            return [{
                "id": 7,
                "title": "Public Project",
                "topic": "public autonomous work",
                "goal": "Publish public outputs.",
                "plan": {"goals": ["ship report"], "steps": ["review sources"]},
                "state": "researching",
                "turn_count": 4,
                "last_run_at": None,
                "updated_at": None,
            }]
        if "FROM autonomous_project_events" in sql:
            assert "publication_created" in sql
            assert "ORDER BY created_at DESC, id DESC" in sql
            return [{
                "event_type": "publication_created",
                "content": "research: Public Report\nhttps://cyber-lenin.com/reports/research/public-report",
                "created_at": "2026-05-25 10:00:00+00",
            }]
        raise AssertionError(f"unexpected query: {sql}")

    try:
        web_chat.db_query = fake_query
        output = await web_chat._exec_web_read_self(content_type="autonomous_project", id=7)
        assert "Autonomous project status, public summary only" in output
        assert "publication_created" in output
        assert "Public Report" in output
        assert "operator" not in output.lower() or "not exposed" in output
    finally:
        web_chat.db_query = original_query


async def _assert_stage_public_records_autonomous_staged_draft_event() -> None:
    import runtime_tools.research as research

    original_save = research._save_publication_draft
    original_get = research.research_store.get_document
    original_upsert = research.research_store.upsert_document
    original_excerpt = research.research_store.extract_excerpt
    original_record = research.record_autonomous_staged_draft

    calls: list[dict] = []

    def fake_upsert(**kwargs):
        assert kwargs["filename"] == "autonomous-draft.md"
        assert kwargs["status"] == "staged"
        assert kwargs["source_task_id"] == 123
        return ({"id": 55, "content_sha256": "abcdef1234567890"}, False)

    def fake_record(**kwargs):
        calls.append(kwargs)

    try:
        research._save_publication_draft = lambda **_kwargs: "/tmp/autonomous-draft.md"
        research.research_store.get_document = lambda *_args, **_kwargs: None
        research.research_store.upsert_document = fake_upsert
        research.research_store.extract_excerpt = lambda _document: "excerpt"
        research.record_autonomous_staged_draft = fake_record

        result = await research._exec_research_document_publish_public(
            title="Autonomous Draft",
            content="본문과 footnote 형식 인용[^1]\n\n[^1]: Example source. https://example.com",
            filename="autonomous-draft.md",
            fact_check_passed=False,
            source_task_id=123,
            broadcast=False,
        )
        assert "Draft saved, not published" in result
        assert calls == [{
            "publication_kind": "research",
            "title": "Autonomous Draft",
            "public_url": "https://cyber-lenin.com/reports/research/autonomous-draft",
            "meta": {
                "filename": "autonomous-draft.md",
                "research_document_id": 55,
                "status": "staged",
                "source_task_id": 123,
            },
        }]
    finally:
        research._save_publication_draft = original_save
        research.research_store.get_document = original_get
        research.research_store.upsert_document = original_upsert
        research.research_store.extract_excerpt = original_excerpt
        research.record_autonomous_staged_draft = original_record


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
    _assert_autonomous_base_finalization_tools()
    _assert_agent_runtime_dynamic_reload()
    _assert_autonomous_prompt_surfaces_staged_drafts()
    _assert_staged_drafts_prioritize_project_events()
    await _assert_read_self_autonomous_project_uses_note_table()
    _assert_autonomous_project_selection_prioritizes_advice()
    await _assert_autonomous_tick_failure_updates_cooldown()
    _assert_collect_tick_actions_includes_publications()
    await _assert_successful_noop_tick_logs_no_durable_action()
    _assert_autonomous_cli_events_orders_by_id()
    _assert_autonomous_cli_list_includes_operational_signals()
    _assert_autonomous_cli_show_uses_note_table()
    _assert_orchestrator_autonomous_status_includes_operational_signals()
    await _assert_telegram_projects_lists_operational_signals()
    await _assert_telegram_project_show_includes_tick_signals()
    _assert_web_political_line_dynamic_reload()
    await _assert_web_autonomous_summary_includes_publication_events()
    await _assert_stage_public_records_autonomous_staged_draft_event()
    _assert_diary_web_context_injection()
    await _assert_diary_runtime_route_fallbacks()
    await _assert_guarded_diary_save_handler_accepts_tool_payloads()
    print("runtime smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
