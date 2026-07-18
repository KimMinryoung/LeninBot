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

    kimi_task = await resolve_runtime_profile("task", provider_override="kimi")
    assert kimi_task.provider == "kimi"
    assert kimi_task.prompt_format == "markdown"
    assert kimi_task.alias == "kimi_k3"
    assert kimi_task.model_id == "kimi-k3"
    assert kimi_task.display_name == "Kimi K3"
    assert kimi_task.max_tokens >= bot_config._KIMI_MIN_OUTPUT_TOKENS

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


def _assert_writer_heavy_policy_uses_gateway_defaults() -> None:
    from tool_gateway.inference import (
        DEFAULT_AGENT_MAX_INPUT_TOKENS, DEFAULT_AGENT_MAX_OUTPUT_TOKENS,
    )
    from writer.config import writer_call_policy

    for role in ("main", "revision"):
        policy = writer_call_policy(role)
        assert policy.max_input_tokens == DEFAULT_AGENT_MAX_INPUT_TOKENS
        assert policy.max_output_tokens == DEFAULT_AGENT_MAX_OUTPUT_TOKENS
    assert writer_call_policy("diagnosis").max_output_tokens == 8000
    assert writer_call_policy("line_edit").max_output_tokens == 8000
    assert writer_call_policy("research").max_output_tokens == 3000


def _assert_writer_kimi_catalog_and_resolution() -> None:
    import bot_config
    import claude_loop
    from writer.models import list_writer_models, resolve_writer_model

    original = bot_config._kimi_anthropic_client
    dummy = object()
    try:
        bot_config._kimi_anthropic_client = dummy
        rows = {row["key"]: row for row in list_writer_models()}
        assert rows["kimi_k3"] == {
            "key": "kimi_k3",
            "id": "kimi-k3",
            "display_name": "Kimi K3",
            "provider": "kimi",
            "input_price_per_mtok": 3.0,
            "output_price_per_mtok": 15.0,
            "available": True,
            "default": False,
        }
        client, model, display, extra = resolve_writer_model("kimi_k3")
        assert client is dummy
        assert model == "kimi-k3"
        assert display == "Kimi K3"
        assert extra == {}
        assert claude_loop._pricing_for("kimi-k3") == {
            "input": 3.0 / 1_000_000,
            "output": 15.0 / 1_000_000,
            "cache_creation": 3.0 / 1_000_000,
            "cache_read": 0.30 / 1_000_000,
        }
    finally:
        bot_config._kimi_anthropic_client = original


def _assert_openai_input_replay_checkpoint() -> None:
    import json as _json
    from openai_tool_loop import _checkpoint_tool_results_for_replay

    tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "read_self",
            "arguments": _json.dumps({
                "chapter": 3,
                "start_sentence": "첫 문장",
                "end_sentence": "끝 문장",
            }, ensure_ascii=False),
        },
    }]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call-1", "content": "원문" * 5000},
    ]
    checkpointed, estimated = _checkpoint_tool_results_for_replay(messages, 500)
    assert checkpointed[1]["tool_calls"] == tool_calls
    assert checkpointed[2]["content"].startswith("[Input checkpoint:")
    assert estimated <= 500

    write_calls = [{
        "id": "call-write",
        "type": "function",
        "function": {"name": "broadcast_to_channel", "arguments": "{}"},
    }]
    write_messages = [
        {"role": "assistant", "content": "", "tool_calls": write_calls},
        {"role": "tool", "tool_call_id": "call-write", "content": "sent" * 5000},
    ]
    unchanged, write_estimate = _checkpoint_tool_results_for_replay(write_messages, 500)
    assert unchanged[1]["content"] == write_messages[1]["content"]
    assert write_estimate > 500


def _assert_inference_reasoning_policy() -> None:
    from tool_gateway.inference import AgentInferencePolicy, resolve_inference_extra

    base = dict(
        max_input_tokens=160000, max_output_tokens=32000, max_rounds=10,
        budget_usd=1.0, max_output_continuations=2, thinking_budget_tokens=8192,
    )
    thinking = AgentInferencePolicy(**base, thinking_policy="thinking")
    disabled = AgentInferencePolicy(**base, thinking_policy="disabled")
    model_default = AgentInferencePolicy(**base, thinking_policy="model_default")
    assert resolve_inference_extra(thinking, "claude")["thinking"] == {
        "type": "enabled", "budget_tokens": 8192,
    }
    assert resolve_inference_extra(disabled, "claude") == {}
    assert resolve_inference_extra(thinking, "openai")["extra_body"] == {
        "reasoning_effort": "high",
    }
    assert resolve_inference_extra(disabled, "openai")["extra_body"] == {
        "reasoning_effort": "none",
    }
    assert resolve_inference_extra(model_default, "openai") == {}


async def _assert_openai_continuation_extends_round_limit() -> None:
    from types import SimpleNamespace
    from openai_tool_loop import chat_with_tools

    class _Completions:
        def __init__(self):
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            finish = "length" if self.calls < 3 else "stop"
            text = f"part-{self.calls}"
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason=finish,
                    message=SimpleNamespace(content=text, tool_calls=None, refusal=None),
                )],
                usage=None,
            )

    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = await chat_with_tools(
        [{"role": "user", "content": "continue test"}],
        client=client, model="gpt-test", tools=[], tool_handlers={},
        system_prompt="system", max_rounds=1, max_tokens=8, budget_usd=1.0,
        continue_on_length=True, max_length_continuations=2, return_metadata=True,
    )
    assert completions.calls == 3
    assert result["continuations_used"] == 2
    assert result["text"] == "part-1\npart-2\npart-3"


async def _assert_kimi_preserves_reasoning_for_tool_replay() -> None:
    from types import SimpleNamespace
    from openai_tool_loop import chat_with_tools

    seen: list[dict] = []

    class _Completions:
        async def create(self, **kwargs):
            seen.append(kwargs)
            if len(seen) == 1:
                tool_call = SimpleNamespace(
                    id="call-kimi-1",
                    function=SimpleNamespace(name="lookup", arguments='{"query":"x"}'),
                )
                message = SimpleNamespace(
                    content="",
                    reasoning_content="private reasoning",
                    tool_calls=[tool_call],
                    refusal=None,
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
                    usage=None,
                )
            replay = kwargs["messages"]
            assistant = next(m for m in replay if m.get("role") == "assistant")
            assert assistant["reasoning_content"] == "private reasoning"
            message = SimpleNamespace(
                content="final answer", reasoning_content="final private reasoning",
                tool_calls=None, refusal=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=message)],
                usage=None,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    result = await chat_with_tools(
        [{"role": "user", "content": "test"}],
        client=client,
        model="kimi-k3",
        tools=[{
            "name": "lookup",
            "description": "Lookup",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }],
        tool_handlers={"lookup": lambda query: f"found {query}"},
        system_prompt="system",
        max_rounds=2,
        max_tokens=128,
        budget_usd=1.0,
        extra_body={"reasoning_effort": "max"},
        sdk_max_token_param="max_tokens",
        include_parallel_tool_calls=False,
        preserve_reasoning_content=True,
    )
    assert result == "final answer"
    assert len(seen) == 2
    assert seen[0]["extra_body"] == {"reasoning_effort": "max"}
    assert seen[0]["max_tokens"] == 128
    assert "parallel_tool_calls" not in seen[0]


async def _assert_kimi_content_filter_falls_back_in_forced_final() -> None:
    from types import SimpleNamespace
    from openai_tool_loop import chat_with_tools, _looks_like_prompt_content_filter

    class _FilterError(Exception):
        status_code = 400
        body = {
            "error": {
                "message": "The request was rejected because it was considered high risk",
                "param": "prompt",
                "type": "content_filter",
            }
        }

    assert _looks_like_prompt_content_filter(_FilterError("Error code: 400"))
    assert not _looks_like_prompt_content_filter(
        ValueError("Error code: 400 - invalid max_tokens parameter")
    )
    assert not _looks_like_prompt_content_filter(
        ValueError("Error code: 500 - content_filter backend unavailable")
    )

    primary_seen: list[dict] = []
    fallback_seen: list[dict] = []
    saved: list[str] = []

    class _PrimaryCompletions:
        async def create(self, **kwargs):
            primary_seen.append(kwargs)
            if len(primary_seen) == 1:
                tool_call = SimpleNamespace(
                    id="call-work",
                    function=SimpleNamespace(name="lookup", arguments='{"query":"x"}'),
                )
                return SimpleNamespace(
                    model="kimi-k3",
                    choices=[SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            content="", reasoning_content="kimi reasoning",
                            tool_calls=[tool_call], refusal=None,
                        ),
                    )],
                    usage=None,
                )
            raise _FilterError("Error code: 400 - content_filter prompt high risk")

    class _FallbackCompletions:
        async def create(self, **kwargs):
            fallback_seen.append(kwargs)
            if len(fallback_seen) == 1:
                tool_call = SimpleNamespace(
                    id="call-save",
                    function=SimpleNamespace(name="save_diary", arguments='{"content":"entry"}'),
                )
                return SimpleNamespace(
                    model="deepseek-v4-pro",
                    choices=[SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(content="", tool_calls=[tool_call], refusal=None),
                    )],
                    usage=None,
                )
            return SimpleNamespace(
                model="deepseek-v4-pro",
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="saved", tool_calls=None, refusal=None),
                )],
                usage=None,
            )

    primary = SimpleNamespace(chat=SimpleNamespace(completions=_PrimaryCompletions()))
    fallback = SimpleNamespace(chat=SimpleNamespace(completions=_FallbackCompletions()))

    def _save_diary(content: str):
        saved.append(content)
        return "ok"

    result = await chat_with_tools(
        [{"role": "user", "content": "sensitive diary context"}],
        client=primary,
        model="kimi-k3",
        tools=[
            {
                "name": "lookup",
                "description": "Lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "save_diary",
                "description": "Save diary",
                "input_schema": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            },
        ],
        tool_handlers={"lookup": lambda query: f"found {query}", "save_diary": _save_diary},
        system_prompt="system",
        max_rounds=1,
        max_tokens=128,
        budget_usd=1.0,
        finalization_tools=["save_diary"],
        provider_label="kimi",
        extra_body={"reasoning_effort": "max"},
        sdk_max_token_param="max_tokens",
        include_parallel_tool_calls=False,
        preserve_reasoning_content=True,
        content_filter_fallback_client=fallback,
        content_filter_fallback_model="deepseek-v4-pro",
        content_filter_fallback_label="deepseek",
    )
    assert result == "saved"
    assert saved == ["entry"]
    assert len(primary_seen) == 2
    assert len(fallback_seen) == 2
    assert primary_seen[1]["messages"] == fallback_seen[0]["messages"]
    assert fallback_seen[0]["model"] == "deepseek-v4-pro"
    assert "extra_body" not in fallback_seen[0]


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
            "max_input_tokens": int(cfg.get("max_input_tokens", base["max_input_tokens"])),
            "max_output_tokens": int(cfg.get("max_output_tokens", base["max_output_tokens"])),
            "max_output_continuations": int(cfg.get("max_output_continuations", base["max_output_continuations"])),
            "thinking_policy": str(cfg.get("thinking_policy", base["thinking_policy"])),
            "thinking_budget_tokens": int(cfg.get("thinking_budget_tokens", base["thinking_budget_tokens"])),
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
        assert spec.max_input_tokens == expected["max_input_tokens"], name
        assert spec.max_output_tokens == expected["max_output_tokens"], name
        assert spec.max_output_continuations == expected["max_output_continuations"], name
        assert spec.thinking_policy == expected["thinking_policy"], name
        assert spec.thinking_budget_tokens == expected["thinking_budget_tokens"], name
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
                        "max_input_tokens": 48000,
                        "max_output_tokens": 12000,
                        "max_output_continuations": 3,
                        "thinking_policy": "disabled",
                        "thinking_budget_tokens": 6000,
                    }
                }),
                encoding="utf-8",
            )
            runtime_config.apply_agent_runtime_config(registry)
            assert registry["dummy"].provider == "claude"
            assert registry["dummy"].model == "sonnet"
            assert registry["dummy"].budget_usd == 2.0
            assert registry["dummy"].max_rounds == 12
            assert registry["dummy"].max_input_tokens == 48000
            assert registry["dummy"].max_output_tokens == 12000
            assert registry["dummy"].max_output_continuations == 3
            assert registry["dummy"].thinking_policy == "disabled"
            assert registry["dummy"].thinking_budget_tokens == 6000

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

        advisories = [{"id": 91, "content": "Finish the waiting staged draft before starting new research.", "created_at": None}]
        markdown_prompt = ap._build_task_prompt(project, turn_budget=3, advisories=advisories, provider="deepseek")
        assert "### Operator Advice" in markdown_prompt
        assert "remain pending until this tick saves durable project work" in markdown_prompt
        assert "shown once" not in markdown_prompt
        assert "### Staged Research Drafts" in markdown_prompt
        assert "[this-project] slug=draft-report" in markdown_prompt
        assert "status='staged'" in markdown_prompt
        assert "### Recent Tick Warnings" in markdown_prompt
        assert "tick_no_durable_action" in markdown_prompt
        assert "advisories_retained_no_durable_action" in markdown_prompt

        from agents import get_agent

        system_prompt = get_agent("autonomous_project").render_prompt(provider="deepseek")
        assert "Advice remains pending until a tick saves" in system_prompt
        assert "Shown once" not in system_prompt

        xml_prompt = ap._build_task_prompt(project, turn_budget=3, advisories=advisories, provider="claude")
        assert "<operator-advice>" in xml_prompt
        assert "remain pending until this tick saves durable project work" in xml_prompt
        assert "shown once" not in xml_prompt
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
        if "event_type = 'research_draft_staged'" in sql:
            return [{"content": "research: Staged Draft\n/reports/research/staged-draft", "meta": {"filename": "staged-draft.md"}, "created_at": None}]
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
        assert "-- last staged research draft (?) --" in output
        assert "Staged Draft" in output
        assert "read_self(content_type=\"research_document\", slug=\"staged-draft\", status=\"staged\")" in output
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
            {"event_type": "research_draft_staged", "content": "research: Staged Report\n/reports/staged", "meta": {}},
            {"event_type": "publication_created", "content": "research: Public Report\n/reports/public", "meta": {}},
            {"event_type": "plan_revised", "content": "next plan", "meta": {}},
            {"event_type": "state_transition", "content": "finished", "meta": {"from": "researching", "to": "paused"}},
        ]

    try:
        ap.db_query = fake_query
        actions = ap._collect_tick_actions(7, "2026-05-25T10:00:00+00:00")
        assert actions["notes"] == ["saved note"]
        assert actions["staged_drafts"] == ["research: Staged Report\n/reports/staged"]
        assert actions["publications"] == ["research: Public Report\n/reports/public"]
        assert actions["plan_rationale"] == "next plan"
        assert actions["state_change"] == ("researching", "paused", "finished")
    finally:
        ap.db_query = original_query



async def _assert_successful_staged_draft_tick_consumes_advisories() -> None:
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

    async def staged_chat(*_args, **kwargs):
        tracker = kwargs.get("budget_tracker")
        if isinstance(tracker, dict):
            tracker["rounds_used"] = 1
            tracker["total_cost"] = 0.003
        return "Staged a public draft.\n\nSelf-critique: draft is ready for fact-checking."

    def fake_log_event(_project_id, event_type, content="", meta=None, **_kwargs):
        events.append((event_type, content, meta or {}))

    async def fake_notify(_project, _result_text, actions, runtime, tick_review=None):
        notifications.append({"actions": actions, "runtime": runtime})

    import agents
    import telegram.bot as bot

    try:
        agents.get_agent = lambda _name: DummySpec()
        bot._chat_with_tools = staged_chat
        ap.db_execute = lambda *_args, **_kwargs: None
        ap._log_event = fake_log_event
        ap._notify_telegram = fake_notify
        ap._collect_tick_actions = lambda _project_id, _since: {
            "notes": [],
            "staged_drafts": ["research: Staged Report\n/reports/research/staged-report"],
            "publications": [],
            "plan_rationale": None,
            "state_change": None,
        }
        ap._fetch_pending_advisories = lambda _project_id: [
            {"id": 201, "content": "stage the checked draft", "created_at": None},
        ]
        ap._mark_advisories_consumed = lambda project_id, ids: consumed_advisories.append((project_id, ids))
        ap._recent_notes = lambda _project: []
        ap._recent_staged_research_drafts = lambda _project_id: []
        ap._fetch_last_tick_tool_log = lambda _project_id: None

        project = {
            "id": 45,
            "title": "Staged Draft Advisory Action",
            "topic": "runtime",
            "goal": "Treat staged drafts as durable progress.",
            "state": "researching",
            "plan": {"goals": [], "steps": []},
            "research_notes": [],
            "turn_count": 8,
            "last_run_at": None,
            "created_at": None,
        }
        result = await ap._run_one_tick(project)
        assert result["project_id"] == 45
        assert consumed_advisories == [(45, [201])]
        assert any(event_type == "advisories_consumed" for event_type, _content, _meta in events)
        assert not any(event_type == "tick_no_durable_action" for event_type, _content, _meta in events)
        assert notifications and notifications[-1]["actions"]["staged_drafts"] == ["research: Staged Report\n/reports/research/staged-report"]
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


async def _assert_successful_durable_tick_consumes_advisories() -> None:
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

    async def durable_chat(*_args, **kwargs):
        tracker = kwargs.get("budget_tracker")
        if isinstance(tracker, dict):
            tracker["rounds_used"] = 1
            tracker["total_cost"] = 0.002
        return "Saved a durable note.\n\nSelf-critique: advice was acted on."

    def fake_log_event(_project_id, event_type, content="", meta=None, **_kwargs):
        events.append((event_type, content, meta or {}))

    async def fake_notify(_project, _result_text, actions, runtime, tick_review=None):
        notifications.append({"actions": actions, "runtime": runtime})

    import agents
    import telegram.bot as bot

    try:
        agents.get_agent = lambda _name: DummySpec()
        bot._chat_with_tools = durable_chat
        ap.db_execute = lambda *_args, **_kwargs: None
        ap._log_event = fake_log_event
        ap._notify_telegram = fake_notify
        ap._collect_tick_actions = lambda _project_id, _since: {
            "notes": ["durable note from advised work"],
            "publications": [],
            "plan_rationale": None,
            "state_change": None,
        }
        ap._fetch_pending_advisories = lambda _project_id: [
            {"id": 101, "content": "do the durable work", "created_at": None},
            {"id": 102, "content": "record the result", "created_at": None},
        ]
        ap._mark_advisories_consumed = lambda project_id, ids: consumed_advisories.append((project_id, ids))
        ap._recent_notes = lambda _project: []
        ap._recent_staged_research_drafts = lambda _project_id: []
        ap._fetch_last_tick_tool_log = lambda _project_id: None

        project = {
            "id": 44,
            "title": "Durable Advisory Action",
            "topic": "runtime",
            "goal": "Consume advice only after durable work.",
            "state": "researching",
            "plan": {"goals": [], "steps": []},
            "research_notes": [],
            "turn_count": 7,
            "last_run_at": None,
            "created_at": None,
        }
        result = await ap._run_one_tick(project)
        assert result["project_id"] == 44
        assert consumed_advisories == [(44, [101, 102])]
        assert any(event_type == "advisories_consumed" for event_type, _content, _meta in events)
        assert not any(event_type == "advisories_retained_no_durable_action" for event_type, _content, _meta in events)
        assert not any(event_type == "tick_no_durable_action" for event_type, _content, _meta in events)
        assert notifications and notifications[-1]["actions"]["notes"] == ["durable note from advised work"]
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

    async def fake_notify(_project, _result_text, actions, runtime, tick_review=None):
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



def _assert_autonomous_cli_status_uses_config_without_db() -> None:
    import argparse
    import bot_config
    import contextlib
    import io
    import scripts.autonomous_cli as cli

    original_config = dict(bot_config._config)
    original_load = bot_config._load_config
    original_ensure = cli._ensure_tables
    original_systemctl_state = cli._systemctl_state
    original_systemctl_show = cli._systemctl_show
    try:
        cfg = dict(original_config)
        cfg.update({
            "autonomous_active": False,
            "autonomous_provider": "deepseek",
            "autonomous_model": "medium",
        })
        bot_config._config.clear()
        bot_config._config.update(cfg)
        bot_config._load_config = lambda: dict(cfg)
        cli._ensure_tables = lambda: (_ for _ in ()).throw(AssertionError("status must not touch DB tables"))
        cli._systemctl_state = lambda unit: "active" if unit.endswith(".timer") else "inactive"
        def fake_systemctl_show(unit, *props):
            if unit.endswith(".timer"):
                return {
                    "NextElapseUSecRealtime": "Mon 2026-05-25 14:17:00 UTC",
                    "LastTriggerUSec": "Mon 2026-05-25 13:17:02 UTC",
                }
            return {
                "Result": "success",
                "ExecMainStatus": "0",
                "InactiveEnterTimestamp": "Mon 2026-05-25 13:17:03 UTC",
            }

        cli._systemctl_show = fake_systemctl_show

        systemd_buf = io.StringIO()
        with contextlib.redirect_stdout(systemd_buf):
            systemd_rc = cli._cmd_status(argparse.Namespace(no_systemd=False, json=False))
        systemd_output = systemd_buf.getvalue()
        assert systemd_rc == 0
        assert "timer: active" in systemd_output
        assert "timer_next: Mon 2026-05-25 14:17:00 UTC" in systemd_output
        assert "timer_last: Mon 2026-05-25 13:17:02 UTC" in systemd_output
        assert "service: inactive" in systemd_output
        assert "service_result: success (exit=0)" in systemd_output
        assert "service_last_exit: Mon 2026-05-25 13:17:03 UTC" in systemd_output

        json_buf = io.StringIO()
        with contextlib.redirect_stdout(json_buf):
            json_rc = cli._cmd_status(argparse.Namespace(no_systemd=False, json=True))
        json_output = json.loads(json_buf.getvalue())
        assert json_rc == 0
        assert json_output["autonomous_active"] is False
        assert json_output["provider"] == "deepseek"
        assert json_output["model_id"] == "deepseek-v4-flash"
        assert json_output["timer"]["next"] == "Mon 2026-05-25 14:17:00 UTC"
        assert json_output["service"]["result"] == "success"
        assert json_output["service"]["exit_status"] == "0"

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli._cmd_status(argparse.Namespace(no_systemd=True, json=False))
        output = buf.getvalue()
        assert rc == 0
        assert "autonomous_active: false" in output
        assert "provider: deepseek" in output
        assert "model: medium (deepseek-v4-flash)" in output
        assert "paused by config" in output
        assert "timer:" not in output
    finally:
        bot_config._config.clear()
        bot_config._config.update(original_config)
        bot_config._load_config = original_load
        cli._ensure_tables = original_ensure
        cli._systemctl_state = original_systemctl_state
        cli._systemctl_show = original_systemctl_show


def _assert_autonomous_cli_main_reports_missing_db_config() -> None:
    import contextlib
    import io
    import scripts.autonomous_cli as cli

    original_list = cli._cmd_list
    original_argv = sys.argv

    try:
        cli._cmd_list = lambda _args: (_ for _ in ()).throw(
            RuntimeError("Missing database configuration: DB_PASSWORD")
        )
        sys.argv = ["autonomous_cli.py", "list"]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main()
        assert rc == 1
        assert "Missing database configuration: DB_PASSWORD" in err.getvalue()
        assert "Traceback" not in err.getvalue()
    finally:
        cli._cmd_list = original_list
        sys.argv = original_argv


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
        if "research_draft_staged" in sql and "research_documents" in sql:
            assert "rd.status = 'staged'" in sql
            return [{"content": "research: CLI Staged Draft\n/reports/research/cli-staged", "meta": {"slug": "cli-staged"}, "created_at": None}]
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
        assert "last_staged_research_draft: ?" in output
        assert "CLI Staged Draft" in output
        assert "read_self(content_type=\"research_document\", slug=\"cli-staged\", status=\"staged\")" in output
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
    import bot_config
    import db
    from telegram.bot import _format_autonomous_status

    original_query = db.query
    original_is_active = bot_config.is_autonomous_active

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
        bot_config.is_autonomous_active = lambda: False
        markdown = _format_autonomous_status("deepseek")
        assert "Loop is PAUSED by config" in markdown
        assert "pending advice 2" in markdown
        assert "last event tick_error" in markdown
        assert "Operational Signals" in markdown

        xml = _format_autonomous_status("claude")
        assert xml.startswith("<autonomous-agent-status>")
        assert "Loop is PAUSED by config" in xml
        assert "pending advice 2" in xml
    finally:
        db.query = original_query
        bot_config.is_autonomous_active = original_is_active


async def _assert_telegram_projects_lists_operational_signals() -> None:
    import bot_config
    import telegram.commands as commands

    original_query = commands._query
    original_ctx = getattr(commands, "_ctx", None)
    original_is_active = bot_config.is_autonomous_active

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
        bot_config.is_autonomous_active = lambda: False
        message = Message()
        await commands.cmd_projects(message)
        assert message.replies
        output = message.replies[-1]
        assert "inactive (autonomous_active=false" in output
        assert "advice=1" in output
        assert "event=tick_error" in output
        assert "Project List Signals" in output
    finally:
        commands._query = original_query
        bot_config.is_autonomous_active = original_is_active
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
        if "event_type = 'research_draft_staged'" in sql:
            return {
                "content": "research: Telegram Staged Draft\n/reports/research/telegram-staged",
                "meta": {"filename": "telegram-staged.md"},
                "created_at": datetime(2026, 5, 25, 10, 4, tzinfo=timezone.utc),
            }
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
        assert "last_staged_research_draft:" in output
        assert "Telegram Staged Draft" in output
        assert "read_self(content_type=\"research_document\", slug=\"telegram-staged\", status=\"staged\")" in output
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


def _assert_web_feedback_is_one_shot() -> None:
    import web_chat

    original_query = web_chat.db_query
    original_query_one = web_chat.db_query_one
    original_execute = web_chat.db_execute
    calls: dict[str, object] = {}

    def fake_query(sql, params=()):
        calls["load_sql"] = sql
        calls["load_params"] = params
        return [{
            "id": 11,
            "rating": 4,
            "tone_feedback": "more_in_character",
            "note": "avoid repeating Khrushchev correction",
            "updated_at": None,
        }]

    def fake_query_one(sql, params=()):
        calls["save_sql"] = sql
        calls["save_params"] = params
        return {"id": 12, "consumed_at": "now" if params[-1] is False else None}

    def fake_execute(sql, params=()):
        calls.setdefault("execute_calls", []).append((sql, params))

    try:
        web_chat.db_query = fake_query
        web_chat.db_query_one = fake_query_one
        web_chat.db_execute = fake_execute

        rows = web_chat._load_web_feedback_rows(["fp1"], "sess1", "yezhov", 8)
        assert rows and rows[0]["id"] == 11
        assert "f.consumed_at IS NULL" in calls["load_sql"]

        context = web_chat._render_web_feedback_context(rows, "deepseek")
        assert "for this next answer only" in context
        assert "avoid repeating Khrushchev correction" in context

        web_chat._mark_web_feedback_consumed([11])
        execute_calls = calls.get("execute_calls") or []
        assert execute_calls
        assert "SET consumed_at = COALESCE(consumed_at, now())" in execute_calls[-1][0]
        assert execute_calls[-1][1] == ([11],)

        saved = web_chat.save_web_chat_feedback(
            chat_log_id=5,
            session_id="sess1",
            fingerprint="fp1",
            persona="yezhov",
            tone_feedback="more_in_character",
            note="regenerate only",
            pending=False,
        )
        assert saved["consumed_at"] == "now"
        assert "CASE WHEN %s THEN NULL ELSE now() END" in calls["save_sql"]
        assert calls["save_params"][-1] is False
    finally:
        web_chat.db_query = original_query
        web_chat.db_query_one = original_query_one
        web_chat.db_execute = original_execute


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
    import bot_config
    import web_chat

    original_query = web_chat.db_query
    original_is_active = bot_config.is_autonomous_active

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
        bot_config.is_autonomous_active = lambda: False
        output = await web_chat._exec_web_read_self(content_type="autonomous_project", id=7)
        assert "Autonomous project status, public summary only" in output
        assert "Autonomous loop: paused by config" in output
        assert "publication_created" in output
        assert "Public Report" in output
        assert "operator" not in output.lower() or "not exposed" in output
    finally:
        web_chat.db_query = original_query
        bot_config.is_autonomous_active = original_is_active


async def _assert_web_public_summary_includes_autonomous_loop_state() -> None:
    import bot_config
    import web_chat

    original_query = web_chat.db_query
    original_is_active = bot_config.is_autonomous_active

    def fake_query(sql, params=()):
        assert "research_count" in sql
        return [{
            "research_count": 12,
            "static_page_count": 3,
            "active_project_count": 2,
        }]

    try:
        web_chat.db_query = fake_query
        bot_config.is_autonomous_active = lambda: False
        output = await web_chat._exec_web_read_self(content_type="system")
        assert "Public research reports: 12" in output
        assert "Static pages: 3" in output
        assert "Active autonomous projects: 2" in output
        assert "Autonomous loop: paused by config" in output
    finally:
        web_chat.db_query = original_query
        bot_config.is_autonomous_active = original_is_active


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
    from telegram.diary_mode import DEFAULT_DIARY_WRITING_PROMPT

    original_web_formatter = tasks._format_diary_web_chat_context
    original_activity_formatter = tasks._format_diary_activity_context
    try:
        tasks._format_diary_web_chat_context = (
            lambda provider: "### Diary Web Chat Preflight\n\nWeb user: 일기에서 이 수정 지시를 반영해"
        )
        tasks._format_diary_activity_context = (
            lambda provider: "### Diary Activity Preflight\n\nAnchor: scheduled test"
        )
        content = tasks._build_task_context_content(
            {"id": 900, "user_id": 0, "agent_type": "diary", "content": DEFAULT_DIARY_WRITING_PROMPT},
            DEFAULT_DIARY_WRITING_PROMPT,
            context_provider="deepseek",
        )
        assert "### Diary Activity Preflight" in content
        assert "### Diary Web Chat Preflight" in content
        assert "Web user: 일기에서 이 수정 지시를 반영해" in content
        assert content.index("### Diary Web Chat Preflight") < content.index("### Task")

        maintenance = tasks._build_task_context_content(
            {"id": 901, "user_id": 0, "agent_type": "diary", "content": "일기 335번 삭제해줘"},
            "일기 335번 삭제해줘",
            context_provider="deepseek",
        )
        assert "Diary Activity Preflight" not in maintenance
        assert "Diary Web Chat Preflight" not in maintenance

        non_diary = tasks._build_task_context_content(
            {"id": 902, "user_id": 0, "agent_type": "analyst", "content": "analyze"},
            "analyze",
            context_provider="deepseek",
        )
        assert "Diary Web Chat Preflight" not in non_diary
    finally:
        tasks._format_diary_web_chat_context = original_web_formatter
        tasks._format_diary_activity_context = original_activity_formatter


async def _assert_diary_runtime_route_fallbacks() -> None:
    # Keyword-heuristic fallback was removed 2026-07-11: with the LLM
    # classifier unavailable, route_task must return NO recommendation and
    # hand the orchestrator the curated routing cards instead of a
    # substring guess.
    import self_runtime.tools as tools

    original_classifier = tools._classify_route_with_llm
    try:
        async def unavailable_classifier(_task: str, _candidates=None):
            return None

        tools._classify_route_with_llm = unavailable_classifier
        result = json.loads(await tools._exec_route_task("일기 123번 오타 수정해줘", include_store_guide=False))
        rec = result["recommendation"]
        assert result["status"] == "ok"
        assert result["classifier"]["used"] is False
        assert rec["recommended_agent"] is None
        assert rec["source"] == "none"
        cards = rec.get("routing_cards") or {}
        assert "diary" in cards and "programmer" in cards and "analyst" in cards
        assert any("CommuLingo" in u for u in cards["analyst"].get("use_for", []))
    finally:
        tools._classify_route_with_llm = original_classifier


async def _assert_guarded_diary_save_handler_accepts_tool_payloads() -> None:
    import telegram.bot as bot
    import telegram.diary_publication as publication
    from telegram.diary_mode import DEFAULT_DIARY_WRITING_PROMPT
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

        handler = bot._make_guarded_diary_save_handler(
            None,
            {"id": 900, "agent_type": "diary", "content": DEFAULT_DIARY_WRITING_PROMPT},
        )
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

        blocked_handler = bot._make_guarded_diary_save_handler(
            None,
            {"id": 901, "agent_type": "diary", "content": "일기 335번 삭제해줘"},
        )
        blocked_result, blocked_is_error = await execute_tool(
            "save_diary",
            {"title": "제목", "content": "본문"},
            {"save_diary": blocked_handler},
            log_event=None,
        )
        assert blocked_is_error
        assert "allowed only for the configured scheduled diary-writing prompt" in blocked_result

        missing_title = await handler(content="본문")
        assert "missing required argument: title" in missing_title
    finally:
        bot._run_stasova_diary_review = original_review
        bot._apply_stasova_diary_review = original_apply
        publication.publish_reviewed_diary_entry = original_publish


async def _assert_diary_unpublish_action() -> None:
    import runtime_tools.post_edit as post_edit

    original_delete = post_edit._delete_diary_sync
    original_cache = post_edit._invalidate_cache_sync
    original_cf = post_edit._purge_cloudflare_sync
    try:
        post_edit._delete_diary_sync = lambda target: ({"id": target, "title": "비공개 대상"}, 2, 1)
        post_edit._invalidate_cache_sync = lambda kind, target: {"ok": True, "deleted": 3}
        post_edit._purge_cloudflare_sync = lambda kind, target: {"ok": True, "purged": 7, "urls": []}

        missing_confirm = await post_edit._exec_edit_content(
            content_type="diary",
            id=335,
            action="unpublish",
        )
        assert "requires confirm=true" in missing_confirm

        result = await post_edit._exec_edit_content(
            content_type="diary",
            id=335,
            action="unpublish",
            confirm=True,
        )
        assert "Unpublished diary id=335" in result
        assert "cleared 2 publication audit link" in result
        assert "public ai_diary storage" in result

        combined = await post_edit._exec_edit_content(
            content_type="diary",
            id=335,
            action="delete",
            confirm=True,
            title="새 제목",
        )
        assert "cannot be combined with edit fields" in combined
    finally:
        post_edit._delete_diary_sync = original_delete
        post_edit._invalidate_cache_sync = original_cache
        post_edit._purge_cloudflare_sync = original_cf


async def main() -> None:
    _assert_prompt_context()
    await _assert_runtime_profiles()
    _assert_writer_heavy_policy_uses_gateway_defaults()
    _assert_writer_kimi_catalog_and_resolution()
    _assert_openai_input_replay_checkpoint()
    _assert_inference_reasoning_policy()
    await _assert_openai_continuation_extends_round_limit()
    await _assert_kimi_preserves_reasoning_for_tool_replay()
    await _assert_kimi_content_filter_falls_back_in_forced_final()
    _assert_agent_runtime_config()
    _assert_autonomous_base_finalization_tools()
    _assert_agent_runtime_dynamic_reload()
    _assert_autonomous_prompt_surfaces_staged_drafts()
    _assert_staged_drafts_prioritize_project_events()
    await _assert_read_self_autonomous_project_uses_note_table()
    _assert_autonomous_project_selection_prioritizes_advice()
    await _assert_autonomous_tick_failure_updates_cooldown()
    _assert_collect_tick_actions_includes_publications()
    await _assert_successful_durable_tick_consumes_advisories()
    await _assert_successful_staged_draft_tick_consumes_advisories()
    await _assert_successful_noop_tick_logs_no_durable_action()
    _assert_autonomous_cli_status_uses_config_without_db()
    _assert_autonomous_cli_main_reports_missing_db_config()
    _assert_autonomous_cli_events_orders_by_id()
    _assert_autonomous_cli_list_includes_operational_signals()
    _assert_autonomous_cli_show_uses_note_table()
    _assert_orchestrator_autonomous_status_includes_operational_signals()
    await _assert_telegram_projects_lists_operational_signals()
    await _assert_telegram_project_show_includes_tick_signals()
    _assert_web_feedback_is_one_shot()
    _assert_web_political_line_dynamic_reload()
    await _assert_web_autonomous_summary_includes_publication_events()
    await _assert_web_public_summary_includes_autonomous_loop_state()
    await _assert_stage_public_records_autonomous_staged_draft_event()
    _assert_diary_web_context_injection()
    await _assert_diary_runtime_route_fallbacks()
    await _assert_guarded_diary_save_handler_accepts_tool_payloads()
    await _assert_diary_unpublish_action()
    print("runtime smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
