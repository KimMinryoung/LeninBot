"""Hermetic #4 policy regressions; never call an LLM, DB, or publishing service."""
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobs import practice_output as p
from jobs import autonomous_project as ap


@pytest.mark.parametrize("text", [
    "다음 경제지표 발표일 확인", "CPI 발표를 기다림", "금리 공표 대기",
    "Check the next CPI release date", "Await the next GDP announcement",
    "CPI 발표 후 재개", "발표일 확인 후 교육자료 작성",
])
def test_wait_filter(text):
    assert p.validate_plan([], [text]).startswith("error:")


@pytest.mark.parametrize("text", [
    "2026-09-15 임금 통계 비교표 작성", "이미 발표된 CPI로 교육자료 작성",
    "필수 자료 부재를 기록하고 기존 노트로 검증 요청에 답변", "운영자 시작 승인 대기",
])
def test_dates_and_actionable_work_allowed(text):
    assert p.validate_plan([], [{"task": text}]) is None


def test_serialized_restart_counts_failures_without_false_completion():
    state = None
    phases = []
    for index in range(4):
        # Serialization simulates process restart, independent of turn_count.
        state = p.reserve(json.loads(json.dumps(state)), str(index))
        phases.append(state["attempts"][-1]["phase"])
        state = p.finish(state, {}, error="provider failed" if index in (1, 3) else None)
    assert phases == ["a", "b", "b", "c"]
    assert state["closed"] and state["outcome"] == "publication_unconfirmed"
    assert state["attempts"][1]["status"] == "failed"
    assert p.reserve(state, "next")["output_id"] != state["output_id"]
    assert p.reserve(state, "next")["attempts"][0]["phase"] == "a"


def test_interrupted_attempt_is_charged():
    state = p.reserve(None, "first")
    recovered = p.reserve(json.loads(json.dumps(state)), "second")
    assert recovered["attempts"][0]["status"] == "interrupted"
    assert recovered["attempts"][1]["phase"] == "b"


def test_draft_review_and_publication_are_distinct():
    state = None
    for i in range(4):
        state = p.reserve(state, str(i))
        if i < 3:
            state = p.finish(state, {})
    evidence = {"research_draft_staged": [{"event_id": 1}], "publication_reviewed": [{"event_id": 2}]}
    assert p.finish(state, evidence)["outcome"] == "publication_unconfirmed"
    evidence["publication_created"] = [{"event_id": 3}]
    result = p.finish(state, evidence, error="crash after publish")
    assert result["outcome"] == "published"
    assert result["attempts"][-1]["status"] == "failed"
    assert result["attempts"][-1]["evidence"] == evidence
    assert all(v["value"] is None and v["status"] == "unknown" for v in p.value_metrics().values())


@pytest.mark.asyncio
async def test_revise_plan_blocks_before_write_and_preserves_other_projects():
    for project_id in (4, 3):
        with patch.object(ap, "db_query_one", return_value={"plan": {}}), \
             patch.object(ap, "db_execute") as write, patch.object(ap, "_log_event"):
            _, handlers = ap._build_project_tools(project_id)
            result = await handlers["revise_plan"](rationale="test", steps=["다음 경제지표 발표일 확인"])
            assert result.startswith("error:" if project_id == 4 else "ok:")
            assert write.call_count == (0 if project_id == 4 else 1)


@pytest.mark.asyncio
async def test_publication_phase_and_failure_cap_and_existing_asset_protection():
    state = p.reserve(None, "a")
    state["descriptor"] = {"topic": "임금", "audience": "실무자", "purpose": "검증", "form": "표"}
    handler = AsyncMock(return_value="error: fact check failed")
    handlers = {"research_document": handler, "edit_content": handler, "set_project_state": handler}
    guarded = p.guard_handlers(handlers, state)
    assert "phase c" in await guarded["research_document"](action="publish_public", slug=state["output_id"])
    assert "only current artifact" in await guarded["research_document"](action="edit_staged", slug="project-3-report")
    assert "cannot edit" in await guarded["edit_content"](slug="project-3-report")
    assert "cannot edit" in await guarded["set_project_state"](state="researching", reason="resume")
    handler.assert_not_awaited()
    for i in range(3):
        state = p.reserve(p.finish(state, {}), str(i))
    guarded = p.guard_handlers(handlers, state)
    assert "fact check failed" in await guarded["research_document"](action="publish_public", slug=state["output_id"])
    assert "cap reached" in await guarded["research_document"](action="publish_public", slug=state["output_id"])
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_direct_call_never_reserves_or_executes():
    with patch("bot_config.is_autonomous_active", return_value=False), \
         patch.object(p, "begin") as begin, patch.object(ap, "_execute_one_tick", new_callable=AsyncMock) as execute:
        result = await ap._run_one_tick({"id": 4, "state": "researching"})
    assert result["skipped"] == "autonomous_active=false"
    begin.assert_not_called()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_paused_db_state_overrides_stale_active_snapshot():
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (True,)
    @contextmanager
    def connection():
        yield conn
    with patch("db.get_conn", connection), patch("bot_config.is_autonomous_active", return_value=True), \
         patch.object(ap, "db_query_one", return_value={"id": 4, "state": "paused"}), \
         patch.object(p, "begin") as begin, patch.object(ap, "_execute_one_tick", new_callable=AsyncMock) as execute:
        result = await ap._run_one_tick({"id": 4, "state": "researching"})
    assert result["skipped"] == "inactive"
    begin.assert_not_called()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_is_finalized_and_lock_released():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (True,)
    @contextmanager
    def connection():
        yield conn
    with patch("db.get_conn", connection), patch("bot_config.is_autonomous_active", return_value=True), \
         patch.object(ap, "db_query_one", return_value={"id": 4, "state": "researching"}), \
         patch.object(p, "begin", return_value=p.reserve(None, "test")), \
         patch.object(p, "complete") as complete, \
         patch.object(ap, "_execute_one_tick", new_callable=AsyncMock, side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            await ap._run_one_tick({"id": 4, "state": "researching"})
    assert "failed" in complete.call_args.kwargs["error"]
    assert "pg_advisory_unlock" in cursor.execute.call_args.args[0]


def test_crashed_publication_reconciles_before_next_reservation():
    state = p.reserve(None, "lost-process")
    calls = []
    with patch.object(p, "query_one", return_value={"meta": state}), \
         patch.object(p, "complete", side_effect=lambda *a, **k: calls.append("reconcile")), \
         patch.object(p, "_mutate", side_effect=lambda *a: calls.append("reserve")):
        p.begin("new-process")
    assert calls == ["reconcile", "reserve"]


def test_state_write_failure_is_not_swallowed():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [("researching",), None]
    def execute(sql, params=None):
        if sql.startswith("INSERT"):
            raise RuntimeError("storage unavailable")
    cursor.execute.side_effect = execute
    @contextmanager
    def connection():
        yield conn
    with patch.object(p, "get_conn", connection):
        with pytest.raises(RuntimeError, match="storage unavailable"):
            p._mutate(lambda previous, cur: p.reserve(previous, "test"))


@pytest.mark.parametrize("lifecycle", ["paused", "archived"])
def test_inactive_project_preserves_reserved_publication_but_rejects_new_work(lifecycle):
    state = None
    for index in range(4):
        state = p.reserve(state, str(index))
        if index < 3:
            state = p.finish(state, {})
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [(lifecycle,), (state,), (10,)]
    cursor.fetchall.return_value = [
        (11, "publication_created", {"filename": state["output_id"] + ".md"})
    ]

    @contextmanager
    def connection():
        yield conn

    with patch.object(p, "get_conn", connection):
        result = p.complete("3")
        assert result["closed"] and result["outcome"] == "published"
        assert result["attempts"][-1]["evidence"]["publication_created"][0]["event_id"] == 11
        assert not any("UPDATE autonomous_projects" in call.args[0]
                       for call in cursor.execute.call_args_list)
        cursor.reset_mock()
        cursor.fetchone.side_effect = [(lifecycle,)]
        with pytest.raises(RuntimeError, match="active project"):
            p._mutate(lambda previous, cur: p.reserve(previous, "new"))
        assert not any(call.args[0].startswith("INSERT")
                       for call in cursor.execute.call_args_list)


@pytest.mark.asyncio
async def test_tick_integration_disables_repeat_review_and_passes_hard_limits(monkeypatch):
    import sys
    import types
    from llm.runtime_profile import RuntimeProfile
    import runtime_tools.registry as registry
    import agents
    import llm.runtime_profile as profiles

    state = p.reserve(None, "integration")
    chat = AsyncMock(return_value="model output is not value evidence")
    monkeypatch.setitem(sys.modules, "telegram.bot", types.SimpleNamespace(_chat_with_tools=chat))
    spec = MagicMock()
    spec.effective_provider.return_value = "deepseek"
    spec.filter_tools.return_value = ([], {})
    spec.render_prompt.return_value = "base system"
    spec.max_rounds = 50
    spec.budget_usd = 2.0
    spec.model = "low"
    spec.finalization_tools = []
    spec.terminal_tools = []
    monkeypatch.setattr(agents, "get_agent", lambda name: spec)
    monkeypatch.setattr(profiles, "resolve_runtime_profile", AsyncMock(return_value=RuntimeProfile(
        "autonomous", "deepseek", "xml", "low", "test", "test", "test", 50, 1024, 2.0, True)))
    monkeypatch.setattr(ap, "_build_project_tools", lambda pid: ([], {}))
    monkeypatch.setattr(ap, "_build_task_prompt", lambda *a, **k: "snapshot")
    monkeypatch.setattr(ap, "_fetch_pending_advisories", lambda pid: [])
    monkeypatch.setattr(ap, "_collect_tick_actions", lambda *a: {})
    monkeypatch.setattr(ap, "db_execute", MagicMock())
    monkeypatch.setattr(ap, "_log_event", MagicMock())
    monkeypatch.setattr(ap, "_notify_telegram", AsyncMock())
    forbidden = {}
    for name in ("_plan_tick_objective", "_review_tick_outcome", "_diagnose_staged_drafts_for_tick", "_build_deep_dive_tool"):
        forbidden[name] = MagicMock(side_effect=AssertionError(name + " must not run"))
        monkeypatch.setattr(ap, name, forbidden[name])
    await ap._execute_one_tick({"id": 4, "state": "researching", "goal": "산출물 생산"}, production=state)
    chat.assert_awaited_once()
    arguments = chat.call_args.kwargs
    assert arguments["budget_usd"] == p.TICK_BUDGET
    assert arguments["max_rounds"] == p.TICK_ROUNDS
    assert arguments["request_id"] == "integration"
    assert "define_practice_output" in arguments["extra_handlers"]
    assert "research_deep_dive" not in arguments["extra_handlers"]
    assert "모델 자기 채점" in arguments["system_prompt"]
    for mocked in forbidden.values():
        mocked.assert_not_called()


def test_publication_evidence_requires_current_output_identity():
    assert p.matches_output({"filename": "p4-test.md"}, "p4-test")
    assert p.matches_output({"public_url": "https://cyber-lenin.com/research/p4-test"}, "p4-test")
    assert not p.matches_output({"public_url": "https://cyber-lenin.com/research/project-3"}, "p4-test")
    assert not p.matches_output({"content": "published p4-test"}, "p4-test")


@pytest.mark.asyncio
async def test_descriptor_required_immutable_and_cannot_reopen_same_topic():
    state = p.reserve(None, "descriptor")
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    saved = {}
    def mutate(change):
        result = change(json.loads(json.dumps(state)), cursor)
        saved.update(result)
        return result
    with patch.object(p, "_mutate", side_effect=mutate):
        _, handlers = p.build_tools(state)
        define = handlers["define_practice_output"]
        assert "required" in await define(topic="임금")
        descriptor = dict(topic="임금", audience="실무자", purpose="검증", form="비교표")
        assert (await define(**descriptor)).startswith("ok:")
        assert saved["descriptor"] == descriptor
        assert "immutable" in await define(**{**descriptor, "topic": "다른 주제"})
        cursor.fetchone.return_value = (1,)
        assert "already attempted" in await define(**descriptor)


def test_local_descriptor_tool_has_state_risk_class():
    from security_gateway.policy import risk_class
    assert risk_class("define_practice_output") == "state"


@pytest.mark.asyncio
async def test_concurrent_tick_is_skipped_without_reservation():
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (False,)
    @contextmanager
    def connection():
        yield conn
    with patch("db.get_conn", connection), patch("bot_config.is_autonomous_active", return_value=True), \
         patch.object(p, "begin") as begin, patch.object(ap, "_execute_one_tick", new_callable=AsyncMock) as execute:
        result = await ap._run_one_tick({"id": 4, "state": "researching"})
    assert result["skipped"] == "already running"
    begin.assert_not_called()
    execute.assert_not_awaited()


def test_scheduler_handles_pause_race_without_success_log():
    with patch("bot_config.is_autonomous_active", return_value=True), \
         patch.object(ap, "_pick_next_project", return_value={"id": 4, "title": "test", "state": "researching"}), \
         patch.object(ap, "_run_one_tick", new_callable=AsyncMock,
                      return_value={"project_id": 4, "skipped": "inactive"}):
        assert ap.run_tick()["skipped"] == "inactive"
