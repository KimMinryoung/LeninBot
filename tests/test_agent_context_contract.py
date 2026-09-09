"""Hermetic regression checks for task handoff, trust boundaries and verdicts."""

import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import telegram.tasks as tasks
from prompt_context import format_subtask_results


class TaskContextTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.recall = Mock(return_value="")
        self.kg = Mock(return_value="")
        self.stack.enter_context(patch.dict(sys.modules, {
            "memory_store.experiential": SimpleNamespace(recall_experiences_block=self.recall),
            "kg_runtime.recall": SimpleNamespace(entity_gated_kg_block=self.kg),
            "redis_state": SimpleNamespace(
                format_board_for_context=lambda *a, **kw: "",
                format_task_chain_for_context=lambda *a, **kw: "parent evidence",
            ),
            "telegram.mission": SimpleNamespace(
                get_mission_events=lambda *a, **kw: [], add_mission_event=lambda *a: None,
            ),
        }))
        self.stack.enter_context(patch("db.query", return_value=[]))
        self.stack.enter_context(patch.object(tasks, "build_current_state", return_value=""))
        self.query = self.stack.enter_context(patch.object(tasks, "_query", return_value=[]))
        self.previous = self.stack.enter_context(patch.object(tasks, "_query_one", return_value=None))
        self.task = {"id": 10, "user_id": 7, "agent_type": "analyst"}

    def test_dependencies_do_not_replace_recall_query_or_task(self):
        self.task["metadata"] = {"depends_on_task_ids": [1, 2]}
        self.query.return_value = [{"id": 1, "agent_type": "scout", "status": "done", "result": "old-topic " * 900}]
        for provider in ("claude", "deepseek"):
            with self.subTest(provider=provider):
                built = tasks._build_task_context_content(self.task, "Investigate the current target", context_provider=provider)
                self.recall.assert_called_with("Investigate the current target", provider, 3)
                self.kg.assert_called_with("Investigate the current target", provider)
                self.assertIn("missing", built)
                self.assertIn("truncated:", built)
                self.assertIn("read_self(content_type='task_report', id=1)", built)
                task_heading = "<task>" if provider == "claude" else "### Task\n"
                self.assertNotIn("old-topic", built.split(task_heading)[-1])

    def test_synthesis_includes_failure_and_verification(self):
        self.task.update(plan_role="synthesis", plan_id=3)
        self.query.return_value = [
            {"id": 1, "agent_type": "scout", "status": "done", "result": "Useful output", "verification_status": "passed"},
            {"id": 2, "agent_type": "analyst", "status": "failed", "result": "source unavailable", "verification_status": "failed", "verification_details": "goal blocked"},
            {"id": 3, "agent_type": "programmer", "status": "handed_off"},
        ]
        built = tasks._build_task_context_content(self.task, "Combine all requirements")
        self.assertIn("source unavailable", built)
        self.assertIn("goal blocked", built)
        self.assertIn("handed_off", built)
        sql = self.query.call_args.args[0]
        self.assertNotIn("status = 'done'", sql)
        self.assertIn("verification_details", sql)
        self.recall.assert_called_with("Combine all requirements", "claude", 3)

    def test_lookup_failure_is_visible(self):
        self.task.update(plan_role="synthesis", plan_id=3, metadata={"depends_on_task_ids": [1]})
        self.query.side_effect = RuntimeError("offline")
        built = tasks._build_task_context_content(self.task, "Finish task")
        self.assertIn("Subtask lookup failed", built)
        self.assertIn("Dependency lookup failed", built)
        self.assertIn("<task>\nFinish task\n</task>", built)

    def test_no_cross_mission_history(self):
        tasks._build_task_context_content(self.task, "New target")
        self.previous.assert_not_called()
        self.task["mission_id"] = 55
        tasks._build_task_context_content(self.task, "Same mission")
        sql, params = self.previous.call_args.args
        self.assertIn("mission_id = %s", sql)
        self.assertEqual(params[-1], 55)

    def test_parent_history_takes_priority(self):
        self.task.update(mission_id=55, parent_task_id=9)
        built = tasks._build_task_context_content(self.task, "Resume")
        self.assertIn("parent evidence", built)
        self.previous.assert_not_called()

    def test_missing_status_does_not_imply_success(self):
        for provider in ("claude", "deepseek"):
            result = format_subtask_results([{"id": 1}], provider)
            self.assertIn("unknown", result)
            self.assertIn("unverified", result)
            self.assertIn("No result available", result)


def verdict(goal="complete", execution="appropriate", retry="no", verdict="PASS"):
    return f"VERDICT: {verdict}\nReason: inspected source and stored result\nExecution: {execution}\nGoal: {goal}\nRetry: {retry}"


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    async def verify(self, response=None, error=None, tools=None, handlers=None, report="Result with evidence"):
        self.chat = AsyncMock(return_value=response, side_effect=error)
        self.execute = Mock()
        with patch.object(tasks, "_execute", self.execute):
            return await tasks._run_verification(
                None, {"id": 1, "user_id": 0, "agent_type": "analyst", "content": "Check source"}, report,
                chat_with_tools_fn=self.chat, get_model_fn=AsyncMock(return_value="stub"),
                extra_tools=tools, extra_handlers=handlers,
            )

    async def test_complete_evidence_passes_and_persists_axes(self):
        result = await self.verify(verdict())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["goal"], "complete")
        self.assertIn('"execution": "appropriate"', result["details"])
        self.assertEqual(self.execute.call_args.args[1][0], "passed")

    async def test_external_block_is_not_completion_or_auto_retry(self):
        result = await self.verify(verdict(goal="blocked", retry="conditional"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["execution"], "appropriate")
        with patch.object(tasks, "_query_one") as query:
            retry = await tasks._maybe_redelegate_after_verification_failure(None, {"id": 1}, result)
            query.assert_not_called()
            self.assertEqual(retry["status"], "blocked")

    async def test_provider_error_is_unverified_without_task_retry(self):
        result = await self.verify(error=RuntimeError("provider unavailable"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["goal"], "unverified")
        self.assertEqual(result["retry"], "conditional")

    async def test_malformed_or_conflicting_response_never_passes(self):
        for response in ("VERDICT: PASS", verdict() + "\nVERDICT: FAIL", verdict().replace("Goal: complete", "Goal: unknown")):
            with self.subTest(response=response):
                result = await self.verify(response)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["goal"], "unverified")

    async def test_empty_report_fails_without_llm(self):
        result = await self.verify(verdict(), report="")
        self.chat.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["retry"], "yes")

    async def test_restart_instructions_follow_actual_tool_surface(self):
        await self.verify(verdict())
        prompt = self.chat.call_args.args[0][0]["content"]
        self.assertIn("No restart tool is available", prompt)
        self.assertNotIn("restart_service is available:", prompt)
        await self.verify(verdict(), tools=[{"name": "restart_service"}], handlers={"restart_service": Mock()})
        prompt = self.chat.call_args.args[0][0]["content"]
        self.assertIn("restart_service is available:", prompt)
        self.assertNotIn("telegram_bot.py", prompt)


class PromptBoundaryTests(unittest.TestCase):
    def test_all_registered_agents_receive_source_boundary_once(self):
        from agents import list_agents
        from identity.prompts import EXTERNAL_SOURCE_RULE

        for spec in list_agents():
            for provider in ("claude", "deepseek"):
                with self.subTest(agent=spec.name, provider=provider):
                    prompt = spec.render_prompt(provider=provider)
                    self.assertEqual(prompt.count(EXTERNAL_SOURCE_RULE), 1)
                    self.assertIn("partial success never implies overall completion", prompt)


class HandoffRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_detail_recovers_full_request_and_verification(self):
        from self_runtime.tools import _exec_read_task_reports

        request = "requirement " * 300 + "FINAL_REQUIREMENT"
        row = {"id": 7, "status": "done", "content": request, "result": "report",
               "verification_status": "failed", "verification_details": "goal blocked"}
        with patch("db.query_one", return_value=row):
            result = await _exec_read_task_reports(task_id=7)
        self.assertIn("FINAL_REQUIREMENT", result)
        self.assertIn("goal blocked", result)

    async def test_inline_report_retains_evidence_and_blockers_at_tail(self):
        from self_runtime.tools import build_run_agent_handler

        full_report = "finding " * 700 + "SOURCE_AND_BLOCKER_AT_END"
        chat = AsyncMock(return_value=full_report)
        result = await build_run_agent_handler(chat)("analyst", "Analyze", "Source context")
        self.assertEqual(result, full_report)
        prompt = chat.call_args.args[0][0]["content"]
        self.assertIn("no durable task history", prompt)


if __name__ == "__main__":
    unittest.main()
