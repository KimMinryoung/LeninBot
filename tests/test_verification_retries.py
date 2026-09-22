"""Verification retries use a chain budget, never free-form restart prose."""
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from telegram import tasks


def task(task_id=1, parent=None, content="Fix the report", limit=1, metadata=None):
    return {"id": task_id, "parent_task_id": parent, "agent_type": "analyst",
            "user_id": 42, "status": "done", "content": content, "result": "Partial result",
            "metadata": metadata if metadata is not None else {"verification": {"retry_limit": limit}}}


def failure(**overrides):
    return {"status": "failed", "execution": "error", "goal": "partial", "retry": "yes",
            "restart": "none", "details": "Repairable failure", **overrides}


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rows = {1: task()}
        self.query = self.enterContext(patch.object(tasks, "_query_one", side_effect=lambda sql, args: self.rows.get(args[0])))
        self.execute = self.enterContext(patch.object(tasks, "_execute"))
        self.created = []

        def create(content, user_id, priority, **kwargs):
            task_id = max(self.rows) + 1
            row = task(task_id, kwargs.get("parent_task_id"), content, metadata=kwargs.get("metadata"))
            row.update(user_id=user_id, status=kwargs.get("status", "pending"))
            if kwargs.get("restart_state"):
                row["metadata"]["restart_state"] = kwargs["restart_state"]
            self.rows[task_id] = row
            self.created.append(row)
            return {"status": "ok", "task_id": task_id}

        self.create = self.enterContext(patch("task_store.create_task_in_db", side_effect=create))
        self.restart = self.enterContext(patch("runtime_tools.registry._exec_restart_service", new_callable=AsyncMock,
                                              return_value="✅ leninbot-telegram: restarted"))
        self.persist_restart = self.enterContext(patch.object(tasks, "persist_task_restart_state"))

    async def run_retry(self, row, verdict=None):
        return await tasks._maybe_redelegate_after_verification_failure(None, row, verdict or failure())

    async def test_exact_chain_limits(self):
        for limit in (0, 1, 3):
            with self.subTest(limit=limit):
                self.rows = {1: task(limit=limit)}
                self.created.clear()
                row = self.rows[1]
                for index in range(limit):
                    result = await self.run_retry(row)
                    self.assertEqual(result["status"], "redelegated")
                    row = self.rows[result["task_id"]]
                    self.assertEqual(row["user_id"], 42)
                    self.assertEqual(row["metadata"]["verification_retry"], {"root_task_id": 1, "used": index + 1})
                    self.assertEqual(row["content"].count("[AUTO-RETRY"), 1)
                self.assertEqual((await self.run_retry(row))["status"], "limit_reached")
                self.assertEqual(len(self.created), limit)
        self.restart.assert_not_awaited()

    async def test_blocked_unverified_and_missing_axes_never_act(self):
        for verdict in (failure(goal="blocked"), failure(goal="unverified"),
                        failure(execution="appropriate"), failure(retry=None),
                        failure(restart="maybe"), {"status": "failed"}):
            result = await self.run_retry(self.rows[1], verdict)
            self.assertEqual(result["status"], "blocked")
        self.query.assert_not_called()
        self.create.assert_not_called()
        self.restart.assert_not_awaited()

    async def test_legacy_current_retry_counts_and_handoff_does_not(self):
        self.rows[2] = task(2, 1, "[AUTO-RETRY after verification failure for task #1]\nOriginal task:\nFix")
        self.assertEqual((await self.run_retry(self.rows[2]))["status"], "limit_reached")
        self.rows[2]["status"] = "handed_off"
        self.rows[3] = task(3, 2, tasks._RESTART_COMPLETED_MARKER + "\n" + self.rows[2]["content"])
        state = await tasks._verification_retry_state(self.rows[3])
        self.assertEqual(state, {"root_task_id": 1, "used": 1})

    async def test_retry_after_legacy_handoff_keeps_original_root(self):
        self.rows[1]["status"] = "handed_off"
        self.rows[2] = task(2, 1, tasks._RESTART_COMPLETED_MARKER + "\nFix the report")
        self.rows[3] = task(3, 2, "[AUTO-RETRY after verification failure for task #2]\nOriginal task:\nFix")
        self.assertEqual(await tasks._verification_retry_state(self.rows[3]), {"root_task_id": 1, "used": 1})

    async def test_missing_cyclic_and_invalid_history_fail_closed(self):
        marker = "[AUTO-RETRY after verification failure for task #2]\nFix"
        self.rows[1] = task(1, 2, marker)
        self.assertEqual((await self.run_retry(self.rows[1]))["status"], "blocked")
        self.rows[2] = task(2, 1, marker)
        self.assertEqual((await self.run_retry(self.rows[1]))["status"], "blocked")
        self.rows[1] = task(metadata={"verification_retry": {"root_task_id": 1, "used": -1}})
        self.assertEqual((await self.run_retry(self.rows[1]))["status"], "blocked")
        self.create.assert_not_called()

    async def test_restart_prose_cannot_trigger_restart(self):
        for details in ("No telegram restart required", "telegram service restart required"):
            result = await self.run_retry(self.rows[1], failure(details=details))
            self.assertEqual(result["status"], "redelegated")
        self.restart.assert_not_awaited()

    async def test_restart_uses_same_budget_and_records_real_success(self):
        result = await self.run_retry(self.rows[1], failure(restart="telegram"))
        self.assertEqual(result["status"], "restart_initiated")
        child = self.created[0]
        self.assertEqual(child["status"], "queued")
        self.assertFalse(child["metadata"]["restart_state"]["restart_completed"])
        self.assertNotIn(tasks._RESTART_COMPLETED_MARKER, child["content"])
        self.assertEqual(child["metadata"]["verification_retry"]["used"], 1)
        self.persist_restart.assert_called_once()
        self.assertTrue(self.persist_restart.call_args.kwargs["mark_completed"])
        self.assertEqual((await self.run_retry(child, failure(restart="telegram")))["status"], "limit_reached")
        self.restart.assert_awaited_once()

    async def test_restart_zero_budget_and_already_requested_do_not_act(self):
        self.rows[1] = task(limit=0)
        self.assertEqual((await self.run_retry(self.rows[1], failure(restart="telegram")))["status"], "limit_reached")
        self.rows[1] = task(limit=3)
        self.rows[1]["metadata"]["restart_state"] = {"restart_initiated": True, "restart_completed": False}
        self.assertEqual((await self.run_retry(self.rows[1], failure(restart="telegram")))["status"], "post_restart_verification_failed")
        self.create.assert_not_called()
        self.restart.assert_not_awaited()

    async def test_child_creation_failure_never_restarts(self):
        self.create.side_effect = None
        self.create.return_value = {"status": "error", "error": "DB rejected insert"}
        result = await self.run_retry(self.rows[1], failure(restart="telegram"))
        self.assertEqual(result["status"], "error")
        self.restart.assert_not_awaited()

    async def test_restart_failure_cancels_waiting_child_without_completion_claim(self):
        for response in ("❌ Restart blocked — syntax errors", RuntimeError("restart failed"), "⏱ timeout"):
            self.restart.side_effect = response if isinstance(response, Exception) else None
            self.restart.return_value = response
            result = await self.run_retry(self.rows[1], failure(restart="telegram"))
            self.assertEqual(result["status"], "restart_failed")
            self.assertIn("status = 'failed'", self.execute.call_args.args[0])
        self.persist_restart.assert_not_called()

    async def test_startup_handoff_preserves_policy_and_attempt(self):
        row = task(metadata={"verification": {"retry_limit": 3},
                             "verification_retry": {"root_task_id": 10, "used": 2},
                             "restart_state": {"restart_initiated": True, "restart_completed": False,
                                               "restart_target_service": "telegram"}})
        row.update(created_at=datetime.now(timezone.utc), depth=1, scratchpad="")
        query = Mock(side_effect=[[row], [{"id": 2}]])
        with patch.object(tasks, "_query", query), patch("redis_state.get_task_progress", return_value=[]), \
                patch("redis_state.save_task_summary"), patch("redis_state.clear_task_progress"):
            result = await tasks.recover_processing_tasks_on_startup()
        self.assertEqual(result["handed_off"], 1)
        self.assertIn("metadata", query.call_args_list[0].args[0])
        saved = json.loads(query.call_args_list[1].args[1][7])
        self.assertEqual(saved["verification_retry"], row["metadata"]["verification_retry"])
        self.assertEqual(saved["verification"], {"retry_limit": 3})
        self.assertTrue(saved["restart_state"]["restart_completed"])

    async def test_telegram_startup_does_not_confirm_an_api_restart(self):
        row = task(metadata={"restart_state": {"restart_initiated": True,
                                               "restart_completed": False,
                                               "restart_target_service": "api"}})
        row.update(created_at=datetime.now(timezone.utc), depth=1, scratchpad="")
        query = Mock(side_effect=[[row], [{"id": 2}]])
        with patch.object(tasks, "_query", query), patch("redis_state.get_task_progress", return_value=[]), \
                patch("redis_state.save_task_summary"), patch("redis_state.clear_task_progress"):
            await tasks.recover_processing_tasks_on_startup()
        saved = json.loads(query.call_args_list[1].args[1][7])
        self.assertFalse(saved["restart_state"]["restart_completed"])


class VerdictTests(unittest.TestCase):
    def test_restart_field_is_optional_but_unambiguous(self):
        base = "VERDICT: FAIL\nReason: checked files\nExecution: error\nGoal: partial\nRetry: yes"
        self.assertEqual(tasks._parse_verification_response(base)["restart"], "none")
        self.assertEqual(tasks._parse_verification_response(base + "\nRestart: telegram")["restart"], "telegram")
        for suffix in ("\nRestart: maybe", "\nRestart: telegram\nRestart: none", "\nRestart:"):
            result = tasks._parse_verification_response(base + suffix)
            self.assertEqual(result["goal"], "unverified")
            self.assertEqual(result["restart"], "none")


if __name__ == "__main__":
    unittest.main()
