"""Only real diary artifacts may satisfy a scheduled writing task."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from telegram import tasks, diary_publication
from telegram.diary_mode import DEFAULT_DIARY_WRITING_PROMPT
from publishing import review_artifacts


def diary_task():
    return {"id": 42, "user_id": 0, "agent_type": "diary",
            "content": DEFAULT_DIARY_WRITING_PROMPT}


class ReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_publication_passes_without_a_tool_log_or_model_verifier(self):
        receipt = {"diary_id": 541, "title": "실제 저장된 일기"}
        with patch.object(tasks, "_query_one", return_value=receipt), \
                patch.object(tasks, "_execute") as execute:
            verifier = AsyncMock()
            result = await tasks._run_verification(None, diary_task(), "<analysis>bad report</analysis>",
                chat_with_tools_fn=verifier)
            self.assertEqual((result["status"], result["goal"]), ("passed", "complete"))
            self.assertIn("#541", result["details"])
            self.assertIs(tasks._warn_if_diary_unsaved(diary_task(), ""), receipt)
            verifier.assert_not_called()
            execute.assert_called_once()

    async def test_tool_name_in_log_without_artifact_is_not_a_receipt(self):
        with patch.object(tasks, "_query_one", return_value=None), \
                patch.object(tasks, "_execute"):
            result = await tasks._run_verification(None, diary_task(), "save_diary(title='fake')")
            self.assertEqual((result["status"], result["goal"]), ("failed", "unverified"))
            self.assertEqual(result["retry"], "no")
            with self.assertLogs(tasks.logger, level="WARNING"):
                self.assertIsNone(tasks._warn_if_diary_unsaved(diary_task(), "save_diary(fake)"))

    async def test_audit_insert_failure_can_use_verified_handler_receipt(self):
        row = {"diary_id": 541, "title": "persisted"}
        task = diary_task()
        task["_published_diary_id"] = 541
        with patch.object(tasks, "_query_one", side_effect=[None, row]) as query:
            self.assertEqual(tasks._diary_publication_receipt(task), row)
            self.assertEqual(query.call_args.args[1], (541,))

    async def test_unsaved_summary_never_persists_done(self):
        with patch.object(tasks, "_query_one", return_value=None), \
                patch.object(tasks, "_execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "without an observed persisted publication"):
                await tasks._persist_task_success(task=diary_task(), content="task", report="<analysis>summary",
                    budget_tracker={}, mission_id=None, restart_ctx={"initiated": False})
            execute.assert_not_called()

    async def test_saved_diary_report_is_derived_from_actual_row(self):
        with patch.object(tasks, "_query_one", return_value={"diary_id": 541, "title": "actual"}), \
                patch.object(tasks, "_execute") as execute, \
                patch("memory_store.redis_state.save_task_summary"):
            result = await tasks._persist_task_success(task=diary_task(), content="task", report="<analysis>summary",
                budget_tracker={}, mission_id=None, restart_ctx={"initiated": False})
        self.assertIn("/541", result)
        self.assertNotIn("analysis", result)
        self.assertEqual(execute.call_args.args[1][0], result)

    async def test_skipped_verification_does_not_pass(self):
        with patch.object(tasks, "_execute") as execute:
            result = await tasks._run_verification(None, {"id": 1, "agent_type": "visualizer"}, "result")
        self.assertEqual((result["status"], result["goal"]), ("skipped", "unverified"))
        self.assertEqual(execute.call_args.args[1][0], "skipped")

    async def test_no_warning_publication_still_records_receipt_without_ddl(self):
        with patch.object(diary_publication, "publish_diary_entry", new=AsyncMock(return_value=(541, ""))), \
                patch.object(diary_publication, "record_diary_publication_audit", new=AsyncMock(return_value=10)) as audit, \
                patch.object(diary_publication, "ensure_diary_publication_audit_table") as ddl:
            result = await diary_publication.publish_reviewed_diary_entry(
                task_id=42, title="title", content="draft", final_title="title", final_content="final",
                review_report="위험도 총평: 낮음\n경고 항목: 없음")
        self.assertEqual(result, (541, "", 10))
        self.assertFalse(audit.call_args.kwargs["warning_detected"])
        ddl.assert_not_called()


class ArtifactTests(unittest.TestCase):
    def test_runtime_creates_report_directory_and_saves_exact_final_report(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(review_artifacts, "PROJECT_ROOT", Path(directory)):
            relative = "temp_dev/stasova_reviews/diary_task_42.md"
            review_artifacts.save_review_report(relative, "실제 최종 보고서")
            self.assertEqual((Path(directory) / relative).read_text(), "실제 최종 보고서\n")
            self.assertEqual(len(list((Path(directory) / relative).parent.iterdir())), 1)
            with self.assertRaises(ValueError):
                review_artifacts.save_review_report("outside.md", "report")


if __name__ == "__main__":
    unittest.main()
