"""leninbot's CommuLingo writes go through the admin MCP: event/office edits,
suggestion review, event-link backfills, and following id renames."""
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commulingo import id_sync, people  # noqa: E402


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executed = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_conn(cursor):
    @contextmanager
    def get_conn():
        conn = MagicMock()
        conn.cursor.return_value = cursor
        conn.__enter__.return_value = conn
        yield conn
    return get_conn


class EventWritesThroughMcp(unittest.TestCase):
    def run_edit(self, result, pending=None, direct=True):
        cursor = FakeCursor([pending] if pending else [])
        with patch.object(people, "get_conn", fake_conn(cursor)), \
                patch.object(people, "_validate", return_value=None), \
                patch.object(people, "direct_apply_enabled", return_value=direct), \
                patch.object(people, "call_person_service", return_value=result) as rpc:
            message = people._run_edit("history_event_person", "create", "civil-war",
                                       {"personId": "trotsky", "relationKind": "leader"}, ["https://src"], None)
        return message, rpc

    def test_direct_apply_submits_to_the_store(self):
        message, rpc = self.run_edit({"status": "approved", "suggestionId": "41", "summary": "linked person 'trotsky'"})
        request = rpc.call_args.args[0]
        self.assertEqual(request["command"], "submit")
        self.assertEqual(request["target"], "history_event_person")
        self.assertEqual(request["id"], "civil-war")
        self.assertTrue(request["directApply"])
        self.assertEqual(request["sources"], ["https://src"])
        self.assertIn("OK — applied: linked person 'trotsky'. Logged as edit #41.", message)
        self.assertIn("cyber-lenin.com/commulingo/events/civil-war", message)

    def test_staging_reports_the_pending_suggestion(self):
        message, rpc = self.run_edit({"status": "pending", "suggestionId": "42"}, pending={"id": 40}, direct=False)
        self.assertFalse(rpc.call_args.args[0]["directApply"])
        self.assertIn("staged as suggestion #42", message)
        self.assertIn("suggestion #40", message)

    def test_store_rejection_is_returned_to_the_model(self):
        cursor = FakeCursor()
        with patch.object(people, "get_conn", fake_conn(cursor)), \
                patch.object(people, "_validate", return_value=None), \
                patch.object(people, "call_person_service", side_effect=ValueError("400: side must be one of reds")):
            message = people._run_edit("history_event_person", "create", "civil-war", {}, [], None)
        self.assertEqual(message, "Error: 400: side must be one of reds")

    def test_curator_policy_runs_first(self):
        with patch.object(people, "get_conn", fake_conn(FakeCursor())), \
                patch.object(people, "_validate", return_value="Error: too long"), \
                patch.object(people, "call_person_service") as rpc:
            self.assertEqual(people._run_edit("history_event", "update", "x", {}, [], None), "Error: too long")
        rpc.assert_not_called()

    def test_no_direct_sql_writer_is_left(self):
        for name in ("apply_edit", "_record_suggestion", "_apply_event_section", "_apply_office_row_create"):
            self.assertFalse(hasattr(people, name), name)


class FollowRenames(unittest.TestCase):
    def test_open_rows_follow_each_redirect(self):
        pages = {"person": {"total": 1, "items": [{"from_id": "old-id", "to_id": "new-id"}]},
                 "term": {"total": 0, "items": []}}
        cursor = FakeCursor()
        with patch.object(id_sync, "call_tool", side_effect=lambda name, args: pages[args["entityType"]]), \
                patch.object(id_sync, "get_conn", fake_conn(cursor)):
            moved = id_sync.follow_renames()
        self.assertEqual(moved, {"commulingo_curation_gaps": 1, "commulingo_pipeline_jobs": 1})
        sqls = [sql for sql, _ in cursor.executed]
        self.assertTrue(sqls[0].startswith("UPDATE commulingo_curation_gaps SET target_id"))
        self.assertIn("SET status = 'cancelled'", sqls[1], "a duplicate active job is cancelled before the move")
        self.assertTrue(sqls[2].startswith("UPDATE commulingo_pipeline_jobs SET target"))
        self.assertEqual(cursor.executed[0][1], ("new-id", "person", "old-id"))

    def test_outage_does_not_stop_the_worker(self):
        with patch.object(id_sync, "call_tool", side_effect=RuntimeError("down")):
            self.assertIsNone(id_sync.follow_renames_best_effort())


class SuggestionReview(unittest.TestCase):
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "commulingo_suggestions", Path(__file__).resolve().parents[1] / "scripts" / "commulingo_suggestions.py")
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)

    def test_review_uses_the_row_target(self):
        row = {"id": "7", "status": "pending", "target_type": "office_row", "reviewer": ""}
        with patch.object(self.cli, "call_tool", return_value={"items": [row], "total": 1}), \
                patch.object(self.cli, "call_person_service", return_value={"status": "approved"}) as rpc:
            self.assertEqual(self.cli.cmd_review(7, True, "checked"), 0)
        self.assertEqual(rpc.call_args.args[0], {"command": "review", "target": "office_row", "suggestionId": "7",
                                                 "approve": True, "note": "checked", "changedBy": "agent-suggestion:7"})

    def test_already_reviewed_is_not_resent(self):
        row = {"id": "7", "status": "approved", "target_type": "person", "reviewer": "x"}
        with patch.object(self.cli, "call_tool", return_value={"items": [row], "total": 1}), \
                patch.object(self.cli, "call_person_service") as rpc:
            self.assertEqual(self.cli.cmd_review(7, True, "checked"), 1)
        rpc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
