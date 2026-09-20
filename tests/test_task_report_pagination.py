import unittest
from unittest.mock import patch

from self_runtime.tools import _exec_read_self


class TaskReportPaginationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.log = "도구 결과\n" * 3000 + "SUCCESS_RECEIPT_AT_END"
        self.row = {"id": 7, "status": "done", "content": "original request",
                    "result": "final report", "tool_log": self.log}
        self.enterContext(patch("db.query_one", return_value=self.row))

    async def test_default_preview_marks_truncation_and_recovery(self):
        result = await _exec_read_self(content_type="task_report", id=7)
        self.assertIn("original request", result)
        self.assertIn("final report", result)
        self.assertIn("truncated=True", result)
        self.assertIn("field='tool_log', offset=5000", result)
        self.assertNotIn("SUCCESS_RECEIPT_AT_END", result)

    async def test_all_log_pages_reconstruct_exact_evidence(self):
        pages = []
        for offset in range(0, len(self.log), 5000):
            result = await _exec_read_self(content_type="task_report", id=7, field="tool_log", offset=offset)
            header, body = result.split("\n\n", 1)
            self.assertIn(f"chars={len(self.log)}", header)
            if offset + 5000 < len(self.log):
                self.assertIn(f"field='tool_log', offset={offset + 5000}", header)
            else:
                self.assertIn("truncated=False", header)
                self.assertNotIn("next:", header)
            pages.append(body)
        self.assertEqual("".join(pages), self.log)

    async def test_selected_fields_do_not_repeat_other_content(self):
        for field in ("content", "result"):
            result = await _exec_read_self(content_type="task_report", id=7, field=field, max_chars=4, offset=2)
            self.assertEqual(result.split("\n\n", 1)[1], self.row[field][2:6])
            self.assertNotIn("## Tool Log", result)

    async def test_bounds_empty_and_missing_fields(self):
        self.row["tool_log"] = "x" * 25000
        result = await _exec_read_self(content_type="task_report", id=7, field="tool_log", max_chars=100000)
        self.assertEqual(len(result.split("\n\n", 1)[1]), 20000)
        result = await _exec_read_self(content_type="task_report", id=7, field="tool_log", offset=30000)
        self.assertIn("returned_chars=25000:25000 truncated=False", result)
        self.row["tool_log"] = None
        result = await _exec_read_self(content_type="task_report", id=7, field="tool_log")
        self.assertIn("chars=0 returned_chars=0:0 truncated=False", result)
        self.assertTrue((await _exec_read_self(content_type="task_report", field="tool_log")).startswith("Error:"))
        self.assertTrue((await _exec_read_self(content_type="task_report", id=7, field="wrong")).startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
