"""Visitor non-publication requests are extracted, kept per task and rendered as a checklist.

Run from repo root:  venv/bin/python -m unittest tests.test_diary_privacy -v
"""
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from telegram import diary_privacy as dp

MESSAGES = [
    {"id": 10, "created_at": datetime(2026, 10, 13, 4, 13, tzinfo=timezone.utc),
     "user_query": "나는 컴앞 노동자다. 시간이 많지 않다."},
    {"id": 11, "created_at": datetime(2026, 10, 13, 7, 35, tzinfo=timezone.utc),
     "user_query": "조직명도 개인조건도 일기에 넣지 말길"},
]


class ParseRequestsTests(unittest.TestCase):
    def test_keeps_requests_tied_to_the_visitors_messages(self):
        reply = json.dumps({"requests": [
            {"message_id": 11, "quote": "개인조건도 일기에 넣지 말길", "withheld": ["방문자의 노동 조건", " "]},
            {"message_id": 99, "quote": "invented", "withheld": ["x"]},
            {"message_id": 10, "quote": "no items", "withheld": []},
        ]}, ensure_ascii=False)
        found = dp.parse_requests(reply, "fab6ba6e", MESSAGES)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].chat_log_id, 11)
        self.assertEqual(found[0].withheld, ["방문자의 노동 조건"])
        self.assertIn("KST", found[0].requested_at)

    def test_empty_list_is_a_clean_scan_and_garbage_is_a_failure(self):
        self.assertEqual(dp.parse_requests('{"requests": []}', "v", MESSAGES), [])
        self.assertIsNone(dp.parse_requests("not json", "v", MESSAGES))
        self.assertIsNone(dp.parse_requests('{"other": 1}', "v", MESSAGES))


class ScanTests(unittest.TestCase):
    def setUp(self):
        dp._SCANS.clear()

    def _scan(self, replies):
        calls = []

        def fake_generate(feature, prompt, *, system=None, **kw):
            calls.append((feature, prompt))
            reply = replies[len(calls) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

        with patch.object(dp, "_active_visitors", return_value=["visitor-a", "visitor-b"]), \
                patch.object(dp, "_visitor_messages", return_value=MESSAGES), \
                patch.object(dp.call_registry, "generate_sync", side_effect=fake_generate):
            result = dp.scan_for_task(7, 72)
            again = dp.scan_for_task(7, 72)
        return result, again, calls

    def test_failed_visitor_is_reported_not_treated_as_clean(self):
        ok = json.dumps({"requests": [{"message_id": 11, "quote": "넣지 말길", "withheld": ["조직명"]}]},
                        ensure_ascii=False)
        result, again, calls = self._scan([ok, RuntimeError("timeout")])
        self.assertIs(result, again)  # cached for the save handler
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], dp.FEATURE)
        self.assertIn("[message_id=11]", calls[0][1])
        self.assertEqual([r.visitor for r in result.requests], ["visitor-a"])
        self.assertEqual(result.failed_visitors, ["visitor-b"])
        text = dp.checklist(result)
        self.assertIn("- 조직명", text)
        self.assertIn("자동 검토에 실패", text)
        self.assertIn("binding", dp.format_diary_block(result))

    def test_no_requests_renders_nothing(self):
        result, _, _ = self._scan(['{"requests": []}', '{"requests": []}'])
        self.assertEqual(dp.checklist(result), "")
        self.assertEqual(dp.format_diary_block(result), "")


if __name__ == "__main__":
    unittest.main()
