"""Visitor non-publication requests: parsed, read once per message, retried on failure.

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
STANDING = [dp.PrivacyRequest(id=5, visitor="v", chat_log_id=3, requested_at="2026-10-01 10:00 KST",
                              quote="이름 빼라", withheld=["방문자 이름"])]


class ParseReplyTests(unittest.TestCase):
    def test_keeps_requests_and_revocations_tied_to_this_chunk(self):
        reply = json.dumps({
            "requests": [
                {"message_id": 11, "quote": "개인조건도 일기에 넣지 말길", "withheld": ["방문자의 노동 조건", " "]},
                {"message_id": 99, "quote": "invented", "withheld": ["x"]},
                {"message_id": 10, "quote": "no items", "withheld": []},
            ],
            "revoked": [{"request_id": 5, "message_id": 10}, {"request_id": 6, "message_id": 10},
                        {"request_id": 5, "message_id": 99}],
        }, ensure_ascii=False)
        found, revoked = dp.parse_reply(reply, MESSAGES, STANDING)
        self.assertEqual([r["chat_log_id"] for r in found], [11])
        self.assertEqual(found[0]["withheld"], ["방문자의 노동 조건"])
        self.assertEqual(revoked, [(5, 10)])

    def test_empty_is_clean_and_garbage_is_a_failure(self):
        self.assertEqual(dp.parse_reply('{"requests": []}', MESSAGES, []), ([], []))
        self.assertIsNone(dp.parse_reply("not json", MESSAGES, []))
        self.assertIsNone(dp.parse_reply('{"other": 1}', MESSAGES, []))


class ChunkTests(unittest.TestCase):
    def test_chunks_never_split_a_message(self):
        messages = [{"id": i, "user_query": "x" * n} for i, n in enumerate([40, 40, 100, 10])]
        chunks = dp.chunk_messages(messages, limit=90)
        self.assertEqual([[m["id"] for m in c] for c in chunks], [[0, 1], [2], [3]])


class ScanVisitorTests(unittest.TestCase):
    def _run(self, replies, messages):
        calls, stored = [], []

        def fake_generate(feature, prompt, *, system=None, **kw):
            calls.append((feature, prompt))
            reply = replies[len(calls) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

        with patch.object(dp, "_new_messages", return_value=messages), \
                patch.object(dp, "_standing_requests", return_value=STANDING), \
                patch.object(dp, "_store", side_effect=lambda *a: stored.append(a)), \
                patch.object(dp.call_registry, "generate_sync", side_effect=fake_generate), \
                patch.object(dp, "CHUNK_CHARS", 20):
            ok = dp.scan_visitor("v")
        return ok, calls, stored

    def test_watermark_advances_per_successful_chunk_and_stops_at_failure(self):
        ok_reply = json.dumps({"requests": [{"message_id": 10, "quote": "q", "withheld": ["조직명"]}]},
                              ensure_ascii=False)
        ok, calls, stored = self._run([ok_reply, RuntimeError("timeout")], MESSAGES)
        self.assertFalse(ok)
        self.assertEqual(calls[0][0], dp.FEATURE)
        self.assertIn("[request_id=5]", calls[0][1])
        self.assertIn("[message_id=10]", calls[0][1])
        self.assertEqual(len(stored), 1)  # chunk 2 failed: watermark stays at 10
        visitor, found, revoked, last_id = stored[0]
        self.assertEqual((visitor, last_id, revoked), ("v", 10, []))
        self.assertEqual(found[0]["withheld"], ["조직명"])

    def test_nothing_new_makes_no_call(self):
        ok, calls, stored = self._run([], [])
        self.assertTrue(ok)
        self.assertEqual((calls, stored), ([], []))


class ScanTests(unittest.TestCase):
    def test_failed_visitor_is_reported_and_standing_requests_rendered(self):
        with patch.object(dp, "_active_visitors", return_value=["a", "b"]), \
                patch.object(dp, "scan_visitor", side_effect=[True, False]), \
                patch.object(dp, "_standing_requests", return_value=STANDING):
            result = dp.scan(72)
        self.assertEqual(result.failed_visitors, ["b"])
        text = dp.checklist(result)
        self.assertIn("- 방문자 이름", text)
        self.assertIn("자동 검토에 실패", text)
        self.assertIn("binding", dp.format_diary_block(result))

    def test_clean_scan_renders_nothing(self):
        with patch.object(dp, "_active_visitors", return_value=["a"]), \
                patch.object(dp, "scan_visitor", return_value=True), \
                patch.object(dp, "_standing_requests", return_value=[]):
            result = dp.scan(72)
        self.assertEqual(dp.format_diary_block(result), "")


if __name__ == "__main__":
    unittest.main()
