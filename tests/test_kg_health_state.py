"""Recorded KG healthcheck: MCP-visible state, Telegram only once after 3 days."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from kg_runtime import health_state as hs

T0 = datetime(2026, 10, 17, tzinfo=timezone.utc)


def failing(*reasons):
    return {"ok": not reasons, "failures": list(reasons)}


class HealthStateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(hs, "STATE_DIR", Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_escalates_once_after_three_days_of_one_component(self):
        self.assertEqual(hs.record(failing("sync.documents: 1 document(s) failed: a"), now=T0), [])
        later = T0 + timedelta(days=2, hours=23)
        self.assertEqual(hs.record(failing("sync.documents: unresolved source reconciliation"), now=later), [])
        sent = hs.record(failing("sync.documents: 2 document(s) failed: b"), now=T0 + timedelta(days=3))
        self.assertEqual([i["key"] for i in sent], ["sync.documents"])
        self.assertIn("since 2026-10-17T00:00", hs.format_escalation(sent))
        self.assertEqual(hs.record(failing("sync.documents: x"), now=T0 + timedelta(days=4)), [])

    def test_resolution_restarts_the_clock_and_logs_history(self):
        hs.record(failing("smoke_search: degraded"), now=T0)
        hs.record(failing(), now=T0 + timedelta(days=1))
        hs.record(failing("smoke_search: degraded"), now=T0 + timedelta(days=2))
        self.assertEqual(hs.record(failing("smoke_search: degraded"), now=T0 + timedelta(days=4)), [])
        events = (hs.STATE_DIR / "history.jsonl").read_text().splitlines()
        self.assertEqual(len(events), 3)

    def test_summary_reports_open_failures_and_staleness(self):
        self.assertFalse(hs.summary()["recorded"])
        hs.record(failing("semantic_search: degraded"), now=T0)
        s = hs.summary(now=T0 + timedelta(hours=5))
        self.assertTrue(s["stale"])
        self.assertEqual(s["open_failures"]["semantic_search"]["open_hours"], 5.0)
