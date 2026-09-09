import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from scripts import commulingo_lane_health as health


class LaneHealth(unittest.TestCase):
    def test_pipeline_cost_does_not_add_shared_legacy_spend_twice(self):
        value={'applied':2,'escalated':0,'retrying':0,'running':1,'expired_leases':0,
               'pipeline_cost':.2,'today_actual':.8,'today_reserved':.2}
        with patch.object(health.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(value))):
            lines,alerts,cost=health.pipeline_health('-24h')
        self.assertEqual(cost,.2)
        self.assertEqual(alerts,[])
        self.assertIn('spent $0.8000',lines[1])

    def test_pending_submissions_are_not_exhausted_runs(self):
        journal = '''{
  "status": "no_edit",
  "result": "OK — pending: create person 'test'",
  "rounds": 5
}
{
  "status": "pending_review",
  "rounds": 7
}'''
        with patch.object(health, "journal", return_value=journal):
            stats = health.tally("gap", "-24h")
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["pending_review"], 2)
        self.assertEqual(stats["no_edit"], 0)
        self.assertEqual(health.problems("gap", stats), [])

    def test_actual_no_edit_still_alerts_without_inventing_cause(self):
        with patch.object(health, "journal", return_value='  "status": "no_edit"'):
            stats = health.tally("gap", "-24h")
        alerts = health.problems("gap", stats)
        self.assertTrue(any("ended with no edit" in alert for alert in alerts))
        self.assertFalse(any("rounds exhausted" in alert for alert in alerts))

    def test_completed_topic_is_successful_work_without_an_edit(self):
        with patch.object(health,'journal',return_value='  "status": "not_applicable",\n  "cost_usd": 0.02'):
            stats=health.tally('enrich','-24h')
        self.assertEqual(stats['completed'],1)
        self.assertEqual(stats['total'],1)
        self.assertEqual(stats['cost'],.02)
        self.assertEqual(health.problems('enrich',stats),[])
