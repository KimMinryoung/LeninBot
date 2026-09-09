import unittest
from unittest.mock import patch

from scripts import commulingo_lane_health as health


class LaneHealth(unittest.TestCase):
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
