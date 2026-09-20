import unittest
import json
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts import commulingo_lane_health as health


class LaneHealth(unittest.TestCase):
    def test_pipeline_only_digest_does_not_read_retired_lane_journals(self):
        for phase in ('draft', 'live'):
            output = io.StringIO()
            with self.subTest(phase=phase), \
                 patch('sys.argv', ['commulingo_lane_health.py']), \
                 patch.object(health.Path, 'read_text', return_value=json.dumps(
                     {'phase': phase, 'legacy_shared_budget': False})), \
                 patch.object(health, 'journal') as journal, \
                 patch.object(health, 'pipeline_health', return_value=(['pipeline healthy'], [], .2)) as pipeline, \
                 patch.object(health, 'tool_rejections', return_value=([], [])), \
                 patch.object(health, 'bio_length_drift', return_value=([], [])), \
                 patch.object(health, 'execution_metrics', return_value=[]), \
                 redirect_stdout(output):
                self.assertEqual(health.main(), 0)
            pipeline.assert_called_once_with('-24h')
            journal.assert_not_called()
            self.assertIn('pipeline healthy', output.getvalue())
            self.assertIn('$0.2000', output.getvalue())
            self.assertNotIn('PROBLEMS:', output.getvalue())

    def test_query_json_keeps_multiline_aggregate(self):
        value=[{'tool_name':'a'},{'tool_name':'b'}]
        with patch.object(health.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(value,indent=2))):
            self.assertEqual(health.query_json('SELECT fixture'),value)
    def test_tool_report_shows_causes_without_claiming_retry_cost(self):
        rows = [{'tool_name':'commulingo_pipeline_result','result_status':'rejected',
                 'calls':30,'scopes':3,'reason':'explicit gap must match'},
                {'tool_name':'commulingo_pipeline_result','result_status':'ok',
                 'calls':10,'scopes':8,'reason':''}]
        with patch.object(health,'query_json',return_value=rows):
            lines,alerts=health.tool_rejections('-24h')
        self.assertIn('30/40',lines[0])
        self.assertIn('explicit gap must match',lines[1])
        self.assertIn('3개 실행 범위',lines[1])
        self.assertNotIn('paid rounds',str(alerts))

    def test_long_notification_is_not_truncated(self):
        text='보고서 🐶\n'*2000
        chunks=list(health.notification_chunks(text))
        self.assertEqual(''.join(chunks),text)
        self.assertTrue(all(len(c.encode('utf-16-le'))//2<=3500 for c in chunks))

    def test_report_separates_window_results_from_current_queue(self):
        value={'applied':1,'escalated':2,'retrying':0,'running':0,'expired_leases':0,
               'pipeline_cost':.2,'today_actual':.8,'today_reserved':0,
               'publications':[{'job_id':8,'kind':'person','action':'update','target':'example',
                                'label':'인물','topic':'basics'}]}
        with patch.object(health.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(value))), patch.object(health,'query_json',return_value=[]):
            lines,alerts,cost=health.pipeline_health('-24h')
        self.assertIn('기간 내 반영 1건',lines[0])
        self.assertTrue(any('반영 #8' in line for line in lines))
        self.assertTrue(any('누적 대기열' in a for a in alerts))

    def test_pipeline_cost_does_not_add_shared_legacy_spend_twice(self):
        value={'applied':2,'escalated':0,'retrying':0,'running':1,'expired_leases':0,
               'pipeline_cost':.2,'today_actual':.8,'today_reserved':.2}
        with patch.object(health.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(value))), patch.object(health,'query_json',return_value=[]):
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
