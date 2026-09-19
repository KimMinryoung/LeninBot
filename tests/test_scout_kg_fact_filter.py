"""Scout→KG fact filter: bookkeeping lines never reach the graph, outages keep everything."""
import unittest
from unittest.mock import patch

from kg_runtime import scout_ingest
from llm.call_registry import CallSiteProfile, Decision

PROFILE = CallSiteProfile(feature='scout_kg_fact_filter', provider='openrouter', model='typesafe/jev-1.13',
                          extra={'thresholds': {'keep': 0.8}})
REPORT = ('## Findings\n'
          '- INBOX 최고 UID는 345로 변동 없음 — 새 메일은 없다.\n'
          '- 스탠퍼드·Arc Institute 연구진이 언어모델로 바이러스 16종을 설계해 Science에 게재.\n'
          '- 저장(원본 .md): UID 326, 329, 330 3통만 저장.\n')


def decision(scores):
    return Decision(answers={f'line_{i + 1}': {'type': 'noul', 'noul': p} for i, p in enumerate(scores)},
                    model='typesafe/jev-1.13-test')


class FactFilterTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(scout_ingest, 'add_kg_episode', return_value={'status': 'ok', 'message': 'stored'})
        self.write = p.start(); self.addCleanup(p.stop)
        c = patch.object(scout_ingest, '_classify_group_id', return_value='economy')
        self.classify = c.start(); self.addCleanup(c.stop)
        r = patch('llm.call_registry.resolve', return_value=PROFILE)
        r.start(); self.addCleanup(r.stop)

    def test_only_fact_lines_are_written_and_classified(self):
        with patch('llm.call_registry.decide_sync', return_value=decision([0.02, 0.95, 0.01])) as decide:
            result = scout_ingest.process_scout_report_to_kg(REPORT, task_content='메일함 확인', task_id=5)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['facts_count'], 1)
        self.assertEqual(result['fact_filter']['kept'], 1)
        content = self.write.call_args.kwargs['content']
        self.assertIn('바이러스 16종', content)
        self.assertNotIn('UID', content)
        # One decision per report with a noul per line; the group is classified over the kept facts only.
        state, questions = decide.call_args.args[1], decide.call_args.args[2]
        self.assertEqual(set(questions), {'line_1', 'line_2', 'line_3'})
        self.assertEqual(state['lines']['line_2'][:6], '스탠퍼드·A')
        self.assertNotIn('UID', self.classify.call_args.args[1])

    def test_all_bookkeeping_report_writes_no_episode(self):
        with patch('llm.call_registry.decide_sync', return_value=decision([0.02, 0.1, 0.01])):
            result = scout_ingest.process_scout_report_to_kg(REPORT, task_content='메일함 확인', task_id=5)
        self.assertEqual(result['status'], 'skip')
        self.assertIn('fact filter', result['message'])
        self.write.assert_not_called()
        self.classify.assert_not_called()

    def test_unavailable_filter_keeps_every_line(self):
        with patch('llm.call_registry.decide_sync', return_value=None):
            result = scout_ingest.process_scout_report_to_kg(REPORT, task_content='메일함 확인', task_id=5)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['facts_count'], 3)
        self.assertTrue(result['fact_filter']['unavailable'])

    def test_disabled_filter_makes_no_call(self):
        with patch('llm.call_registry.resolve', return_value=CallSiteProfile(
                feature='scout_kg_fact_filter', provider='openrouter', model='m', extra={'enabled': False})), \
             patch('llm.call_registry.decide_sync') as decide:
            result = scout_ingest.process_scout_report_to_kg(REPORT, task_content='메일함 확인', task_id=5)
        decide.assert_not_called()
        self.assertEqual(result['facts_count'], 3)


if __name__ == '__main__':
    unittest.main()
