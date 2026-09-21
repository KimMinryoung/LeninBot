import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.eval_jev_citations import main, summarize


def row(support='supports', confidence=.9, expected='supports', reject=False):
    return {'answers': {'support': {'choice': support, 'confidence': confidence},
                        'boilerplate': {'noul': 0}, 'specific': {'noul': .9}},
            'model': 'fixture-only', 'latency_ms': 100, 'cost_usd': .0001,
            'expected_support': expected, 'expected_reject': reject}


class EvaluationTests(unittest.TestCase):
    def test_uncertainty_and_outage_are_not_counted_as_correct_confident_answers(self):
        records = [row(), row('supports', .4, 'contradicts', True),
                   {'answers': None}, row('unrelated', .99, 'supports')]
        result = summarize(records, {'reject': .85, 'boilerplate': .9})
        self.assertEqual(result['unavailable'], 1)
        self.assertEqual(result['confident_coverage'], .5)
        self.assertEqual(result['confident_support_accuracy'], .5)
        self.assertEqual(result['false_rejections'], 1)
        self.assertEqual(result['missed_rejections_available_only'], 1)
        self.assertAlmostEqual(result['reported_or_estimated_cost_usd'], .0003)

    def test_replay_uses_production_dispute_policy_and_threshold(self):
        records = [{**row('contradicts', .9, 'contradicts'), 'stance': 'disputes'},
                   row('unrelated', .9, 'unrelated', True)]
        loose = summarize(records, {'reject': .85, 'boilerplate': .9})
        strict = summarize(records, {'reject': .95, 'boilerplate': .9})
        self.assertEqual(loose['false_rejections'], 0)
        self.assertEqual(loose['missed_rejections_available_only'], 0)
        self.assertEqual(strict['missed_rejections_available_only'], 1)
        self.assertIsNone(strict['confident_support_accuracy'])

    def test_empty_evaluation_does_not_claim_perfect_accuracy(self):
        result = summarize([], {'reject': .85, 'boilerplate': .9})
        self.assertIsNone(result['confident_support_accuracy'])
        self.assertIsNone(result['reported_or_estimated_cost_usd'])

    def test_replay_cli_never_calls_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'saved.json'
            path.write_text(json.dumps({'records': [row()]}))
            output = io.StringIO()
            with patch('sys.argv', ['eval', '--replay', str(path)]), \
                 patch('scripts.eval_jev_citations.decide_detailed') as call, \
                 redirect_stdout(output):
                self.assertEqual(main(), 0)
            call.assert_not_called()
            report = json.loads(output.getvalue())
            self.assertEqual(report['mode'], 'replay')
            self.assertEqual(len(report['threshold_sweep']), 5)
            self.assertEqual(report['summary']['models'], ['fixture-only'])


if __name__ == '__main__':
    unittest.main()
