import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_person_activities as audit
from commulingo.activities import load_catalog
from llm.call_registry import Decision, DecisionResult


class AuditTests(unittest.TestCase):
    def person(self, sourced=True):
        return {'id': 'example', 'name': 'Example', 'bio': 'Directed land reform.',
                'primaryActivity': {'functionId': 'government'},
                'activities': [{'functionId': 'government', 'primary': True,
                                'evidence': [{'source': 'https://example.org', 'locator': 'Career',
                                              'claim': 'Land reform', 'excerpt': 'Directed land reform.'}] if sourced else []}]}

    def decide(self, basis='0', confidence=.96):
        def call(feature, state, questions, label=None):
            self.assertNotIn('primaryActivity', state)
            self.assertNotIn('activities', state)
            key = next(iter(questions))
            choice = 'agriculture' if key == 'activity_function' else basis
            return DecisionResult(decision=Decision(answers={key: {'choice': choice, 'confidence': confidence}}, model='test'))
        return call

    def test_sourced_candidate_and_no_source_candidate_are_distinct(self):
        for sourced, status, calls in [(True, 'source_supported_candidate', 2), (False, 'needs_source', 1)]:
            row = audit.judge(self.person(sourced), load_catalog(), .85, self.decide())
            self.assertEqual(row['status'], status)
            self.assertEqual(row['calls'], calls)
            self.assertTrue(row['changed'])
            self.assertEqual('evidence' in row, sourced)

    def test_unsupported_low_confidence_and_provider_failure(self):
        self.assertEqual(audit.judge(self.person(), load_catalog(), .85, self.decide('unsupported'))['status'], 'unsupported')
        self.assertEqual(audit.judge(self.person(), load_catalog(), .85, self.decide(confidence=.4))['status'], 'low_confidence')
        row = audit.judge(self.person(), load_catalog(), .85, lambda *a, **kw: DecisionResult(error='offline'))
        self.assertEqual(row['status'], 'error')

    def test_checkpoint_resume_and_changed_input_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.jsonl'
            row = audit.judge(self.person(), load_catalog(), .85, self.decide())
            path.write_text(json.dumps(row) + '\n\n{"incomplete":')
            self.assertEqual(len(audit.checkpoints(path)), 1)
            source = Path(directory) / 'people.json'
            source.write_text(json.dumps({'people': [self.person()]}))
            with patch.object(audit, 'judge', side_effect=AssertionError('should reuse checkpoint')):
                self.assertEqual(audit.main(['--input', str(source), '--out', str(path)]), 0)
            self.assertEqual(len(audit.checkpoints(path)), 1)
            changed = self.person()
            changed['bio'] = 'New career evidence'
            self.assertNotEqual(audit.prepare(changed, load_catalog(), .85)[2]['fingerprint'], row['fingerprint'])
            self.assertNotEqual(audit.prepare(self.person(), load_catalog(), .7)[2]['fingerprint'], row['fingerprint'])


if __name__ == '__main__':
    unittest.main()
