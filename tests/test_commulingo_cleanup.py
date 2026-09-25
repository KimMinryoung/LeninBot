from contextlib import contextmanager
from copy import deepcopy
from unittest import TestCase
from unittest.mock import Mock

from commulingo.pipeline.cleanup import automatic, has_work, retire
from commulingo.pipeline.engine import disposition, Result

JOB = {'id': 1, 'kind': 'term', 'action': 'update', 'target': 'fixture',
       'topic': 'enrichment', 'reason': 'Bundled enrichment: definition, history',
       'payload': {'topics': ['definition', 'history'], 'gap_ids': []}}
FULL = {'definition': {'ko': '정의', 'en': 'Definition'}, 'body': {'ko': '본문', 'en': 'Body'}}


class CleanupTests(TestCase):
    def test_all_remaining_topics_and_explicit_commissions(self):
        self.assertFalse(has_work(JOB, FULL))
        self.assertTrue(has_work(JOB, {**FULL, 'body': {'ko': '본문', 'en': ''}}))
        person = {**JOB, 'kind': 'person', 'payload': {'topics': ['bio','sections']}}
        self.assertTrue(has_work(person, {'bio': FULL['body'], 'sections':False}))
        self.assertFalse(has_work(person, {'bio': FULL['body'], 'sections':True}))
        self.assertTrue(automatic(JOB))
        for key, value in [('gap_ids',[2]), ('gap_id',2), ('review_feedback',{'decision':'revise'}),
                           ('original_proposal',{'id':3})]:
            self.assertFalse(automatic({**JOB, 'payload': {key:value}}))
        nested = {**JOB, 'payload': {'commissions': [{'reason':'Please add a trial section', 'payload':{}}]}}
        self.assertFalse(automatic(nested))

    def store(self, rows):
        cur = Mock()
        cur.fetchall.return_value = rows
        store = Mock()
        @contextmanager
        def transaction():
            yield cur
        store.transaction = transaction
        return store, cur

    def test_preview_is_read_only_and_apply_preserves_needed_jobs(self):
        missing = {**deepcopy(JOB), 'id':2, 'current':{**FULL, 'body':{}}}
        explicit = {**deepcopy(JOB), 'id':3, 'current':FULL,
                    'payload':{'commissions':[{'reason':'Requested change'}]}}
        store, cur = self.store([{**JOB, 'current': FULL}, missing, explicit])
        preview = retire(store)
        self.assertEqual([r['id'] for r in preview['candidates']], [1])
        self.assertEqual(preview['retired'], 0)
        self.assertEqual(cur.execute.call_count, 1)
        self.assertTrue(cur.execute.call_args.args[0].startswith('SELECT'))
        self.assertIn("'sections',EXISTS(SELECT 1 FROM commulingo_person_sections", cur.execute.call_args.args[0])
        cur.reset_mock()
        result = retire(store, apply=True)
        self.assertEqual(result['retired'], 1)
        calls = cur.execute.call_args_list
        self.assertIn('NOWAIT', calls[0].args[0])
        self.assertEqual(calls[-1].args[1][1], [1])
        self.assertEqual(calls[-2].args[1][0], [1,2,3])

    def test_batch_bounds(self):
        for limit in (0,201):
            with self.assertRaises(ValueError):
                retire(Mock(), limit=limit)

    def test_publication_is_not_inferred_from_complete(self):
        self.assertEqual(disposition('judge',Result({},'complete','complete')), 'no_edit')
        self.assertEqual(disposition('submit',Result({'status':'approved'},'research')), 'published')
        self.assertEqual(disposition('research',Result({},'complete','complete')), 'progress')
        self.assertEqual(disposition('review',Result({},'complete','escalated')), 'held')
