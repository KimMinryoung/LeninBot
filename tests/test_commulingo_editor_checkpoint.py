from copy import deepcopy
import unittest

from commulingo.pipeline.author_draft import obj
from commulingo.pipeline.editor_checkpoint import restore_draft


class EditorCheckpointTests(unittest.TestCase):
    def test_historical_section_restores_without_rewriting_journal(self):
        schema = obj({'body': obj({'ko': {'type': 'string'}, 'en': {'type': 'string'}})})
        snapshot = {'tool': 'commulingo_pipeline_result', 'args': {
            'fields': {'slug': 'old-slug', 'sortOrder': 193707, 'notes': 'private note',
                       'body': {'ko': ['첫 문장.', '다음 문장.'], 'en': ['First.', 'Next.']}},
            'notes': 'private note',
            'claims': [{'field': 'sortOrder', 'passages': ['P1']},
                       {'field': 'body', 'passages': ['P2']}], 'issue_results': []}}
        original = deepcopy(snapshot)
        restored = restore_draft(snapshot, schema, section=True)
        self.assertEqual(snapshot, original)
        self.assertEqual(restored['args']['fields'], {
            'startYear': 1937, 'startMonth': 7,
            'body': {'ko': '첫 문장. 다음 문장.', 'en': 'First. Next.'}})
        self.assertEqual(restored['args']['notes'], 'private note')
        self.assertEqual(restored['args']['claims'], [{'field': 'body', 'passages': ['P2']}])
        self.assertEqual(restore_draft(restored, schema, section=True), restored)

    def test_malformed_checkpoint_is_retained_for_explicit_replacement(self):
        snapshot = {'args': {'fields': ['bad'], 'claims': 'bad'}}
        restored = restore_draft(snapshot, obj({}))
        self.assertEqual(restored, snapshot)
        restored['args']['fields'].append('another')
        self.assertEqual(snapshot['args']['fields'], ['bad'])
