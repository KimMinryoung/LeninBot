"""Current author schemas and write normalization, without live storage.

Person, section and term validation lives in the CommuLingo JS store.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch
from commulingo_test_support import FakeReads, no_external_io

from commulingo.pipeline.decisions import Decisions
from commulingo.pipeline.engine import Usage
from commulingo.pipeline.patches import schema_for
from commulingo import people


class WriteContractTests(unittest.TestCase):
    def setUp(self):
        reads = patch.object(people, '_reads', FakeReads())
        reads.start()
        self.addCleanup(reads.stop)

    def test_term_author_omits_category_but_storage_requires_it(self):
        tool = people.COMMULINGO_TERM_CREATE_TOOL['input_schema']['properties']['fields']
        self.assertNotIn('category', tool['properties'])
        self.assertNotIn('category', tool['required'])
        self.assertIn('period', tool['required'])
        job = {'kind': 'term', 'action': 'create', 'topic': 'basics'}
        storage = schema_for(job, None)
        self.assertIn('category', storage['required'])
        author = Decisions(job, None, None, Usage()).author_schema(storage)
        self.assertNotIn('category', author['properties'])

    def test_term_period_is_the_structured_career_shape(self):
        tool = people.COMMULINGO_TERM_CREATE_TOOL['input_schema']['properties']['fields']
        self.assertTrue({'startYear', 'endYear'}.isdisjoint(tool['properties']))
        self.assertIn('periodNote', tool['properties'])
        self.assertEqual(tool['properties']['period']['type'], ['object', 'null'])

    def test_person_author_provides_labels_and_runner_assigns_codes(self):
        fields = people.COMMULINGO_PERSON_CREATE_TOOL['input_schema']['properties']['fields']
        self.assertTrue({'citizenship', 'nationalOrigin', 'evidence'} <= set(fields['required']))
        self.assertTrue({'groupId', 'role'}.isdisjoint(fields['properties']))
        for name, code in (('citizenship', 'code'), ('nationalOrigin', 'code'), ('fate', 'kind')):
            self.assertNotIn(code, fields['properties'][name]['properties'])
            self.assertIn('label', fields['properties'][name]['properties'])

    def test_update_tools_accept_runner_baseline_and_evidence(self):
        for tool in (people.COMMULINGO_PERSON_UPDATE_TOOL, people.COMMULINGO_TERM_UPDATE_TOOL):
            with self.subTest(tool=tool['name']):
                fields = tool['input_schema']['properties']['fields']
                self.assertTrue({'expectedRevision', 'evidence'} <= set(fields['properties']))
                self.assertFalse(tool['input_schema']['additionalProperties'])
        person = people.COMMULINGO_PERSON_UPDATE_TOOL['input_schema']['properties']['fields']
        self.assertTrue({'expectedRevision', 'evidence'} <= set(person['required']))

    def test_cross_target_fields_are_rejected_before_storage(self):
        with self.assertRaises(people.CommulingoInputError) as failure:
            people.normalize_commulingo_write('person', 'fixture', {'definition': {'ko': '정의', 'en': 'Definition'}}, ['https://example.org/source'], None)
        self.assertEqual(failure.exception.code, 'unknown_field')

    def test_language_normalization_preserves_nationality_labels(self):
        original = {'bio': {'ko': '조지아 공산당에서 활동했다.', 'en': 'Worked in Georgia.'},
                    'citizenship': {'code': 'georgia', 'label': {'ko': '조지아', 'en': 'Georgia'}}}
        normalized, _, _, _ = people.normalize_commulingo_write('person', 'fixture', deepcopy(original), ['https://example.org/source'], None)
        self.assertIn('그루지야', normalized['bio']['ko'])
        self.assertEqual(normalized['citizenship'], original['citizenship'])


class OfflineBoundaryTests(unittest.TestCase):
    def test_swallowed_provider_error_still_fails_the_check(self):
        import httpx
        with self.assertRaisesRegex(AssertionError, 'unexpected external IO: httpx.Client.send'):
            with no_external_io():
                # Provider adapters can swallow failures and report unavailable.
                # That must not hide an accidentally unmocked paid call.
                try:
                    with httpx.Client() as client:
                        client.get('https://example.invalid/never-requested')
                except AssertionError:
                    pass

    def test_storage_and_frontend_processes_require_explicit_fakes(self):
        import psycopg2
        import subprocess
        for call, boundary in ((lambda: psycopg2.connect('dbname=never-connected'), 'psycopg2.connect'),
                               (lambda: subprocess.run(['never-executed']), 'subprocess.run')):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(AssertionError, boundary):
                with no_external_io():
                    call()


if __name__ == '__main__':
    unittest.main()
