"""Current author schemas and shared write validation, without live storage."""
from copy import deepcopy
import unittest
from unittest.mock import Mock
from commulingo_test_support import no_external_io

from commulingo.pipeline.decisions import Decisions
from commulingo.pipeline.engine import Usage
from commulingo.pipeline.patches import schema_for
from commulingo import people


class WriteContractTests(unittest.TestCase):
    def setUp(self):
        self.cursor = Mock()
        self.cursor.fetchone.return_value = None
        self.term = {
            'term': {'ko': '전시 공산주의', 'en': 'War communism'},
            'definition': {'ko': '전시 경제정책이다.', 'en': 'A wartime economic policy.'},
            'aliases': {'ko': [], 'en': []},
            'period': {'ko': '1918–1921', 'en': '1918–1921'},
            'startYear': 1918, 'endYear': 1921, 'category': 'economy',
        }

    def test_term_author_omits_category_but_storage_requires_classification(self):
        tool = people.COMMULINGO_TERM_CREATE_TOOL['input_schema']['properties']['fields']
        self.assertNotIn('category', tool['properties'])
        self.assertNotIn('category', tool['required'])
        self.assertIn('period', tool['required'])
        job = {'kind': 'term', 'action': 'create', 'topic': 'basics'}
        storage = schema_for(job, None)
        self.assertIn('category', storage['required'])
        author = Decisions(job, None, None, Usage()).author_schema(storage)
        self.assertNotIn('category', author['properties'])
        self.assertIsNone(people._validate(self.cursor, 'term', 'create', 'fixture', self.term))
        for category, message in ((None, 'category is required'), ('not-a-category', 'category must be one of')):
            with self.subTest(category=category):
                fields = {**self.term, 'category': category}
                if category is None:
                    fields.pop('category')
                self.assertIn(message, people._validate(self.cursor, 'term', 'create', 'fixture', fields))

    def test_dated_terms_require_ordered_years_and_bilingual_period(self):
        for update, message in (({'period': None}, 'period is required'),
                                ({'period': '1918–1921'}, 'must be an object'),
                                ({'startYear': None}, 'startYear is required'),
                                ({'endYear': 1900}, 'before startYear')):
            with self.subTest(update=update):
                self.assertIn(message, people._validate(self.cursor, 'term', 'create', 'fixture', {**self.term, **update}))
        undated = {**self.term, 'period': {'ko': '개념', 'en': 'Concept'}, 'startYear': None, 'endYear': None}
        self.assertIsNone(people._validate(self.cursor, 'term', 'create', 'fixture', undated))

    def test_person_author_provides_labels_and_runner_assigns_codes(self):
        fields = people.COMMULINGO_PERSON_CREATE_TOOL['input_schema']['properties']['fields']
        self.assertTrue({'citizenship', 'nationalOrigin', 'evidence'} <= set(fields['required']))
        self.assertTrue({'groupId', 'role'}.isdisjoint(fields['properties']))
        for name, code in (('citizenship', 'code'), ('nationalOrigin', 'code'), ('fate', 'kind')):
            self.assertNotIn(code, fields['properties'][name]['properties'])
            self.assertIn('label', fields['properties'][name]['properties'])
        nationality = {'code': 'russia', 'label': {'ko': '러시아', 'en': 'Russia'}}
        # Public nationalOrigin is normalized to the legacy SQL field origin.
        self.assertIsNone(people._person_create_nationality_problem({'citizenship': nationality, 'origin': nationality}))
        self.assertIn('nationalOrigin', people._person_create_nationality_problem({'citizenship': nationality}))
        self.assertIn('citizenship', people._person_create_nationality_problem({'origin': nationality}))

    def test_update_tools_accept_runner_baseline_and_evidence(self):
        for tool in (people.COMMULINGO_PERSON_UPDATE_TOOL, people.COMMULINGO_TERM_UPDATE_TOOL):
            with self.subTest(tool=tool['name']):
                fields = tool['input_schema']['properties']['fields']
                self.assertTrue({'expectedRevision', 'evidence'} <= set(fields['properties']))
                self.assertFalse(tool['input_schema']['additionalProperties'])
        person = people.COMMULINGO_PERSON_UPDATE_TOOL['input_schema']['properties']['fields']
        self.assertTrue({'expectedRevision', 'evidence'} <= set(person['required']))

    def test_field_limits_are_enforced_at_the_shared_write_boundary(self):
        for field in ('bio', 'moment', 'epithet'):
            for index, language in enumerate(('ko', 'en')):
                with self.subTest(field=field, language=language):
                    value = {'ko': '본문', 'en': 'Text'}
                    value[language] = ('가' if language == 'ko' else 'x') * (people.FIELD_LIMITS[field][index] + 1)
                    self.assertIn('too long', people._validate(self.cursor, 'person', 'update', 'fixture', {field: value}))

    def test_section_duplicate_topics_and_length_are_rejected(self):
        cursor = Mock()
        cursor.fetchone.side_effect = lambda: None if 'WHERE person_id' in cursor.execute.call_args.args[0] else {'ok': 1}
        cursor.fetchall.return_value = [{'slug': 'existing', 'heading_ko': '기존 제목', 'heading_en': 'Existing heading'}]
        section = {'slug': 'new-section', 'heading': {'ko': '기존 제목', 'en': 'Existing heading'},
                   'body': {'ko': '본문', 'en': 'Body'}}
        self.assertIn('already covers this topic', people._validate(cursor, 'person_section', 'create', 'fixture', section))
        section['heading'] = {'ko': '새 제목', 'en': 'New heading'}
        self.assertIsNone(people._validate(cursor, 'person_section', 'create', 'fixture', section))
        section['body']['ko'] = '가' * (people.FIELD_LIMITS['section_body'][0] + 1)
        self.assertIn('too long', people._validate(cursor, 'person_section', 'create', 'fixture', section))

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
