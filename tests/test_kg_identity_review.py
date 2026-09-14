import unittest
from unittest.mock import Mock, AsyncMock

from kg_runtime.identity_review import reviewed_target, checked_result
from kg_runtime.identity import resolve_entity_sync, resolve_entity_async


class ReviewTests(unittest.TestCase):
    def test_city_and_incident_remain_distinct(self):
        city = reviewed_target('숨가이트', 'Location')
        incident = reviewed_target('숨가이트', 'Incident')
        self.assertNotEqual(city['target_uuid'], incident['target_uuid'])
        self.assertEqual(reviewed_target(city['name'], 'Location')['target_uuid'], city['target_uuid'])

    def test_unreviewed_names_and_new_source_ids_never_redirect(self):
        self.assertIsNone(reviewed_target('어떤 사람', 'Person'))
        self.assertIsNone(reviewed_target('Cyber-Lenin', 'Person', external_id='commulingo:person:new-namesake'))
        self.assertIsNone(reviewed_target('숨가이트', 'Person'))

    def test_misclassified_collective_resolves_to_reviewed_concept(self):
        entry = reviewed_target('자영업자', 'Person')
        session = Mock()
        session.run.return_value.single.return_value = {'uuid': entry['target_uuid'], 'name': '자영업자', 'labels': ['Entity', 'Concept']}
        result = resolve_entity_sync(session, name='자영업자', entity_type='Person', trusted=False)
        self.assertEqual(result.uuid, entry['target_uuid'])
        self.assertEqual(result.method, 'reviewed')

    def test_missing_or_retyped_target_fails_closed(self):
        entry = reviewed_target('Cyber-Lenin', 'Person')
        with self.assertRaises(RuntimeError):
            checked_result(None, entry)
        with self.assertRaises(RuntimeError):
            checked_result({'uuid': entry['target_uuid'], 'name': 'Cyber-Lenin', 'labels': ['Person']}, entry)

    def test_maintenance_exclusion_does_not_choose_another_alias(self):
        entry = reviewed_target('자영업자', 'Concept')
        session = Mock()
        result = resolve_entity_sync(session, name='자영업자', entity_type='Concept', exclude_uuid=entry['target_uuid'])
        self.assertFalse(result.found)
        session.run.assert_not_called()


class AsyncReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_matches_sync_review(self):
        entry = reviewed_target('Cyber-Lenin', 'Person')
        record = {'uuid': entry['target_uuid'], 'name': 'Cyber-Lenin', 'labels': ['Entity', 'Organization']}
        session = Mock()
        session.run = AsyncMock(return_value=Mock(single=AsyncMock(return_value=record)))
        result = await resolve_entity_async(session, name='Cyber-Lenin', entity_type='Person', trusted=False)
        self.assertEqual(result.uuid, entry['target_uuid'])
        self.assertEqual(result.method, 'reviewed')
