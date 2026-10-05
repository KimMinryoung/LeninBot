"""Regression coverage for source changes, withdrawal, dates and provenance."""
import unittest
from unittest.mock import patch, Mock, AsyncMock
from types import SimpleNamespace
from copy import deepcopy
from datetime import datetime, timezone

from graph_memory.structured_writer import _make_entity_edge, validate_fact
from jobs import kg_sync_commulingo as comm, kg_sync_documents as docs
from kg_runtime import doc_extract as dx, read_policy, search, recall


class VersionTests(unittest.TestCase):
    def test_pinned_endpoint_mismatch_requires_repair(self):
        fact = {'predicate': 'Reference', 'fact': 'mentions', 'object_uuid': 'canonical'}
        old = dict(fact, object_uuid='duplicate', expired=False)
        self.assertTrue(comm.fact_changed(fact, old))
        old['object_uuid'] = 'canonical'
        self.assertFalse(comm.fact_changed(fact, old))

    def test_attributes_and_dates_change_versions_but_provenance_does_not(self):
        args = dict(source_uuid='a', target_uuid='b', predicate='Statement', fact_text='A said B',
                    group_id='documents', valid_at=None, episode_uuid='episode',
                    attributes={'sync_key': 'doc:test', 'same_subject': False})
        first = _make_entity_edge(**args)
        changed = _make_entity_edge(**dict(args, attributes={'sync_key': 'doc:test', 'same_subject': True}))
        retry = _make_entity_edge(**dict(args, episode_uuid='another', attributes=dict(args['attributes'], source_url='url')))
        dated = _make_entity_edge(**dict(args, invalid_at=datetime(2000, 1, 1, tzinfo=timezone.utc)))
        self.assertNotEqual(first.uuid, changed.uuid)
        self.assertNotEqual(first.uuid, dated.uuid)
        self.assertEqual(first.uuid, retry.uuid)

    def test_comparison_observes_meaningful_attributes_and_period(self):
        f = {'predicate': 'Involvement', 'fact': 'A relates to B', 'valid_at': '1953-01-01',
             'invalid_at': '1964-01-01', 'attributes': {'sync_key': 'a', 'role_in_incident': 'victim'}}
        old = dict(deepcopy(f), expired=False)
        self.assertFalse(comm.fact_changed(f, old))
        old['valid_at'] = '1953-01-01T00:00:00+00:00'
        self.assertFalse(comm.fact_changed(f, old))
        old['attributes']['role_in_incident'] = 'witness'
        self.assertTrue(comm.fact_changed(f, old))
        old = dict(deepcopy(f), expired=False, invalid_at='1965-01-01')
        self.assertTrue(comm.fact_changed(f, old))

    def test_invalid_end_date_is_rejected(self):
        fact = {'subject_name': 'Lenin', 'subject_type': 'Person', 'object_name': 'Bolshevik Party',
                'object_type': 'Organization', 'predicate': 'Affiliation', 'fact': 'Member',
                'valid_at': '1910-01-01', 'invalid_at': 'nonsense'}
        self.assertIn('ISO', validate_fact(fact, 0))
        fact['invalid_at'] = '1900-01-01'
        self.assertIn('after', validate_fact(fact, 0))

    def test_reconciliation_flags_obsolete_and_duplicate_versions(self):
        result = comm.compare_sync_facts([], {'old': {'expired': False, 'active_uuids': ['1', '2']}})
        self.assertEqual(result['unresolved'], 2)


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.rec = dx.archival_record({'id': 'fixture', 'title': {'ko': 'Title'}, 'people': ['a']}, '<p>body</p>')

    def test_metadata_hash_changes_independently_of_body(self):
        changed = deepcopy(self.rec)
        changed['title'] = 'New title'
        changed['links']['person'] = ['b']
        self.assertEqual(changed['sha'], self.rec['sha'])
        self.assertNotEqual(changed.metadata_sha, self.rec.metadata_sha)

    def test_mentions_retain_alias_hit_identity(self):
        index = Mock()
        index.match.return_value = [SimpleNamespace(uuid='known-node', name='Named Treaty', labels=['Policy'], key='named treaty')]
        facts = dx.mention_facts(self.rec, index)
        self.assertEqual(facts[0]['object_uuid'], 'known-node')
        self.assertIsNone(validate_fact(facts[0], 0, allow_sync_predicates=True))
        self.assertIn('reserved for sync', validate_fact(facts[0], 0))

    def test_metadata_repair_removes_links_and_preserves_llm_without_extraction(self):
        old = {'doc:archival:fixture:about:person:a': {'uuid': 'removed', 'expired': False},
               'doc:archival:fixture:llm:0': {'uuid': 'claim', 'expired': False}}
        with patch.object(dx, 'build_document_facts', return_value=[]), \
             patch.object(dx, 'filter_resolved_self_loops', return_value=([], [])), \
             patch.object(comm, 'expire_edges') as expire, patch.object(dx, 'stamp_document_node'), \
             patch.object(dx, 'run_llm_extraction', side_effect=AssertionError('LLM')):
            dx.refresh_document_metadata(self.rec, names={}, alias_index=None, existing_edges=old,
                                         state={'metadata_sha': self.rec.metadata_sha, 'active': True})
        expire.assert_called_once_with(['removed'])

    def test_failed_metadata_write_does_not_expire_or_stamp(self):
        fact = {'attributes': {'sync_key': 'doc:archival:fixture:collection'}}
        with patch.object(dx, 'build_document_facts', return_value=[fact]), \
             patch.object(dx, 'filter_resolved_self_loops', return_value=([fact], [])), \
             patch.object(dx, 'write_document_facts', return_value={'status': 'error'}), \
             patch.object(comm, 'expire_edges') as expire, patch.object(dx, 'stamp_document_node') as stamp:
            with self.assertRaises(RuntimeError):
                dx.refresh_document_metadata(self.rec, names={}, alias_index=None, existing_edges={}, state={})
        expire.assert_not_called()
        stamp.assert_not_called()

    def test_same_body_republication_restores_without_llm(self):
        with patch.object(dx, 'build_document_facts', return_value=[]), \
             patch.object(dx, 'filter_resolved_self_loops', return_value=([], [])), \
             patch.object(comm, 'expire_edges'), patch.object(dx, 'stamp_document_node'), \
             patch.object(dx, 'set_document_active') as activate, \
             patch.object(dx, 'run_llm_extraction', side_effect=AssertionError('LLM')):
            dx.refresh_document_metadata(self.rec, names={}, alias_index=None, existing_edges={},
                                         state={'sha': self.rec['sha'], 'active': False})
        activate.assert_called_once_with(self.rec.ref, True, exclude_uuids=[])

    def test_empty_successful_source_withdraws_only_selected_kinds(self):
        with patch.object(docs, 'load_records', return_value=[]), \
             patch.object(dx, 'existing_document_states', return_value={'archival:old': {'sha': 'x'}, 'research:keep': {'sha': 'y'}}), \
             patch.object(docs, '_commulingo_names', return_value={}), \
             patch('kg_runtime.identity.get_alias_index', return_value=Mock()), \
             patch.object(dx, 'withdraw_document') as withdraw:
            result = docs.run(kinds=('archival',))
        self.assertTrue(result['complete'])
        withdraw.assert_called_once_with('archival:old')

    def test_failed_source_never_withdraws(self):
        with patch.object(docs, 'load_records', side_effect=RuntimeError('unavailable')), \
             patch.object(dx, 'withdraw_document') as withdraw:
            with self.assertRaises(RuntimeError):
                docs.run()
        withdraw.assert_not_called()

    def test_withdraw_rechecks_current_publication(self):
        with patch.object(dx, 'current_research_record', return_value={'status': 'public'}), \
             patch.object(dx, 'set_document_active') as update:
            dx.withdraw_document('research:republished')
        update.assert_not_called()


class ReadTests(unittest.TestCase):
    def test_historical_assertions_and_half_open_date_bounds(self):
        edge = {'valid_at': '1953-01-01', 'invalid_at': '1964-10-01'}
        def policy(**kw):
            return read_policy.ReadPolicy(_visibility=([], []), **kw)
        self.assertTrue(policy().allows(edge))
        self.assertTrue(policy(as_of='1953-01-01').allows(edge))
        self.assertFalse(policy(as_of='1952-12-31').allows(edge))
        self.assertFalse(policy(as_of='1964-10-01').allows(edge))
        self.assertFalse(policy().allows(dict(edge, expired_at='2026-01-01')))
        self.assertTrue(policy(include_expired=True).allows(dict(edge, expired_at='2026-01-01')))

    def test_private_documents_are_hidden_even_when_expired_requested(self):
        p = read_policy.ReadPolicy(include_expired=True, _visibility=(['research:private'], ['u']))
        self.assertFalse(p.allows({'doc_ref': 'research:private'}))

    def test_publication_status_overrides_stale_graph(self):
        with patch.object(search, '_run_rows', return_value=[{'ref': 'research:private', 'uuid': 'u', 'active': True}]), \
             patch('db.query', return_value=[]):
            self.assertEqual(read_policy.document_visibility(), (['research:private'], ['u']))

    def test_publication_status_failure_does_not_release_document(self):
        with patch.object(search, '_run_rows', return_value=[{'ref': 'research:private', 'uuid': 'u', 'active': True}]), \
             patch('db.query', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                read_policy.document_visibility()

    def test_document_source_and_legacy_trust_are_explicit(self):
        row = {'uuid': 'e', 'sync_key': 'doc:research:slug:llm:0', 'doc_ref': 'research:slug',
               'ep_names': ['[T:corroborated]legacy'], 'fact': 'claim'}
        with patch.object(search, '_run_rows', return_value=[row]), \
             patch.object(read_policy, 'document_visibility', return_value=([], [])):
            edge = search._hydrate_edges(['e'])['e']
        line = search._format_edge_line(edge)
        self.assertIn('src: research:slug', line)
        self.assertIn('https://cyber-lenin.com/research/slug', line)
        self.assertIn('extraction: llm', line)
        self.assertIn('교차 검증 미확인', line)
        self.assertEqual(edge['tier'], 'single')

    def test_hydration_failure_never_returns_unverified_raw_claim(self):
        with patch.object(search, 'get_kg_service', return_value=Mock()), \
             patch.object(search, 'run_kg_task', return_value={'edges': [{'uuid': 'e', 'fact': 'private content'}]}), \
             patch.object(search, '_hydrate_nodes', return_value={}), \
             patch.object(search, '_hydrate_edges', side_effect=RuntimeError('offline')):
            result = search.search_knowledge_graph('test', mode='semantic')
        self.assertNotIn('private content', result)
        self.assertEqual(result.result_metadata['path'], 'error')

    def test_recall_records_failure_separately_from_empty(self):
        with patch.object(recall, 'enabled', return_value=True), \
             patch.object(search, '_alias_hits', side_effect=RuntimeError('offline')), \
             patch.object(recall, '_audit_recall') as audit:
            self.assertEqual(recall.entity_gated_kg_block('known name'), '')
        self.assertTrue(audit.call_args.args[0]['failed'])


class PinnedIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_name_mentions_keep_distinct_known_targets_without_creating_nodes(self):
        from graph_memory import structured_writer as writer
        from graph_memory.conformance import ConformanceReport
        facts = [dict(subject_name='Source document', subject_type='Document', subject_uuid='doc',
                      object_name='Named Treaty', object_type='Policy', object_uuid=uuid,
                      predicate='Reference', fact='Document mentions treaty', attributes={'sync_key': uuid})
                 for uuid in ('first', 'second')]
        graph = SimpleNamespace(driver=SimpleNamespace(client=Mock()), embedder=None)
        async def pinned(driver, database, uuid, label):
            return uuid
        with patch.object(writer, 'find_pinned_entity_uuid', side_effect=pinned), \
             patch.object(writer, 'find_canonical_entity_uuid', side_effect=AssertionError('must not resolve again')), \
             patch.object(writer, '_embed_in_batches', new=AsyncMock()), \
             patch.object(writer, 'add_nodes_and_edges_bulk', new=AsyncMock()) as save, \
             patch.object(writer, 'validate_episode_result', return_value=ConformanceReport()):
            result = await writer.write_structured_facts(graph, facts, group_id='documents', allow_sync_predicates=True)
        self.assertEqual(result['facts_written'], 2)
        self.assertEqual(result['new_entities'], 0)
        self.assertEqual(result['reused_entities'], 3)
        self.assertEqual(len(result['edge_uuids']), 2)
        save.assert_awaited_once()

    async def test_missing_pinned_node_fails_without_falling_back_to_new_identity(self):
        from graph_memory import structured_writer as writer
        session = AsyncMock()
        result = Mock(single=AsyncMock(return_value=None))
        session.run.return_value = result
        client = Mock()
        client.session.return_value.__aenter__ = AsyncMock(return_value=session)
        client.session.return_value.__aexit__ = AsyncMock(return_value=False)
        with self.assertRaisesRegex(ValueError, 'Pinned entity missing'):
            await writer.find_pinned_entity_uuid(client, 'neo4j', 'missing', 'Policy')


if __name__ == '__main__':
    unittest.main()
