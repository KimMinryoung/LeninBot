"""Regression tests for bounded sync, source identity and search diagnostics."""
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch, Mock

from jobs import kg_sync, kg_sync_commulingo as comm, kg_sync_documents as docs
from kg_runtime import identity, search, doc_extract as dx
from graph_memory.structured_writer import _make_entity_edge
from tool_gateway.results import ToolFailure


class SyncTests(unittest.TestCase):
    def facts(self, n):
        return [{"subject_name": "A", "object_name": "B", "predicate": "Statement",
                 "fact": f"new {i}", "attributes": {"sync_key": f"commulingo:{i}"}} for i in range(n)]

    def run_comm(self, n=45, full=True, limit=40, failure=False):
        source = SimpleNamespace(**{k: [] for k in ('people', 'events', 'terms', 'offices', 'office_rows', 'event_people', 'redirects')})
        facts = self.facts(n)
        old = {f"commulingo:{i}": {"uuid": f"old{i}", "fact": "old", "expired": False} for i in range(n)}
        old['commulingo:gone'] = {"uuid": "gone", "fact": "removed", "expired": False}
        order = []
        def write(batch):
            order.append('write')
            return {"successful_keys": [] if failure else [f['attributes']['sync_key'] for f in batch],
                    "errors": ['failure'] if failure else [], "rejected": 0}
        def expire(uuids):
            order.append(list(uuids))
            return len(uuids)
        with patch.object(comm, 'load_source', return_value=source), patch.object(comm, 'build_facts', return_value=facts), \
             patch.object(comm, 'existing_sync_edges', return_value=old), patch.object(comm, 'write_facts', side_effect=write), \
             patch.object(comm, 'expire_edges', side_effect=expire), patch.object(comm, 'refresh_curated_profiles', return_value=0), \
             patch.object(comm, 'apply_redirects', return_value={}):
            stats = comm.run(full=full, limit=limit)
        return stats, order

    def test_cap_keeps_remaining_and_does_not_expire_unwritten_or_deleted(self):
        stats, order = self.run_comm()
        self.assertFalse(stats['complete'])
        self.assertEqual(stats['remaining'], 5)
        self.assertEqual(order, ['write', [f'old{i}' for i in range(40)]])

    def test_failure_preserves_old_relations(self):
        stats, order = self.run_comm(failure=True)
        self.assertEqual(stats['remaining'], 45)
        self.assertEqual(stats['failed'], 40)
        self.assertEqual(order, ['write', []])

    def test_full_success_expires_vanished_only_after_write(self):
        stats, order = self.run_comm(n=3)
        self.assertTrue(stats['complete'])
        self.assertEqual(order, ['write', ['old0', 'old1', 'old2'], ['gone']])

    def test_partial_run_preserves_watermark_in_sql(self):
        with patch.object(kg_sync, '_ensure_table'), patch.object(kg_sync, 'db_execute') as execute:
            kg_sync.set_state('documents', watermark=datetime.now(timezone.utc), full=True,
                              stats={'complete': False, 'remaining': 2})
        args = execute.call_args.args[1]
        self.assertIsNone(args[1])
        self.assertFalse(args[2])
        self.assertFalse(args[-1])

    def test_dry_run_missing_state_never_initializes_table(self):
        with patch.object(kg_sync, 'db_query_one', return_value={'name': None}), \
             patch.object(kg_sync, '_ensure_table', side_effect=AssertionError('write')):
            self.assertIsNone(kg_sync.get_state('documents', dry_run=True)['watermark'])

    def test_failed_requested_full_resumes_full(self):
        state = {'watermark': datetime.now(timezone.utc), 'last_full_at': datetime.now(timezone.utc),
                 'stats': {'mode': 'full', 'complete': False}}
        mod = Mock(); mod.run.return_value = {'complete': True}
        with patch.object(kg_sync, 'get_state', return_value=state), patch.object(kg_sync, '_load_source', return_value=mod), \
             patch.object(kg_sync, 'set_state'):
            kg_sync.run_source('documents')
        self.assertTrue(mod.run.call_args.kwargs['full'])
        self.assertIsNone(mod.run.call_args.kwargs['since'])

    def test_process_interrupt_leaves_running_full_intent(self):
        state = {'watermark': datetime.now(timezone.utc), 'last_full_at': datetime.now(timezone.utc)}
        mod = Mock(); mod.run.side_effect = KeyboardInterrupt()
        with patch.object(kg_sync, 'get_state', return_value=state), patch.object(kg_sync, '_load_source', return_value=mod), \
             patch.object(kg_sync, 'set_state') as save:
            with self.assertRaises(KeyboardInterrupt):
                kg_sync.run_source('documents', full=True)
        saved = save.call_args.kwargs['stats']
        self.assertFalse(saved['complete'])
        self.assertEqual(saved['mode'], 'full')
        self.assertEqual(saved['phase'], 'running')

    def test_invalid_extraction_json_does_not_count_as_empty_success(self):
        with self.assertRaises(ValueError):
            dx.parse_llm_facts('broken JSON', strict=True)
        self.assertEqual(dx.parse_llm_facts('{"facts": []}', strict=True), [])

    def test_document_cap_counts_failed_attempts_and_skips_unchanged(self):
        recs = [dx.research_record({'slug': f's{i}', 'title': 'T', 'markdown': str(i)}) for i in range(4)]
        existing = {recs[0].ref: recs[0]['sha']}
        with patch.object(docs, 'load_records', return_value=recs), patch.object(docs, '_commulingo_names', return_value={}), \
             patch.object(dx, 'existing_document_hashes', return_value=existing), \
             patch.object(identity, 'get_alias_index', return_value=Mock()), \
             patch.object(dx, 'extract_document', side_effect=RuntimeError('offline')) as extract:
            stats = docs.run(limit=1)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(stats['remaining'], 3)
        self.assertEqual(stats['failed'], 1)
        self.assertFalse(stats['complete'])

    def test_document_write_failure_never_stamps_or_expires(self):
        rec = dx.research_record({'slug': 's', 'title': 'T', 'markdown': 'changed'})
        with patch.object(dx, 'build_document_facts', return_value=[{}]), \
             patch.object(dx, 'write_document_facts', return_value={'status': 'error'}), \
             patch.object(dx, 'expire_document_edges') as expire, patch.object(dx, 'stamp_document_node') as stamp:
            result = dx.extract_document(rec, existing_sha='old')
        self.assertEqual(result['status'], 'error')
        expire.assert_not_called(); stamp.assert_not_called()

    def test_sync_edge_version_is_idempotent(self):
        args = dict(source_uuid='a', target_uuid='b', predicate='Statement', fact_text='text',
                    group_id='documents', valid_at=None, episode_uuid='e', attributes={'sync_key': 'doc:x'})
        a = _make_entity_edge(**args)
        b = _make_entity_edge(**dict(args, episode_uuid='retry'))
        c = _make_entity_edge(**dict(args, fact_text='changed'))
        self.assertEqual(a.uuid, b.uuid)
        self.assertNotEqual(a.uuid, c.uuid)


class SearchTests(unittest.TestCase):
    def index(self):
        idx = identity.AliasIndex()
        idx.load_rows([
            {'uuid': 'concept', 'name': 'dialectical materialism', 'labels': ['Concept'], 'raw_aliases': ['diamat'],
             'keys': ['디아마트 diamat']},
            {'uuid': 'org', 'name': '디아마트 (DiaMat)', 'labels': ['Organization'], 'raw_aliases': []},
        ])
        return idx

    def test_diamat_collision_and_exact_name(self):
        idx = self.index()
        self.assertEqual({h.uuid for h in idx.match('DiaMat')}, {'concept', 'org'})
        self.assertEqual([h.uuid for h in idx.match('디아마트 (DiaMat)')], ['org'])
        self.assertEqual(idx.match('DiaMat 조직', broad=False), [])

    def test_ambiguous_write_identity_not_merged(self):
        rows = [{'uuid': i, 'name': i, 'labels': ['Organization'], 'same_label': True} for i in ['A', 'B']]
        self.assertEqual(identity._pick_key_hit(rows, 'Organization', 'Shared').method, 'ambiguous')
        self.assertEqual(identity._pick_key_hit(rows, 'Organization', 'A').uuid, 'A')
        self.assertFalse(identity._labels_compatible('Document', ['Incident']))

    def test_topic_query_calls_semantic_search_and_obeys_cap(self):
        hit = identity.AliasHit('u', '레닌', ['Person'], '레닌')
        result = {'nodes': [{'uuid': 'u', 'name': '레닌', 'labels': ['Person']}],
                  'edges': [{'uuid': str(i), 'fact': f'relevant {i}'} for i in range(10)]}
        with patch.object(search, '_alias_hits', return_value=[hit]), patch.object(search, 'get_kg_service', return_value=Mock()), \
             patch.object(search, 'run_kg_task', return_value=result) as run, patch.object(search, '_hydrate_nodes', return_value={}), \
             patch.object(search, '_hydrate_edges', return_value={}), patch.object(search, '_entity_neighborhood', side_effect=AssertionError('neighbours')):
            out = search.search_knowledge_graph('레닌의 제국주의 분석', 3)
        run.assert_called_once()
        self.assertEqual(out.result_metadata['result_count'], 3)
        self.assertEqual(out.result_metadata['path'], 'semantic')

    def test_entity_cap_and_empty_metadata(self):
        hit = identity.AliasHit('u', '레닌', ['Person'], '레닌')
        with patch.object(search, '_alias_hits', return_value=[hit]), \
             patch.object(search, '_entity_neighborhood', return_value=({'name': '레닌'}, [{'fact': 'x'}]*8)):
            out = search.search_knowledge_graph(entity='레닌', num_results=2)
        self.assertEqual(out.result_metadata['result_count'], 2)
        with patch.object(search, '_alias_hits', return_value=[]):
            out = search.search_knowledge_graph(entity='absent', mode='entity')
        self.assertFalse(out)
        self.assertTrue(out.result_metadata['empty'])

    def test_failed_search_is_not_empty_success(self):
        with patch.object(search, 'get_kg_service', return_value=None), patch.object(search, '_direct_cypher_search', return_value=None):
            out = search.search_knowledge_graph('q', mode='semantic')
        self.assertIsInstance(out, ToolFailure)
        self.assertIsNone(out.result_metadata['empty'])

    def test_audit_metadata_round_trip_and_old_unknown(self):
        import audit_sink
        row = {'tool_name': 'knowledge_graph_search', 'decision': 'allow',
               'result_metadata': {'path': 'semantic', 'result_count': 0, 'empty': True, 'fallback': False}}
        normalized = audit_sink.normalize_row('tool', row)
        self.assertEqual(json.loads(normalized['result_metadata']), row['result_metadata'])
        self.assertEqual(audit_sink.normalize_row('tool', normalized), normalized)
        self.assertIsNone(audit_sink.normalize_row('tool', {'tool_name': 'x', 'decision': 'allow'})['result_metadata'])


class ConformanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_deleted_self_loop_is_not_reported_as_written(self):
        from unittest.mock import AsyncMock
        from graph_memory import structured_writer as writer
        from graph_memory.conformance import ConformanceReport
        fact = {"subject_name": "Alpha", "subject_type": "Organization", "predicate": "OrgRelation",
                "object_name": "Beta", "object_type": "Organization", "fact": "Alpha relates to Beta"}
        def validate(result, **kwargs):
            return ConformanceReport(self_loops=[{"edge_uuid": result.edges[0].uuid}])
        graph = SimpleNamespace(driver=SimpleNamespace(client=Mock()), embedder=None)
        with patch.object(writer, "find_canonical_entity_uuid", new=AsyncMock(return_value="same")), \
             patch.object(writer, "_embed_in_batches", new=AsyncMock()), \
             patch.object(writer, "add_nodes_and_edges_bulk", new=AsyncMock()), \
             patch.object(writer, "validate_episode_result", side_effect=validate), \
             patch.object(writer, "apply_hard_fixes", new=AsyncMock()):
            result = await writer.write_structured_facts(graph, [fact], group_id="commulingo", allow_sync_predicates=True)
        self.assertEqual(result["facts_written"], 0)
        self.assertEqual(result["facts_rejected"], 1)
        self.assertEqual(result["written_fact_indices"], [])
        self.assertEqual(result["edge_uuids"], [])
        self.assertNotEqual(result["status"], "ok")

    def test_wrong_external_endpoint_requires_replacement(self):
        fact = {"fact": "unchanged", "object_external_id": "commulingo:term:united-opposition"}
        old = {"fact": "unchanged", "expired": False, "object_ids": ["commulingo:term:left-opposition"]}
        self.assertTrue(comm.fact_changed(fact, old))

    def test_usage_metrics_are_json_serializable_with_unknown_history(self):
        from decimal import Decimal
        from kg_runtime.metrics import usage_metrics
        row = {"tool_name": "knowledge_graph_search", "interface": "webchat", "agent": "", "result_status": "ok",
               "n": 2, "avg_ms": Decimal(3), "p95_ms": Decimal(4), "empty": 1, "measured": 1, "fallback": 0, "diagnosed": 1}
        with patch("db.query", return_value=[row]):
            metrics = usage_metrics()
        json.dumps(metrics)
        self.assertEqual(metrics["callers"][0]["unknown"], 1)
        self.assertEqual(metrics["callers"][0]["empty_rate"], 1.0)


if __name__ == '__main__':
    unittest.main()
