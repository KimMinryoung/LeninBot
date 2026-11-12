"""Repeated agent/news claims are not stored again; fiction yields no LLM facts."""
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from graph_memory import structured_writer as writer
from kg_runtime import doc_extract as dx
from llm.call_registry import CallSiteProfile, Decision


def _profile(feature, **thresholds):
    return CallSiteProfile(feature=feature, provider='typesafe', model='jev-test', extra={'thresholds': thresholds})


class _Client:
    @asynccontextmanager
    async def session(self, **kwargs):
        yield object()


def _fact(text, obj='Federal Reserve', **extra):
    return {'subject_name': 'Kevin Warsh', 'subject_type': 'Person', 'predicate': 'Affiliation',
            'object_name': obj, 'object_type': 'Organization', 'fact': text, **extra}


EXISTING = {0: [{'uuid': 'old-edge-1234', 'fact': 'Kevin Warsh became Fed chair on 2026-05-15.',
                 'valid_at': '2026-05-15T00:00:00Z'}]}


class ClaimDedupeTests(unittest.IsolatedAsyncioTestCase):
    async def _write(self, facts, *, noul, existing=EXISTING, dedupe=True):
        graph = SimpleNamespace(driver=SimpleNamespace(client=_Client()), embedder=None)
        answers = None if noul is None else Decision(
            answers={f'f{n}e{j}_same': {'noul': noul} for n in range(len(facts)) for j in range(3)}, model='m')
        with patch.object(writer, 'find_canonical_entity_uuid', new=AsyncMock(side_effect=lambda *a, **k: f'node-{a[2]}')), \
             patch.object(writer, '_existing_same_relations', new=AsyncMock(return_value=existing)), \
             patch('llm.call_registry.resolve', return_value=_profile('kg_claim_dedupe', same=0.9)), \
             patch('llm.call_registry.decide', new=AsyncMock(return_value=answers)) as decide, \
             patch.object(writer, '_embed_in_batches', new=AsyncMock()), \
             patch.object(writer, 'upsert_identity_async', new=AsyncMock()), \
             patch.object(writer, 'add_nodes_and_edges_bulk', new=AsyncMock()) as save, \
             patch.object(writer, 'validate_episode_result', return_value=None):
            result = await writer.write_structured_facts(graph, facts, group_id='economy', dedupe_claims=dedupe)
        return result, save, decide

    async def test_restated_claim_is_skipped_and_reported(self):
        result, save, decide = await self._write([_fact('워시는 2026년 5월 15일 연준 의장에 취임했다.')], noul=0.97)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual((result['facts_written'], result['facts_duplicate']), (0, 1))
        self.assertEqual(result['duplicate_facts'][0]['existing_edge'], 'old-edge')
        save.assert_not_awaited()
        state = decide.await_args.args[1]
        self.assertEqual(state['items']['f0e0']['existing_valid_at'], '2026-05-15T00:00:00Z')

    async def test_new_claim_and_unavailable_judgement_are_written(self):
        result, save, _ = await self._write([_fact('Warsh raised rates on 2026-06-17.')], noul=0.2)
        self.assertEqual((result['facts_written'], result['facts_duplicate']), (1, 0))
        save.assert_awaited_once()
        result, _, _ = await self._write([_fact('Warsh raised rates on 2026-06-17.')], noul=None)
        self.assertEqual(result['facts_written'], 1)

    async def test_only_flagged_writes_and_existing_endpoints_are_checked(self):
        _, _, decide = await self._write([_fact('x')], noul=0.99, dedupe=False)
        decide.assert_not_awaited()
        _, _, decide = await self._write([_fact('x')], noul=0.99, existing={})
        decide.assert_not_awaited()


class BatchAndSessionTests(unittest.IsolatedAsyncioTestCase):
    """In-batch repeats, the pair cap, and no open session during the judgement."""

    async def _drop(self, facts, *, existing, noul, decide_hook=None):
        resolved = [(f, 'node-a', False, 'node-b', False) for f in facts]
        answers = Decision(answers={k: {'noul': noul} for k in
                                    [f'f{n}{x}{m}_same' for n in range(len(facts)) for x in 'eb' for m in range(50)]},
                           model='m')
        client = _TrackingClient()

        async def fake_decide(*args):
            if decide_hook:
                decide_hook(client)
            return answers

        with patch.object(writer, '_existing_same_relations', new=AsyncMock(return_value=existing)), \
             patch('llm.call_registry.resolve', return_value=_profile('kg_claim_dedupe', same=0.9)), \
             patch('llm.call_registry.decide', new=AsyncMock(side_effect=fake_decide)) as decide:
            kept, kept_idx, dups = await writer._drop_repeated_claims(client, 'neo4j', resolved, list(range(len(facts))))
        return kept_idx, dups, decide

    async def test_exact_repeat_in_batch_is_dropped_without_judgement(self):
        facts = [_fact('Warsh  joined the Fed.'), _fact('warsh joined the fed.')]
        kept, dups, decide = await self._drop(facts, existing={}, noul=0.0)
        self.assertEqual(kept, [0])
        self.assertEqual((dups[0]['index'], dups[0]['duplicate_of_index']), (1, 0))
        decide.assert_not_awaited()

    async def test_restatement_in_batch_is_judged(self):
        facts = [_fact('Warsh joined the Fed in May.'), _fact('In May Warsh became part of the Fed.')]
        kept, dups, decide = await self._drop(facts, existing={}, noul=0.95)
        self.assertEqual(kept, [0])
        self.assertEqual(dups[0]['duplicate_of_index'], 0)
        self.assertIn('f1b0', decide.await_args.args[1]['items'])

    async def test_session_is_closed_before_judgement_and_cap_is_logged(self):
        existing = {0: [{'uuid': f'edge-{j}', 'fact': f'old {j}', 'valid_at': None} for j in range(45)]}
        open_during_decide = []
        with self.assertLogs(writer.logger, level='WARNING') as logs:
            await self._drop([_fact('new')], existing=existing, noul=0.0,
                             decide_hook=lambda c: open_during_decide.append(c.open))
        self.assertEqual(open_during_decide, [0])
        self.assertTrue(any('judging the first 40' in line for line in logs.output))


class _TrackingClient:
    def __init__(self):
        self.open = 0

    @asynccontextmanager
    async def session(self, **kwargs):
        self.open += 1
        try:
            yield object()
        finally:
            self.open -= 1


class FictionGateTests(unittest.TestCase):
    REC = dx.DocRecord(kind='research', ident='story', ref='research:story', title='기울기 — 루멘에 관한 이야기',
                       description='', text='2028년 2월, 로잔 엘레나 바스케스는 논문을 발표했다.', url=None,
                       lang='ko', published_at='2026-04-11', sha='s', aliases=[],
                       links={'person': [], 'term': [], 'event': []})

    def _build(self, noul):
        genre = {}
        decision = None if noul is None else Decision(answers={'fiction': {'noul': noul}}, model='m')
        with patch('llm.call_registry.resolve', return_value=_profile('kg_document_genre', fiction=0.8)), \
             patch('llm.call_registry.decide_sync', return_value=decision), \
             patch.object(dx, 'run_llm_extraction', return_value=[]) as extract:
            facts = dx.build_document_facts(self.REC, use_llm=True, genre=genre)
        return facts, extract, genre

    def test_fiction_keeps_document_links_but_skips_llm_facts(self):
        facts, extract, genre = self._build(0.96)
        extract.assert_not_called()
        self.assertEqual(genre, {'fiction': True, 'probability': 0.96})
        self.assertTrue(facts)  # the collection link stays

    def test_non_fiction_and_unavailable_judgement_extract(self):
        _, extract, genre = self._build(0.05)
        extract.assert_called_once()
        self.assertFalse(genre['fiction'])
        _, extract, genre = self._build(None)
        extract.assert_called_once()
        self.assertEqual(genre, {'fiction': False, 'probability': None})

    def test_prompts_carry_future_date_rule(self):
        from jobs import news_kg
        self.assertIn(dx.FUTURE_GUIDELINE, dx.EXTRACTION_SYSTEM)
        self.assertIn(dx.FUTURE_GUIDELINE, news_kg.NEWS_EXTRACTION_SYSTEM)


if __name__ == '__main__':
    unittest.main()
