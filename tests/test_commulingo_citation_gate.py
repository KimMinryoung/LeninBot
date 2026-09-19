"""Citation-support gate: confident non-support rejects, uncertainty and outages pass."""
import unittest
from unittest.mock import patch

from commulingo_pipeline import citation_gate
from commulingo_pipeline.citation_gate import check_claims, verdict
from commulingo_pipeline.engine import Usage
from commulingo_pipeline.evidence import snapshot
from llm.call_registry import Decision


def decision(support, confidence, specific=0.9, boilerplate=0.0):
    return Decision(answers={
        'support': {'type': 'choice', 'choice': support, 'confidence': confidence,
                    'probabilities': {support: confidence}},
        'specific': {'type': 'noul', 'noul': specific},
        'boilerplate': {'type': 'noul', 'noul': boilerplate},
    }, model='typesafe/jev-1.13-test')


SETTINGS = {'enabled': True, 'enforce': True, 'thresholds': {'reject': 0.85, 'boilerplate': 0.9}}


class CitationGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.source = snapshot('https://example.org/a', 'Kosygin was born in 1904 in Saint Petersburg. He chaired the Council.')
        self.sources = {self.source['id']: self.source}
        self.claims = [
            {'field': 'years', 'claim': 'Born 1904', 'source_id': self.source['id'], 'start': 0, 'end': 45},
            {'field': 'body', 'claim': 'Executed in 1938', 'source_id': self.source['id'], 'start': 46, 'end': 70},
        ]
        settings = patch.object(citation_gate, 'settings', return_value=dict(SETTINGS))
        settings.start(); self.addCleanup(settings.stop)

    async def test_supported_claims_pass_and_are_recorded(self):
        async def decide(feature, state, questions):
            self.assertEqual(feature, 'commulingo_citation_support')
            self.assertEqual(set(state), {'field', 'claim', 'source_url', 'excerpt'})
            self.assertEqual(set(questions), {'support', 'specific', 'boilerplate'})
            return decision('supports', 0.97)
        usage = Usage()
        checks = await check_claims(self.claims, self.sources, usage=usage, decide=decide)
        self.assertEqual([c['support'] for c in checks], ['supports', 'supports'])
        self.assertEqual(usage.tracker['citation_checks'], 2)
        self.assertNotIn('citation_rejections', usage.tracker)

    async def test_confident_unrelated_rejects_naming_only_that_claim(self):
        async def decide(feature, state, questions):
            return decision('unrelated', 0.99) if state['claim'] == 'Executed in 1938' else decision('supports', 0.95)
        usage = Usage()
        with self.assertRaises(ValueError) as ctx:
            await check_claims(self.claims, self.sources, usage=usage, decide=decide)
        message = str(ctx.exception)
        self.assertIn('one claim', message)
        self.assertIn("claim 2 (body, 'Executed in 1938')", message)
        self.assertIn('judged unrelated', message)
        self.assertNotIn('claim 1', message)
        self.assertEqual(usage.tracker['citation_rejections'], 1)

    async def test_low_confidence_unrelated_passes(self):
        async def decide(feature, state, questions):
            return decision('unrelated', 0.41)
        checks = await check_claims(self.claims, self.sources, decide=decide)
        self.assertEqual([c['support'] for c in checks], ['unrelated', 'unrelated'])
        self.assertTrue(all('reject' not in c for c in checks))

    async def test_boilerplate_page_rejects_even_when_support_uncertain(self):
        async def decide(feature, state, questions):
            return decision('unrelated', 0.5, boilerplate=0.97)
        with self.assertRaisesRegex(ValueError, 'access check, consent notice or other boilerplate'):
            await check_claims(self.claims[:1], self.sources, decide=decide)

    async def test_shadow_mode_records_without_raising(self):
        citation_gate.settings.return_value = {**SETTINGS, 'enforce': False}
        async def decide(feature, state, questions):
            return decision('contradicts', 0.95)
        usage = Usage()
        checks = await check_claims(self.claims, self.sources, usage=usage, decide=decide)
        self.assertTrue(all(c.get('reject') for c in checks))
        self.assertEqual(usage.tracker['citation_rejections'], 2)

    async def test_unavailable_decisions_pass_and_are_counted(self):
        async def decide(feature, state, questions):
            return None
        usage = Usage()
        checks = await check_claims(self.claims, self.sources, usage=usage, decide=decide)
        self.assertEqual(checks, [{'support': None, 'error': 'decision unavailable'}] * 2)
        self.assertEqual(usage.tracker['citation_unavailable'], 2)

    async def test_disabled_gate_makes_no_calls(self):
        citation_gate.settings.return_value = {**SETTINGS, 'enabled': False}
        async def decide(feature, state, questions):
            raise AssertionError('must not be called')
        self.assertEqual(await check_claims(self.claims, self.sources, decide=decide), [])

    async def test_disputing_claim_may_cite_a_contradicting_source(self):
        disputing = [{**self.claims[0], 'stance': 'disputes'}]
        async def decide(feature, state, questions):
            return decision('contradicts', 0.99)
        [check] = await check_claims(disputing, self.sources, decide=decide)
        self.assertNotIn('reject', check)
        async def unrelated(feature, state, questions):
            return decision('unrelated', 0.99)
        with self.assertRaisesRegex(ValueError, 'judged unrelated'):
            await check_claims(disputing, self.sources, decide=unrelated)

    async def test_cache_keeps_verdicts_for_unchanged_claims(self):
        calls = []
        async def decide(feature, state, questions):
            calls.append(state['claim'])
            return decision('unrelated', 0.99) if state['claim'] == 'Executed in 1938' else decision('supports', 0.9)
        cache = {}
        with self.assertRaises(ValueError):
            await check_claims(self.claims, self.sources, decide=decide, cache=cache)
        # The model fixes claim 2 and resubmits claim 1 unchanged: only the new claim is judged.
        fixed = [self.claims[0], {**self.claims[1], 'claim': 'Chaired the Council', 'start': 46, 'end': 70}]
        async def decide2(feature, state, questions):
            calls.append(state['claim'])
            return decision('supports', 0.95)
        checks = await check_claims(fixed, self.sources, decide=decide2, cache=cache)
        self.assertEqual(calls, ['Born 1904', 'Executed in 1938', 'Chaired the Council'])
        self.assertEqual([c['support'] for c in checks], ['supports', 'supports'])

    def test_annotate_puts_compact_check_on_claim_without_reject_text(self):
        from commulingo_pipeline.citation_gate import annotate
        checks = [verdict(decision('unrelated', 0.99), SETTINGS['thresholds']), {'support': None, 'error': 'x'}]
        annotated = annotate(self.claims, checks)
        self.assertEqual(annotated[0]['citation_check'],
                         {'support': 'unrelated', 'confidence': 0.99, 'specific': 0.9, 'boilerplate': 0.0})
        self.assertNotIn('reject', str(annotated[0]))
        self.assertNotIn('model', annotated[0]['citation_check'])
        self.assertNotIn('citation_check', annotated[1])
        self.assertEqual(annotate(self.claims, []), self.claims)

    def test_verdict_rounds_and_thresholds(self):
        check = verdict(decision('contradicts', 0.851234), SETTINGS['thresholds'])
        self.assertEqual(check['confidence'], 0.851)
        self.assertIn('reject', check)
        self.assertNotIn('reject', verdict(decision('contradicts', 0.84), SETTINGS['thresholds']))


if __name__ == '__main__':
    unittest.main()
