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

    def test_verdict_rounds_and_thresholds(self):
        check = verdict(decision('contradicts', 0.851234), SETTINGS['thresholds'])
        self.assertEqual(check['confidence'], 0.851)
        self.assertIn('reject', check)
        self.assertNotIn('reject', verdict(decision('contradicts', 0.84), SETTINGS['thresholds']))


if __name__ == '__main__':
    unittest.main()
