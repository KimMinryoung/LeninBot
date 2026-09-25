"""Passage identity survives viewports, refetches, and source-page restarts."""
import re
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from commulingo.pipeline.evidence import (
    Passages, SourcePages, compile_evidence, resolve_passages, snapshot,
)
from commulingo.review_policy import review_source, resolve_review_checks


class ImmutablePassageTests(unittest.TestCase):
    def test_viewports_select_canonical_paragraphs_without_rebinding_labels(self):
        body = 'First complete paragraph.\nSecond complete paragraph.'
        source = snapshot('https://example.org/page', body)
        registry = Passages()
        second = body.index('Second')
        shown = registry.show(source['id'], body, second + 4, len(body) - 3)
        self.assertEqual(shown, '[P1] Second complete paragraph.')
        whole = registry.show(source['id'], body)
        self.assertEqual(whole, '[P2] First complete paragraph.\n' + shown)
        self.assertEqual(registry.show(source['id'], body, second, second + 1), shown)
        self.assertEqual(registry.resolve(['P1'], lambda _: body), [(source['id'], second, len(body))])
        self.assertEqual(registry.show(source['id'], body, second, second), '')

    def test_long_paragraph_is_split_once_independent_of_viewport(self):
        body = '문장입니다. ' * 1200
        source = snapshot('https://example.org/long', body)
        registry = Passages()
        all_text = registry.show(source['id'], body)
        _, start, end = list(registry.shown.values())[1]
        part = registry.show(source['id'], body, start + 1, end - 1)
        self.assertEqual(part, f'[P2] {body[start:end]}')
        self.assertIn(part, all_text)
        self.assertTrue(all(e - s <= 3000 for _, s, e in registry.shown.values()))

    def test_new_snapshot_at_same_url_keeps_original_evidence(self):
        old = snapshot('https://example.org/page', 'Appointed in 1917.')
        new = snapshot(old['url'], 'Appointed in 1918.')
        sources = {s['id']: s for s in (old, new)}
        registry = Passages()
        registry.show(old['id'], old['body'])
        registry.show(new['id'], new['body'])
        claims = resolve_passages([{'field': 'bio', 'claim': 'Earlier account says 1917.',
                                    'passages': ['P1']}], registry, sources)
        self.assertEqual(claims[0]['source_id'], old['id'])
        self.assertEqual(compile_evidence(claims, sources, {'bio'})[0]['excerpt'], old['body'])
        self.assertEqual(registry.show(old['id'], old['body']), '[P1] Appointed in 1917.')

    def test_growing_and_restarted_snapshots_do_not_invalidate_old_labels(self):
        pages, registry, sources = SourcePages(), Passages(), {}
        with patch('commulingo.pipeline.evidence.MAX_SNAPSHOT_CHARS', 50):
            for body in ('first page', 'second page', 'z' * 45):
                logging = self.assertLogs('commulingo.pipeline.evidence', level='WARNING') if body.startswith('z') else nullcontext()
                with logging:
                    source, span, _ = pages.absorb('https://example.org/page', body)
                sources[source['id']] = source
                registry.show(source['id'], source['body'], *span)
        ranges = registry.resolve(['P1', 'P2', 'P3'], lambda sid: sources[sid]['body'])
        self.assertEqual([sources[sid]['body'][s:e] for sid, s, e in ranges],
                         ['first page', 'second page', 'z' * 45])

    def test_snapshot_mutation_is_refused_without_overwriting_registry(self):
        registry = Passages()
        registry.show('immutable-id', 'Original paragraph.')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            registry.show('immutable-id', 'Different paragraph.')
        self.assertEqual(registry.show('immutable-id', 'Original paragraph.'), '[P1] Original paragraph.')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            registry.resolve(['P1'], lambda _: 'Different paragraph.')
        with self.assertRaisesRegex(ValueError, 'not available'):
            registry.resolve(['P1'], lambda _: None)

    def test_mixed_valid_and_unknown_labels_refuse_only_the_named_claim(self):
        source = snapshot('https://example.org/page', 'Historical fact.')
        registry = Passages()
        registry.show(source['id'], source['body'])
        good = {'field': 'bio', 'claim': 'Historical fact.', 'passages': ['P1']}
        bad = {**good, 'passages': ['P1', 'P999']}
        with self.assertRaisesRegex(ValueError, 'claim 2: passage labels not displayed: P999'):
            resolve_passages([good, bad], registry, {source['id']: source})
        self.assertEqual(bad['passages'], ['P1', 'P999'])
        bad['passages'] = ['P1']
        self.assertEqual(len(resolve_passages([good, bad], registry, {source['id']: source})), 2)

    def test_review_refetch_uses_new_labels_and_preserves_earlier_quote(self):
        url = 'https://example.org/review'
        snapshots, registry = {}, Passages()
        old_id, shown = review_source(url, 'Old independently retrieved text.', snapshots, registry, base=500)
        new_id, fresh = review_source(url, 'New independently retrieved text.', snapshots, registry, base=500)
        self.assertNotEqual(old_id, new_id)
        self.assertEqual(re.findall(r'\[(P\d+)\]', shown + fresh), ['P1', 'P2'])
        self.assertEqual(review_source(url, 'Old independently retrieved text.', snapshots, registry, base=500)[1], shown)
        result = resolve_review_checks({'checks': [{'citation_id': 'S1', 'passages': ['P1'],
                                                   'finding': 'Earlier version confirmed.'}]},
                                       {'source_refs': [url]}, snapshots, registry)
        self.assertEqual(result['checks'][0]['quote'], 'Old independently retrieved text.')
