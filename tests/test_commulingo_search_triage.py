import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from commulingo_pipeline import search_triage
from commulingo_pipeline.search_triage import parse_hits, triage_hits, shadow, search_target

RENDERED = ('<external source="web_search:brave:Постышев реабилитация">\n'
            'Search snippets, not full source pages. Retrieved at 2026-09-20T00:00:00+00:00; cache reuse preserves this retrieval time. '
            'Publication date is not event date.\n\n'
            '### Постышев, Павел Петрович — Википедия (2024-01-02)\n[source_kind=search_snippet; publication=2024-01-02; event_date=unknown]\n'
            'https://ru.wikipedia.org/wiki/Постышев\nПавел Петрович Постышев — советский партийный деятель.\nРеабилитирован в 1956 году.\n\n'
            '### Купить постышев (dress)\n[source_kind=search_snippet; publication=unknown; event_date=unknown]\n'
            'https://shop.example/postyshev\n\n'
            '### Last hit\n[source_kind=search_snippet; publication=unknown; event_date=unknown]\n'
            'https://example.org/last\nA snippet without a blank line before the closing tag\n</external>')


class SearchTriageTests(unittest.IsolatedAsyncioTestCase):
    def test_hits_are_parsed_from_the_rendered_result(self):
        hits = parse_hits(RENDERED)
        self.assertEqual([h['url'] for h in hits], ['https://ru.wikipedia.org/wiki/Постышев', 'https://shop.example/postyshev', 'https://example.org/last'])
        self.assertEqual(hits[0]['title'], 'Постышев, Павел Петрович — Википедия')
        self.assertEqual(hits[0]['snippet'], 'Павел Петрович Постышев — советский партийный деятель.\nРеабилитирован в 1956 году.')
        self.assertEqual(hits[1]['snippet'], '')
        self.assertEqual(hits[2]['snippet'], 'A snippet without a blank line before the closing tag')
        self.assertEqual(parse_hits('No results for: x\nSearch returned no matches'), [])

    async def test_verdicts_are_recorded_on_the_tracker_and_never_shown(self):
        target = search_target('person', 'postyshev', {'name': {'ko': '파벨 포스티셰프', 'en': 'Pavel Postyshev'}}, 'enrichment')
        self.assertEqual(target['labels'], ['파벨 포스티셰프', 'Pavel Postyshev'])
        seen = {}
        async def decide(feature, state, questions):
            seen.update(feature=feature, state=state, questions=questions)
            answers = {'h1': {'choice': 'directly', 'confidence': 0.97}, 'h2': {'choice': 'unrelated', 'confidence': 0.9},
                       'h3': {'choice': 'possibly', 'confidence': 0.5}}
            return SimpleNamespace(choice=lambda k: answers[k]['choice'], confidence=lambda k: answers[k]['confidence'])
        usage = SimpleNamespace(tracker={})
        with patch.object(search_triage, 'settings', return_value={'enabled': True}):
            rows = await triage_hits(target, parse_hits(RENDERED), usage=usage, decide=decide)
        self.assertEqual(seen['feature'], 'commulingo_search_triage')
        self.assertEqual(set(seen['state']), {'target', 'hits'})
        self.assertEqual(list(seen['questions']), ['h1', 'h2', 'h3'])
        self.assertIn('search hit h2', seen['questions']['h2']['instructions'])
        self.assertEqual(rows[1], {'url': 'https://shop.example/postyshev', 'verdict': 'unrelated', 'confidence': 0.9})
        self.assertEqual(usage.tracker['search_triage'], rows)
        self.assertEqual(usage.tracker['search_triage_calls'], 1)
        # Disabled: no call. Unavailable: counted, nothing recorded. A failure never reaches the search.
        with patch.object(search_triage, 'settings', return_value={'enabled': False}):
            self.assertEqual(await triage_hits(target, parse_hits(RENDERED), usage=usage, decide=AsyncMock()), [])
        with patch.object(search_triage, 'settings', return_value={'enabled': True}):
            self.assertEqual(await triage_hits(target, parse_hits(RENDERED), usage=usage, decide=AsyncMock(return_value=None)), [])
            self.assertEqual(usage.tracker['search_triage_unavailable'], 1)
            hook = shadow(target, usage, decide=AsyncMock(side_effect=RuntimeError('proxy down')))
            self.assertIsNone(await hook(RENDERED))

    async def test_review_wrapper_passes_search_results_to_the_hook_unchanged(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location('worker', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'commulingo_person_reviewer.py'))
        worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
        seen = []
        async def triage(text): seen.append(text)
        handlers = worker.make_handlers({'web_search': AsyncMock(return_value=RENDERED)}, {'source_refs': []}, {}, {}, triage=triage)
        self.assertEqual(await handlers['web_search'](query='x'), RENDERED)
        self.assertEqual(seen, [RENDERED])


if __name__ == '__main__':
    unittest.main()
