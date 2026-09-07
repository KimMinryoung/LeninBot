"""Offline regressions for shared execution, structure and source provenance."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llm import call_registry as registry
from runtime_tools.archival_translation import core
from runtime_tools import translation_memory as tm
from translation_runtime.structure import (
    html_problems, markdown_problems, protect_markdown, restore_markdown,
    markdown_chunks, semantic_review,
)
from scripts.translate_research_markdown import translate_markdown_with_retry, translate_one
from translation_runtime import (
    translate_validated, generate_translation, TranslationProviderError, TranslationCallError,
)


class Structure(unittest.TestCase):
    def test_html_relative_links_and_closing_tags(self):
        self.assertTrue(html_problems('<p><a href="/a">가</a></p>', '<p><a href="/b">A</a></p>'))
        self.assertTrue(html_problems('<p>A</p>', '<p>A<p>'))
        self.assertFalse(html_problems('<p class="x">가</p>', '<P class="x">A</P>'))

    def test_markdown_omissions_and_external_links(self):
        source = '# 제목\n\n본문\n\n[자료](https://example.org/a)\n\n결론'
        target = '# Title\n\nText\n\n[Source](https://example.org/b)'
        problems = markdown_problems(source, target)
        self.assertTrue(any('structure' in p for p in problems))
        self.assertTrue(any('destinations' in p for p in problems))

    def test_code_table_footnote_changes(self):
        cases = [('`x = 1`', '`x = 2`'),
                 ('a[^x]\n\n[^x]: note', 'a[^y]\n\n[^y]: note'),
                 ('| a | b |\n|---|---|\n| 1 | 2 |', '| a |\n|---|\n| 1 |')]
        for source, target in cases:
            self.assertTrue(markdown_problems(source, target))

    def test_protection_roundtrip(self):
        source = ('# 제목\n\n```python\nx = "한글"\n```\n\n'
                  '`한글`과 [자료](/연구)\n\n<a href="/자료">자료</a>\n\n'
                  '[자료][id]\n\n[id]: /research\n\n각주[^x]\n\n[^x]: 내용\n')
        masked, spans = protect_markdown(source)
        restored = restore_markdown(masked, spans)
        self.assertEqual(markdown_problems(source, restored), [])
        self.assertNotIn('x = "한글"', masked)
        self.assertNotIn('/연구', masked)

    def test_chunks_preserve_top_level_table_and_list(self):
        source = '# Title\n\n' + ('Paragraph.\n\n' * 12) + '- a\n- b\n\n| x |\n|---|\n| y |\n'
        chunks = markdown_chunks(source, 70)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(chunks), source)
        self.assertTrue(any('- a\n- b' in c for c in chunks))
        self.assertEqual(markdown_problems(source, '\n\n'.join(c.strip() for c in chunks)), [])

    def test_semantics_are_review_hints(self):
        issues = semantic_review('100명이 아니면 금지한다.', 'Unless there are 10 people, it is prohibited.')
        self.assertEqual({i['kind'] for i in issues}, {'numbers_dates_amounts', 'negation_conditions'})


class SharedExecution(unittest.TestCase):
    def test_cached_revalidation_and_correction(self):
        generate = Mock(side_effect=['bad', 'good'])
        store = Mock()
        result = translate_validated(generate=generate, parse=str, validate=lambda x: [] if x == 'good' else ['wrong'],
                                     cached='old', store=store)
        self.assertEqual(result, 'good')
        self.assertIn('wrong', generate.call_args.args[0])
        store.assert_called_once_with('good')
        generate.reset_mock()
        self.assertEqual(translate_validated(generate=generate, parse=str, validate=lambda x: [], cached='good'), 'good')
        generate.assert_not_called()

    def test_provider_failure_does_not_trigger_validation_retry(self):
        generate = Mock(side_effect=RuntimeError('quota'))
        with self.assertRaises(RuntimeError):
            translate_validated(generate=generate, parse=str, validate=lambda x: [], attempts=3)
        self.assertEqual(generate.call_count, 1)

    def test_transient_retry_and_truncated_rejection(self):
        with patch.object(registry, 'generate_detailed', side_effect=[
                registry.GenerationResult(error_kind='rate_limit', retry_after=0.01),
                registry.GenerationResult(text='ok')]) as gen, patch('time.sleep'):
            self.assertEqual(generate_translation('x', 'p', system='s'), 'ok')
            self.assertEqual(gen.call_count, 2)
        with patch.object(registry, 'generate_detailed', return_value=registry.GenerationResult(text='cut', truncated=True)) as gen:
            with self.assertRaises(TranslationProviderError):
                generate_translation('x', 'p', system='s')
            self.assertEqual(gen.call_count, 1)

    def test_markdown_resume_only_failed_chunk(self):
        source = '# 제목\n\n첫 문단.\n\n둘째 문단.'
        def translate(text, **kwargs):
            return text.replace('제목', 'Title').replace('첫 문단', 'First paragraph').replace('둘째 문단', 'Second paragraph')
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            calls = []
            def fail_last(text, **kwargs):
                calls.append(text)
                if '둘째' in text:
                    raise RuntimeError('network')
                return translate(text)
            with patch('scripts.translate_research_markdown._call_translator', side_effect=fail_last):
                with self.assertRaises(RuntimeError):
                    translate_markdown_with_retry(source, max_hangul_ratio=.03, cache_dir=cache, max_chars=12)
            with patch('scripts.translate_research_markdown._call_translator', side_effect=translate) as gen:
                result = translate_markdown_with_retry(source, max_hangul_ratio=.03, cache_dir=cache, max_chars=12)
                self.assertEqual(gen.call_count, 1)
                self.assertIn('Second paragraph', result)

    def test_protected_span_duplication_is_rejected(self):
        from scripts.translate_research_markdown import _translate_segment
        with patch('scripts.translate_research_markdown._call_translator', side_effect=lambda s, **k: s + s):
            with self.assertRaises(TranslationCallError):
                _translate_segment('한글 `code`', max_hangul_ratio=.03, attempts=1)


class Archival(unittest.TestCase):
    def test_quota_stops_queued_chunks_and_never_writes_fragment(self):
        with tempfile.TemporaryDirectory() as d:
            blocks = [{'tag': 'p', 'lines': ['Приказ о мобилизации.']} for _ in range(3)]
            prepared = {'_docs': [{'offset': 0, 'blocks': blocks}], '_glossary': [],
                        '_chunks': [[(i, b)] for i, b in enumerate(blocks)],
                        '_lang': core.RUSSIAN, 'chunks': 3}
            output = Path(d) / 'fragment.html'
            opts = core.Options(cache_path=Path(d) / 'cache.jsonl', out_path=output, concurrency=1)
            with patch.object(core, 'plan', return_value=prepared), patch.object(core, '_tm_prefill', return_value={}), \
                 patch.object(core, 'preflight'), patch.object(registry, 'generate_detailed',
                    return_value=registry.GenerationResult(error_kind='quota', error='no credit', attempts=1)) as gen:
                result = core.run({'id': 'test'}, opts)
            self.assertEqual(gen.call_count, 1)
            self.assertEqual(len(result['failures']), 3)
            self.assertFalse(output.exists())

    def test_explicit_glossary_survives_duplicate_person(self):
        people = {'people': [{'familyName': {'en': 'Hessen', 'ko': '게센'},
                             'name': {'en': 'Boris Hessen'}, 'givenName': {'en': 'Boris'}}]}
        with patch.object(Path, 'read_text', side_effect=[json.dumps(people), '[]']):
            glossary = core.build_glossary(Path('people'), Path('terms'), {'Hessen': '헤센'}, core.GERMAN)
        kept, _ = core.anchor_latin_people(glossary, 'Das Land Hessen')
        self.assertEqual([(g['ru'], g['ko']) for g in kept], [('Hessen', '헤센')])

    def test_duplicate_and_wrong_tag_are_rejected(self):
        chunk = [(0, {'tag': 'p', 'lines': ['Приказ о мобилизации.']})]
        for raw in ('[[0|p]]\n동원 명령.\n[[0|p]]\n명령.', '[[0|h3]]\n동원 명령.'):
            self.assertTrue(core.validate(chunk, core.parse_response(raw)))
        self.assertEqual(core.validate(chunk, core.parse_response('[[0|p]]\n동원 명령.')), [])

    def test_long_heading_does_not_create_empty_chunk(self):
        doc = {'offset': 0, 'blocks': [{'tag': 'h3', 'lines': ['Heading']}, {'tag': 'p', 'lines': ['x' * 100]}]}
        self.assertTrue(all(core.chunk_document(doc, 50)))

    def test_oversized_block_reassembles_and_keeps_partial_cache_separate(self):
        source = ' '.join(f'Приказ о мобилизации номер {i} получен.' for i in range(9))
        chunk = [(7, {'tag': 'p', 'lines': [source]})]
        with tempfile.TemporaryDirectory() as d:
            cache = core.Cache(Path(d) / 'doc.jsonl')
            stats = core.Stats()
            with patch('translation_runtime.generate_translation', return_value='[[7|p]]\n' + '동원 명령이 내려졌다. ' * 5) as gen:
                got = core._translate_chunk(chunk, [], cache, core.Options(max_chars=80), stats, lambda e: None)
                self.assertGreater(gen.call_count, 1)
            # Pieces count as provider calls, the chunk counts once.
            self.assertEqual((stats.translated, stats.failed), (1, 0))
            self.assertEqual(core.validate(chunk, got), [])
            self.assertEqual(len(cache.data), 1)
            self.assertTrue(cache.path.with_suffix('.parts.jsonl').is_file())
            with patch('translation_runtime.generate_translation') as gen:
                self.assertEqual(core._translate_chunk(chunk, [], cache, core.Options(max_chars=80),
                                                     core.Stats(), lambda e: None), got)
                gen.assert_not_called()

    def test_tm_conflict_and_rank(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / 'tm.db'
            for target in ('동지', '동무'):
                tm.record_segments([('товарищ', target)], lang_pair='ru-ko', doc_id=target,
                                   status='published', db_path=db)
            decisions = {}
            self.assertEqual(tm.exact_matches(['товарищ'], lang_pair='ru-ko', db_path=db,
                                              reject_conflicts=True, decisions=decisions), {})
            self.assertTrue(decisions['товарищ']['conflict'])
            tm.record_segments([('товарищ', '동지')], lang_pair='ru-ko', doc_id='review', status='reviewed', db_path=db)
            self.assertEqual(tm.exact_matches(['товарищ'], lang_pair='ru-ko', db_path=db,
                                              reject_conflicts=True), {'товарищ': '동지'})

    def test_tm_validation_and_document_exclusion(self):
        docs = [{'offset': 0, 'blocks': [{'tag': 'p', 'lines': ['Приказ о мобилизации.']}]}]
        with patch.object(tm, 'exact_matches', return_value={'Приказ о мобилизации.': 'სწორედ'}):
            self.assertEqual(core._tm_prefill(docs, core.RUSSIAN, lambda e: None), {})
        with patch.object(tm, 'exact_matches') as lookup:
            self.assertEqual(core._tm_prefill(docs, core.RUSSIAN, lambda e: None, {'tmReuse': {'enabled': False}}), {})
            lookup.assert_not_called()

    def test_context_is_opt_in_and_affects_key(self):
        chunk = [(1, {'tag': 'p', 'lines': ['Приказ.']})]
        a = core._prepare_chunk(chunk, [], core.Options())
        chunk[0][1]['sourceContext'] = {'title': 'Title', 'section': 'Section', 'before': 'Before', 'after': 'After'}
        b = core._prepare_chunk(chunk, [], core.Options())
        self.assertNotEqual(a[1], b[1])
        self.assertIn('번역·출력하지 말 것', b[0])


class Provenance(unittest.TestCase):
    def test_atomic_output_preserves_read_permissions(self):
        from translation_runtime.storage import atomic_write
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'page.html'
            atomic_write(path, 'first')
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            path.chmod(0o640)
            atomic_write(path, 'second')
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(path.read_text(), 'second')

    def test_manual_translation_is_preserved_when_source_is_unchanged(self):
        from translation_runtime.storage import source_hash
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source.md'
            source.write_text('# 원문')
            output = Path(d) / 'en'
            output.mkdir()
            target = output / source.name
            target.write_text('# Human edited title\n')
            target.with_suffix('.translation.json').write_text(json.dumps({
                'sourceHash': source_hash('# 원문'), 'targetHash': 'older machine output hash'}))
            with patch('scripts.translate_research_markdown.translate_markdown_with_retry') as gen:
                translate_one(source, output_dir=output, max_hangul_ratio=.03, force=False, dry_run=False)
                gen.assert_not_called()
            self.assertEqual(target.read_text(), '# Human edited title\n')

    def test_db_compare_and_swap(self):
        from scripts.translate_research_documents import _update_translation
        row = {'id': 7, 'title': '제목', 'slug': 'slug', 'markdown': '# 원문', 'content_sha256': 'stored-hash'}
        with patch('scripts.translate_research_documents.execute_returning_rowcount', return_value=0) as write:
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                _update_translation(row, '# Translation')
            self.assertIn("AND markdown = %s AND status = 'public'", write.call_args.args[0])
            self.assertEqual(write.call_args.args[1][-1], '# 원문')
            # The selection compares against the stored hash, so that is what gets written.
            self.assertEqual(write.call_args.args[1][3], 'stored-hash')

    def test_file_source_changed_during_generation(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source.md'
            output = Path(d) / 'en'
            source.write_text('# 원문')
            def changed(*args, **kwargs):
                source.write_text('# 수정')
                return '# Translation\n'
            with patch('scripts.translate_research_markdown.translate_markdown_with_retry', side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    translate_one(source, output_dir=output, max_hangul_ratio=.03, force=False, dry_run=False)
            self.assertFalse((output / source.name).exists())

    def test_source_upsert_invalidates_stale_translation(self):
        import research_store
        with patch.object(research_store, 'ensure_research_table'), patch.object(research_store, 'get_document', return_value={}), \
             patch.object(research_store, 'db_query_one', return_value={}) as query:
            research_store.upsert_document(filename='a.md', title='A', markdown='new')
            sql, params = query.call_args.args
            self.assertIn('research_documents.markdown = EXCLUDED.markdown', sql)
            self.assertIn('markdown_en_source_sha256', sql)
            self.assertEqual(sql.count('%s'), len(params))


class ReviewFixes(unittest.TestCase):
    def test_gemini_per_minute_429_is_transient(self):
        class GeminiError(Exception):
            code = 429
        exc = GeminiError("429 RESOURCE_EXHAUSTED. Quota exceeded for requests_per_minute. "
                          "Please check your plan and billing details.")
        self.assertEqual(registry._error_kind(exc), 'rate_limit')

        class OpenAIError(Exception):
            status_code = 429
        self.assertEqual(registry._error_kind(OpenAIError('insufficient_quota')), 'quota')

        class DeepSeekError(Exception):
            status_code = 402
        self.assertEqual(registry._error_kind(DeepSeekError('Insufficient Balance')), 'quota')

    def test_fence_roundtrip_is_exact(self):
        source = '# 제목\n\n```text\nLayer 3\n```\n\n각 레이어는 독립적이다.\n\n    indented\n\n끝.\n'
        self.assertEqual(restore_markdown(*protect_markdown(source)), source)

    def test_lenient_json_recovers_quotes_and_missing_brace(self):
        from scripts.translate_db_content import _parse_json_response
        quoted = '{\n  "title_en": "The "Boundary"",\n  "content_en": "He said "no".\\nNext line."\n}'
        self.assertEqual(_parse_json_response(quoted),
                         {'title_en': 'The "Boundary"', 'content_en': 'He said "no".\nNext line.'})
        unclosed = '{\n  "title_en": "Title",\n  "content_en": "Body that ends without a brace."'
        self.assertEqual(_parse_json_response(unclosed)['content_en'], 'Body that ends without a brace.')
        with self.assertRaises(ValueError):
            _parse_json_response('{"title_en": "Only title"}')

    def test_curation_source_title_may_stay_empty(self):
        from scripts.translate_db_content import _validate_translated_fields
        row = {'title': '제목', 'source_title': '원 제목', 'selection_rationale': '이유', 'context': '맥락'}
        translated = {'title_en': 'Title', 'source_title_en': '',
                      'selection_rationale_en': 'Reason', 'context_en': 'Context'}
        self.assertEqual(_validate_translated_fields(row, translated), [])
        self.assertTrue(_validate_translated_fields(row, {**translated, 'context_en': ''}))

    def test_archival_correction_is_korean(self):
        chunk = [(0, {'tag': 'p', 'lines': ['Приказ о мобилизации.']})]
        calls = []
        def fake(feature, prompt, *, system, **kwargs):
            calls.append(system)
            return '[[0|p]]\nПриказ о мобилизации.' if len(calls) == 1 else '[[0|p]]\n동원 명령.'
        with tempfile.TemporaryDirectory() as d:
            with patch('translation_runtime.generate_translation', side_effect=fake):
                core._translate_chunk(chunk, [], core.Cache(Path(d) / 'c.jsonl'), core.Options(),
                                      core.Stats(), lambda e: None)
        self.assertEqual(len(calls), 2)
        self.assertIn('직전 응답에 다음 문제가 있었다', calls[1])
        self.assertNotIn('placeholder', calls[1])


class Metering(unittest.TestCase):
    def test_discarded_budget_attempts_are_audited(self):
        profile = registry.CallSiteProfile(feature='test', provider='deepseek', model='test', max_tokens=100,
                                           extra={'thinking': {'type': 'enabled'}})
        def executor(p, *_):
            return registry._with_output_budget(p, lambda n: ('partial', {'tokens_out': n}, n == 100))
        with patch.dict(registry._EXECUTORS, {'deepseek': executor}), \
             patch('llm.gateway.check_llm_call'), patch('llm.gateway.record_llm_call') as audit:
            result = registry.generate_detailed('test', 'p', profile=profile)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.usage['tokens_out'], 300)
        self.assertEqual(audit.call_count, 2)
        self.assertFalse(result.truncated)

    def test_permanent_failure_is_not_retried(self):
        with patch.object(registry, 'generate_detailed', return_value=registry.GenerationResult(error_kind='quota')) as gen:
            with self.assertRaises(TranslationProviderError):
                generate_translation('test', 'p', system='s')
            self.assertEqual(gen.call_count, 1)


class DeepL(unittest.TestCase):
    def test_code_nodes_are_not_translated(self):
        from scripts import static_page_translation_pipeline as page
        source = '<p>설명</p><pre><code>한글 = 1</code></pre>'
        with patch.object(page, '_deepl_translate_texts', return_value=['Description']) as gen:
            result = page._translate_html_segments(source, api_key='dummy', api_base='http://unused',
                                                   target_lang='EN', source_lang='KO')
        self.assertEqual(gen.call_args.args[0], ['설명'])
        self.assertIn('<code>한글 = 1</code>', result)
        self.assertEqual(html_problems(source, result), [])

    def test_inline_nodes_share_paragraph_context_and_keep_tags(self):
        from scripts import static_page_translation_pipeline as page
        source = '<p>노동자는 <strong>조건</strong>에 동의한다.</p>'
        def translate(texts, **kwargs):
            self.assertIn('노동자는', kwargs['context'])
            self.assertIn('동의한다', kwargs['context'])
            return ['Workers ', 'conditions', ' agree.']
        with patch.object(page, '_deepl_translate_texts', side_effect=translate) as gen:
            result = page._translate_html_segments(source, api_key='dummy', api_base='http://unused',
                                                   target_lang='EN', source_lang='KO')
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(html_problems(source, result), [])


class Evaluation(unittest.TestCase):
    def test_frozen_sources_hashes_and_missing_candidates(self):
        from scripts.evaluate_translation import FIXTURE, evaluate
        samples = json.loads(FIXTURE.read_text())['samples']
        self.assertEqual(len(samples), 16)
        results = evaluate(samples, {})
        self.assertTrue(all(r['missing'] for r in results))


if __name__ == '__main__':
    unittest.main()
