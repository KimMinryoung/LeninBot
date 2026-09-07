"""Offline regressions for publication provenance and scheduled translation work."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from runtime_tools.archival_translation import core
from translation_runtime import TranslationCallError
from translation_runtime.batch_state import BatchState
from translation_runtime.structure import markdown_problems, translate_oversized_markdown


class AssemblyProvenance(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.block = {'tag': 'p', 'lines': ['Приказ о мобилизации.']}
        self.chunk = [(0, self.block)]
        self.prepared = {'_docs': [{'offset': 0, 'blocks': [self.block]}],
                         '_chunks': [self.chunk], '_glossary': [], '_lang': core.RUSSIAN}
        self.opts = core.Options(cache_path=self.root / 'cache.jsonl', out_path=self.root / 'out.html')
        self.spec = {'id': 'test'}
        self.cache = core.Cache(self.opts.cache_path)
        self.key = core._prepare_chunk(self.chunk, [], self.opts)[1]
        self.addCleanup(patch.stopall)
        patch.object(core, 'plan', return_value=self.prepared).start()
        patch.object(core, 'assemble', side_effect=lambda spec, docs, translated: '\n'.join(translated[0])).start()
        self.tm = patch.object(core, '_tm_prefill', return_value={}).start()

    def seed(self, *, legacy=False, key=None, text='동원 명령.'):
        self.cache.put(key or self.key, {0: [text]}, {} if legacy else {'sourceHashes': core._block_source_hashes(self.chunk)})

    def test_legacy_exact_key_is_upgraded_without_generation(self):
        self.seed(legacy=True)
        with patch('translation_runtime.generate_translation') as generate:
            core.reassemble(self.spec, self.opts)
        generate.assert_not_called()
        record = core.Cache(self.opts.cache_path).get(self.key)
        self.assertEqual(record['sourceHashes'], core._block_source_hashes(self.chunk))

    def test_unknown_legacy_key_cannot_overwrite_output(self):
        self.seed(legacy=True, key='old-model')
        self.opts.out_path.write_text('published')
        with self.assertRaisesRegex(core.SpecError, '원문 일치'):
            core.reassemble(self.spec, self.opts)
        self.assertEqual(self.opts.out_path.read_text(), 'published')

    def test_source_hash_allows_model_change(self):
        self.seed(key='old-model')
        core.reassemble(self.spec, self.opts)
        self.assertEqual(self.opts.out_path.read_text(), '동원 명령.')

    def test_changed_source_same_number_is_rejected(self):
        self.seed()
        core.reassemble(self.spec, self.opts)
        self.block['lines'] = ['Приказ об отмене мобилизации.']
        with self.assertRaisesRegex(core.SpecError, '원문이'):
            core.reassemble(self.spec, self.opts)
        self.assertEqual(self.opts.out_path.read_text(), '동원 명령.')

    def test_publication_pins_tm_and_protects_manual_edits(self):
        self.seed()
        self.tm.return_value = {0: ['동원에 관한 명령.']}
        core.reassemble(self.spec, self.opts)
        self.tm.reset_mock()
        self.tm.return_value = {0: ['새 동원 명령.']}
        core.reassemble(self.spec, self.opts)
        self.tm.assert_not_called()
        self.assertEqual(self.opts.out_path.read_text(), '동원에 관한 명령.')
        self.opts.out_path.write_text('manual change')
        with self.assertRaisesRegex(core.SpecError, 'snapshot'):
            core.reassemble(self.spec, self.opts)
        self.assertEqual(self.opts.out_path.read_text(), 'manual change')

    def test_latest_append_wins_even_for_repeated_key(self):
        self.seed(key='a', text='처음 동원 명령.')
        self.seed(key='b', text='중간 동원 명령.')
        self.seed(key='a', text='최종 동원 명령.')
        core.reassemble(self.spec, self.opts)
        self.assertEqual(self.opts.out_path.read_text(), '최종 동원 명령.')


class BatchHistory(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'failures'
        self.now = 100
        self.state = BatchState(self.path, now=lambda: self.now)

    def test_reads_do_not_create_state(self):
        self.assertFalse(self.state.deferred('x', 'hash'))
        self.assertFalse(self.path.exists())

    def test_validation_cooldown_is_bounded_and_clears_on_success(self):
        for _ in range(2):
            self.state.failed('x', 'old', TranslationCallError('invalid'))
            self.assertFalse(self.state.deferred('x', 'old'))
        self.state.failed('x', 'old', TranslationCallError('invalid'))
        self.assertTrue(self.state.deferred('x', 'old'))
        self.assertFalse(self.state.deferred('x', 'new'))
        self.now += 48 * 3600 + 1
        self.assertFalse(self.state.deferred('x', 'old'))
        self.state.succeeded('x', 'old')
        self.assertEqual(self.state.get('x'), {})

    def test_provider_and_runtime_errors_never_delay_next_batch(self):
        from translation_runtime import TranslationProviderError
        from llm.call_registry import GenerationResult
        for error in [RuntimeError('DB unavailable')] + [TranslationProviderError(
                GenerationResult(error_kind=kind)) for kind in ('quota', 'authentication', 'rate_limit', 'network')]:
            with self.subTest(error=error):
                for _ in range(3):
                    self.state.failed('x', 'hash', TranslationCallError('invalid'))
                self.state.failed('x', 'hash', error)
                self.assertFalse(self.state.deferred('x', 'hash'))
                self.assertEqual(self.state.get('x'), {})

    def test_items_are_independent_and_invalid_state_is_disposable(self):
        for _ in range(3):
            self.state.failed('a', 'hash', TranslationCallError('invalid'))
        self.state.succeeded('b', 'hash')
        self.assertTrue(self.state.deferred('a', 'hash'))
        self.state._file('a').write_text('invalid json')
        self.assertFalse(self.state.deferred('a', 'hash'))

    def test_selection_filters_in_sql_without_reading_local_state(self):
        from scripts import translate_db_content as db
        cur = Mock()
        cur.fetchall.return_value = []
        conn = Mock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cur)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)
        with patch.object(db, 'BatchState') as state:
            db._select_rows(conn, 'diary', 'ai_diary', ids=[], limit=2, force=False)
            state.assert_not_called()
        sql, params = cur.execute.call_args.args
        self.assertIn('translation_source_sha256 IS DISTINCT FROM', sql)
        self.assertIn('LIMIT %s', sql)
        self.assertEqual(params, [2])

    def test_translation_and_hash_are_written_in_same_conditional_update(self):
        from scripts import translate_db_content as db
        cur = Mock(rowcount=1)
        conn = Mock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cur)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)
        row = {'id': 1, 'title': '제목', 'content': '내용'}
        db._update_row(conn, 'diary', 'ai_diary', 1, {'title_en': 'Title', 'content_en': 'Body'}, row)
        sql, params = cur.execute.call_args.args
        self.assertIn('translation_source_sha256 = ' + db._source_hash_sql('diary'), sql)
        self.assertIn('content IS NOT DISTINCT FROM %s', sql)
        conn.commit.assert_called_once()
        cur.rowcount = 0
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            db._update_row(conn, 'diary', 'ai_diary', 1, {'title_en': 'Title', 'content_en': 'Body'}, row)

    def test_deferred_row_does_not_consume_batch_limit(self):
        from scripts import translate_db_content as db
        rows = [{'id': i, 'title': '제목', 'content': '내용', 'translation_current_sha256': 'hash'} for i in (1, 2)]
        for _ in range(3):
            self.state.failed('diary:1', rows[0]['translation_current_sha256'], TranslationCallError('invalid'))
        with patch.object(db, 'BatchState', return_value=self.state), \
             patch.object(db, '_load_frontend_env', return_value={}), patch.object(db, '_connect_db'), \
             patch.object(db, '_select_rows', return_value=rows), \
             patch.object(db, '_call_translator', return_value={'title_en': 'Title', 'content_en': 'Body'}) as translate, \
             patch.object(db, '_update_row'), patch.object(db, '_record_tm'):
            changed, _, failures = db.translate_target('diary', ids=[], limit=1, force=False, dry_run=False, select_only=False)
        self.assertEqual(changed, 1)
        self.assertEqual(translate.call_args.args[0]['id'], 2)
        self.assertEqual(len(failures), 1)

    def test_batch_wrapper_runs_second_job_after_first_failure(self):
        from scripts.run_translation_batch import main
        with patch('scripts.run_translation_batch.subprocess.run', side_effect=[Mock(returncode=1), Mock(returncode=0)]) as run:
            self.assertEqual(main(), 1)
        self.assertEqual(run.call_count, 2)


class JsonAmbiguity(unittest.TestCase):
    def test_duplicate_keys_and_embedded_fake_keys_are_rejected(self):
        from scripts._translation_common import parse_json_object
        for raw in (
            '{"title_en":"one","title_en":"two","content_en":"Body"}',
            '{"title_en":"Title", "content_en":"Example: "title_en": "another", "content_en": "oops"}',
            '{"content_en":"He said "no"", "title_en":"Title"}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json_object(raw, keys=['title_en', 'content_en'])

    def test_valid_json_examples_are_preserved(self):
        from scripts._translation_common import parse_json_object
        data = {'title_en': 'Title', 'content_en': 'Example: "title_en": "value"'}
        self.assertEqual(parse_json_object(json.dumps(data), keys=list(data)), data)


class MarkdownSubdivision(unittest.TestCase):
    def check_source(self, source):
        calls = []
        def translate(text):
            calls.append(text)
            return text.replace('내용', 'Body').replace('제목', 'Title')
        result = translate_oversized_markdown(source, translate, 95)
        self.assertGreater(len(calls), 1)
        self.assertEqual(markdown_problems(source, result), [])
        self.assertEqual(result.count('Body'), source.count('내용'))
        self.assertNotIn('내용', result)
        return calls, result

    def test_table_repeats_header_but_assembles_it_once(self):
        source = '| 제목 | N |\n|---|---|\n' + ''.join(f'| 내용 {i} | {i} |\n' for i in range(30))
        calls, result = self.check_source(source)
        self.assertTrue(all(c.startswith('| 제목 | N |') for c in calls))
        self.assertEqual(result.count('Title'), 1)

    def test_tight_loose_and_nested_lists(self):
        for separator in ('\n', '\n\n'):
            for item in ('- 내용 {i}', '{i}. 내용 {i}', '- 내용 {i}\n  - nested item'):
                with self.subTest(separator=separator, item=item):
                    source = separator.join(item.format(i=i) for i in range(1, 30)) + '\n'
                    self.check_source(source)

    def test_dropped_row_is_rejected(self):
        source = '| 제목 |\n|---|\n' + '| 내용 |\n' * 30
        with self.assertRaises(ValueError):
            translate_oversized_markdown(source, lambda text: '\n'.join(text.splitlines()[:-1]), 95)

    def test_production_adapter_subdivides_and_reuses_successful_leaves(self):
        from scripts.translate_research_markdown import translate_markdown_with_retry
        source = '| 제목 |\n|---|\n' + ''.join(f'| 내용 {i} |\n' for i in range(30))
        with tempfile.TemporaryDirectory() as directory:
            def translate(text, **kwargs):
                return text.replace('내용', 'Body').replace('제목', 'Title')
            with patch('scripts.translate_research_markdown._call_translator', side_effect=translate) as call:
                output = translate_markdown_with_retry(source, max_hangul_ratio=.03, max_chars=95, cache_dir=Path(directory))
                self.assertGreater(call.call_count, 1)
            with patch('scripts.translate_research_markdown._call_translator') as call:
                again = translate_markdown_with_retry(source, max_hangul_ratio=.03, max_chars=95, cache_dir=Path(directory))
                call.assert_not_called()
            self.assertEqual(output, again)


class EvaluationAdapters(unittest.TestCase):
    def test_pipeline_masks_code_and_accounts_for_correction(self):
        from scripts.evaluate_translation import generate_candidate
        from llm.call_registry import GenerationResult
        sample = {'id': 'test', 'kind': 'markdown', 'lang': 'ko', 'source': '# 제목\n\n내용 `한글코드`\n'}
        prompts = []
        def generate(feature, prompt, **kwargs):
            prompts.append(prompt)
            text = 'bad output' if len(prompts) == 1 else prompt.replace('제목', 'Title').replace('내용', 'Body')
            return GenerationResult(text=text, attempts=1, usage={'tokens_in': 10, 'tokens_out': 20})
        with patch('llm.call_registry.generate_detailed', side_effect=generate):
            result = generate_candidate(sample)
        self.assertIsNone(result['error'])
        self.assertNotIn('한글코드', prompts[0])
        self.assertIn('`한글코드`', result['text'])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['usage']['tokens_out'], 40)
        self.assertNotIn('adapterHashes', result)
        self.assertTrue(result['profile']['model'])

    def test_archival_split_keeps_frozen_glossary_and_register(self):
        from scripts.evaluate_translation import generate_candidate
        from llm.call_registry import GenerationResult
        block = {'tag': 'p', 'lines': [' '.join(f'Приказ о мобилизации номер {i} получен.' for i in range(8))]}
        source = core.render_chunk([(0, block)])
        prefix = '용어표: Приказ → 명령\n문체: 합쇼체\n아래 단락들을 번역하라.\n\n'
        sample = {'id': 'test', 'kind': 'archival', 'lang': 'ru', 'source': source,
                  'prompt': prefix + source, 'blocks': [(0, block)]}
        prompts = []
        def generate(feature, prompt, **kwargs):
            prompts.append(prompt)
            return GenerationResult(text='[[0|p]]\n' + '동원 명령이 내려졌습니다. ' * 4, attempts=1)
        with patch('llm.call_registry.generate_detailed', side_effect=generate):
            result = generate_candidate(sample, max_chars=80)
        self.assertIsNone(result['error'])
        self.assertGreater(len(prompts), 1)
        self.assertTrue(all(prompt.startswith(prefix) for prompt in prompts))

    def test_archival_chunk_target_repartitions_frozen_blocks(self):
        from scripts.evaluate_translation import generate_candidate
        from llm.call_registry import GenerationResult
        # 두 블록의 내용이 같으면 위치 독립 캐시 키(2026-09-07)가 둘째 청크를 캐시 적중으로
        # 처리해 호출이 한 번이 된다. 재분할을 보려는 테스트이므로 내용을 다르게 둔다.
        blocks = [(10, {'tag': 'p', 'lines': ['Приказ о мобилизации номер один получен.']}),
                  (11, {'tag': 'p', 'lines': ['Приказ о мобилизации номер два получен.']})]
        source = core.render_chunk(blocks)
        sample = {'id': 'test', 'kind': 'archival', 'lang': 'ru', 'source': source,
                  'prompt': '용어표\n\n' + source, 'blocks': blocks}
        def generate(feature, prompt, **kwargs):
            marker = '[[10|p]]' if '[[10|p]]' in prompt else '[[11|p]]'
            self.assertEqual(prompt.count('[[1'), 1)
            return GenerationResult(text=marker + '\n동원 명령이 내려졌습니다.', attempts=1)
        with patch('llm.call_registry.generate_detailed', side_effect=generate) as call:
            result = generate_candidate(sample, max_chars=45)
        self.assertIsNone(result['error'])
        self.assertEqual(call.call_count, 2)
        self.assertIn('[[11|p]]', result['text'])

    def test_evaluation_metadata_is_shared_and_rescoring_keeps_its_origin(self):
        from scripts import evaluate_translation as evaluation
        fixture = {'samples': [{'id': 'one', 'lang': 'ko', 'kind': 'markdown', 'source': 'text',
                               'sourceHash': __import__('hashlib').sha256(b'text').hexdigest()}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / 'fixture.json', root / 'result.json'
            inputs.write_text(json.dumps(fixture))
            with patch('sys.argv', ['evaluate', '--fixture', str(inputs), '--generate', '--output', str(output)]), \
                 patch.object(evaluation, 'generate_candidate', return_value={'text': 'text'}), \
                 patch.object(evaluation, 'adapter_hashes', return_value={'adapter': 'original'}) as hashes:
                self.assertEqual(evaluation.main(), 0)
                hashes.assert_called_once()
            report = json.loads(output.read_text())
            self.assertEqual(report['adapterHashes'], {'adapter': 'original'})
            self.assertNotIn('adapterHashes', report['candidates']['one'])
            with patch('sys.argv', ['evaluate', '--fixture', str(inputs), '--candidates', str(output), '--output', str(output)]), \
                 patch.object(evaluation, 'adapter_hashes') as hashes:
                self.assertEqual(evaluation.main(), 0)
                hashes.assert_not_called()
            self.assertEqual(json.loads(output.read_text())['adapterHashes'], {'adapter': 'original'})

    def test_model_mode_is_an_unprocessed_single_call(self):
        from scripts.evaluate_translation import generate_candidate
        from llm.call_registry import GenerationResult
        sample = {'id': 'test', 'kind': 'markdown', 'lang': 'ko', 'source': '내용 `한글코드`'}
        with patch('llm.call_registry.generate_detailed', return_value=GenerationResult(text='bad output')) as generate:
            result = generate_candidate(sample, mode='model')
        self.assertIn('한글코드', generate.call_args.args[1])
        self.assertEqual(result['text'], 'bad output')
        self.assertEqual(generate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
