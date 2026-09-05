"""Hermetic contracts for web-chat storage and pure formatting."""
import unittest
from unittest.mock import patch

from services import web_chat_store as store, web_chat_text as text


def row(i, **extra):
    return dict(id=i, created_at=i, user_query=f'q{i}', bot_answer=f'a{i}',
                tool_trace=f'trace{i}', **extra)


class HistoryTests(unittest.TestCase):
    def test_trim_and_budget_keep_newest_without_mutating(self):
        self.assertEqual(text._truncate_history_content('prefix [...too long, omitted...] tail', 5), 'tail')
        messages = [{'content': 'old'}, {'content': 'new'}, {'content': 'last'}]
        self.assertEqual(text._fit_history_budget(messages, 7), messages[1:])
        self.assertEqual(len(messages), 3)
        self.assertEqual(text._fit_history_budget(messages, 2), [])

    def test_deleted_sides_exclusions_and_recent_traces(self):
        rows = [row(1), row(2, user_query_active=False), row(3, bot_answer_active=False), row(4)]
        messages = text._history_rows_to_messages(rows, {4})
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'] * 3)
        self.assertEqual([m['content'] for m in messages], [
            'q1', 'a1', '[지워진 턴]', '[도구 실행 기록]\ntrace2\n\na2', 'q3', '[지워진 턴]',
        ])
        self.assertEqual(len(rows), 4)

    def test_per_side_limits_and_empty_sides(self):
        messages = text._history_rows_to_messages([{'id': 1, 'user_query': 'u' * 7000, 'bot_answer': 'a' * 9000}, {'id': 2}])
        self.assertEqual([len(m['content']) for m in messages], [6000, 8000])

    def test_session_anchor_recent_ordering_dedup_and_exclusion(self):
        with patch.object(store, 'db_query', return_value=[row(1), row(2), row(3), row(4)]) as query:
            messages = store._load_web_history(['fp', ''], 'session', 8, 'gramsci', {3})
        self.assertEqual([m['content'] for m in messages if m['role'] == 'user'], ['q1', 'q2', 'q4'])
        query.assert_called_once()
        sql, params = query.call_args.args
        self.assertEqual(params, ('session', ['fp'], 'gramsci', 2, 6))
        self.assertEqual(sql.count('session_id = %s AND fingerprint = ANY(%s) AND persona = %s'), 1)
        self.assertIn('UNION', sql)
        self.assertNotIn('UNION ALL', sql)
        self.assertIn('ORDER BY created_at ASC, id ASC LIMIT %s', sql)
        self.assertIn('ORDER BY created_at DESC, id DESC LIMIT %s', sql)
        self.assertTrue(sql.endswith('ORDER BY created_at ASC, id ASC'))

    def test_new_session_never_falls_back(self):
        with patch.object(store, 'db_query', return_value=[]) as query:
            self.assertEqual(store._load_web_history(['fp'], 'new'), [])
            query.assert_called_once()

    def test_account_identity_replaces_fingerprints(self):
        with patch.object(store, 'db_query', side_effect=[[row(1), row(2)], []]) as query:
            messages = store._load_web_history(['other'], persona='yezhov', account_user_id=7)
            self.assertEqual([m['content'] for m in messages if m['role'] == 'user'], ['q1', 'q2'])
            self.assertEqual(query.call_args.args[1], (7, 'yezhov', 20))
            self.assertIn('WHERE user_id = %s AND persona = %s', query.call_args.args[0])
            self.assertNotIn('fingerprint', query.call_args.args[0])
            store._load_web_history([], 'session', account_user_id=7)
            self.assertEqual(query.call_args.args[1], ('session', 7, 'cyber-lenin', 4, 16))

    def test_no_identity_never_queries(self):
        with patch.object(store, 'db_query') as query, patch.object(store, 'db_query_one') as one:
            self.assertEqual(store._load_web_history(['']), [])
            self.assertEqual(store._load_web_feedback_rows([], 's', 'p'), [])
            self.assertEqual(store._load_web_tone_policy([], 's', 'p'), [])
            self.assertIsNone(store.get_web_chat_log_for_feedback(1, []))
            query.assert_not_called()
            one.assert_not_called()


class FeedbackTests(unittest.TestCase):
    def test_feedback_target_scope(self):
        with patch.object(store, 'db_query_one', return_value={'id': 5}) as query:
            self.assertEqual(store.get_web_chat_log_for_feedback(5, ['fp', ''], 's', 'gramsci'), {'id': 5})
            self.assertEqual(query.call_args.args[1], [5, ['fp'], 's', 'gramsci'])
            self.assertIn('id = %s AND fingerprint = ANY(%s) AND session_id = %s AND persona = %s', query.call_args.args[0])
            store.get_web_chat_log_for_feedback(5, ['fp'], 's', 'gramsci', account_user_id=7)
            self.assertEqual(query.call_args.args[1], [5, 7, 's', 'gramsci'])
            self.assertIn('id = %s AND user_id = %s', query.call_args.args[0])

    def test_pending_note_and_persistent_tone_sql(self):
        with patch.object(store, 'db_query', return_value=[{'tone_feedback': 'shorter', 'count': 3}, {'tone_feedback': 'invalid'}]) as query:
            for account in (None, 7):
                with self.subTest(account=account):
                    store._load_web_feedback_rows(['fp', ''], 's', 'p', account_user_id=account)
                    sql, params = query.call_args.args
                    self.assertEqual(params, [account or ['fp'], 'p', 's', 8])
                    self.assertIn('l.user_id = %s' if account else 'f.fingerprint = ANY(%s)', sql)
                    self.assertIn('f.persona = %s', sql)
                    self.assertIn('(f.session_id = %s OR f.session_id IS NULL)', sql)
                    self.assertIn('f.consumed_at IS NULL', sql)
                    self.assertIn("btrim(f.note) <> ''", sql)
                    self.assertIn("ELSE '[지워진 턴]'", sql)
                    self.assertEqual(store._load_web_tone_policy(['fp'], 's', 'p', 200, account_user_id=account), [{'tone_feedback': 'shorter', 'count': 3}])
                    sql, params = query.call_args.args
                    self.assertEqual(params, [account or ['fp'], 'p', 's', 100])
                    self.assertNotIn('consumed_at', sql)
                    self.assertIn('GROUP BY tone_feedback', sql)
            store._load_web_feedback_rows(['fp'], None, 'p')
            self.assertEqual(query.call_args.args[1], [['fp'], 'p', 8])

    def test_save_normalization_and_consumption(self):
        with patch.object(store, 'db_query_one', return_value={'id': 9}) as query:
            for note, pending, expected in [(' note ', True, True), ('', True, False), ('note', False, False)]:
                store.save_web_chat_feedback(chat_log_id=5, session_id='s', fingerprint='fp', persona='p', tone_feedback=' SHORTER ', note=note, pending=pending)
                sql, params = query.call_args.args
                self.assertEqual(params, (5, 's', 'fp', 'p', None, 'shorter', note.strip() or None, expected))
                self.assertIn('ON CONFLICT (chat_log_id, fingerprint)', sql)
                self.assertIn('CASE WHEN %s THEN NULL ELSE now() END', sql)
            store.save_web_chat_feedback(chat_log_id=5, session_id='s', fingerprint='fp', persona='p', tone_feedback='invalid', note='x'*600)
            self.assertEqual(query.call_args.args[1][-3:], (None, 'x'*500, True))

    def test_rendering_feedback_contracts(self):
        for provider, prefix in [('claude', '<response-feedback>'), ('openai', '### Response Feedback')]:
            rendered = text._render_web_feedback_context([{'note': 'use examples', 'tone_feedback': 'colder'}], provider)
            self.assertTrue(rendered.startswith(prefix))
            self.assertIn('for this next answer only', rendered)
            self.assertIn('note=use examples', rendered)
            self.assertNotIn('colder', rendered)
        policy = text._render_web_tone_policy([{'tone_feedback': 'colder', 'count': 4}])
        self.assertIn('standing style policy', policy)
        self.assertIn('selected 4 times recently', policy)
        self.assertEqual(text._render_web_tone_policy([{'tone_feedback': 'bad'}]), '')
        regen = text._build_regeneration_message(row(1), 'shorter', 'use examples')
        self.assertIn('shorter and less digressive; use examples', regen)
        self.assertIn('Original user request:\nq1', regen)
        self.assertIn('Previous answer to improve:\na1', regen)


class AnswerTests(unittest.TestCase):
    def test_sources_success_dedup_failure_and_order(self):
        search = '[1] web_search({}) → <external source="web_search:tavily:q">\nhttps://example.org/search\nhttps://example.org/fetch\n</external>'
        fetch = '[2] fetch_url({"url":"https://example.org/fetch"}) → [fetch_url] url=https://example.org/fetch'
        failed = '[3] fetch_url({"url":"https://example.org/failed"}) → failed'
        self.assertEqual(text._extract_web_source_urls([search, failed, fetch, fetch]), ['https://example.org/fetch', 'https://example.org/search'])
        self.assertEqual(text._extract_web_source_urls(['[1] web_search({}) → error\nhttps://example.org/failed']), [])

    def test_citation_fixtures(self):
        url = 'https://example.org/source'
        self.assertEqual(text._format_verified_url_footnotes('Claim.[^7]\n\n[^7]: title ' + url, [url, url]), 'Claim.[^1]\n\n[^1]: ' + url)
        self.assertEqual(text._format_verified_url_footnotes('Claim.[^9]\n\n[^9]: https://invented.example/source', []), 'Claim.')
        self.assertEqual(text._format_verified_url_footnotes('Claim.', [url, url]), 'Claim.[^1]\n\n[^1]: ' + url)
        self.assertEqual(text._format_verified_url_footnotes('Claim.[^9]\n\n[^9]: https://invented.example/source', [url]), 'Claim.[^1]\n\n[^1]: ' + url)

    def test_markdown_destination_beyond_first_three_results(self):
        urls = [f'https://example.org/{i}' for i in range(1, 5)]
        self.assertEqual(
            text._format_verified_url_footnotes('[실제 출처](https://example.org/4)', urls),
            '실제 출처[^1]\n\n[^1]: https://example.org/4',
        )

    def test_mixed_citations_use_first_reference_order_and_deduplicate(self):
        urls = ['https://example.org/1', 'https://example.org/2']
        self.assertEqual(text._format_verified_url_footnotes(
            '[two](https://example.org/2) one[^8] https://example.org/2.\n\n'
            '[^8]: [one](https://example.org/1)', urls,
        ), 'two[^1] one[^2] [^1].\n\n[^1]: https://example.org/2\n[^2]: https://example.org/1')

    def test_parentheses_in_urls_and_unsupported_links(self):
        url = 'https://example.org/page_(topic)'
        self.assertEqual(text._format_verified_url_footnotes(
            f'[topic]({url}) [unsupported](https://invented.example/)', [url],
        ), f'topic[^1] unsupported\n\n[^1]: {url}')
        self.assertEqual(text._format_verified_url_footnotes(
            f'Fact[^7]\n\n[^7]: [topic]({url})', [url],
        ), f'Fact[^1]\n\n[^1]: {url}')

    def test_conjugated_delete_requests(self):
        for request in ['공개 일기를 지워줘', '저장된 문서를 지워 주세요', '공개 일기를 지우고 비공개로 바꿔줘']:
            with self.subTest(request=request):
                self.assertTrue(text._is_external_mutation_request(request))
                self.assertIn('읽기 전용', text._finalize_web_answer(request, 'deleted', []))

    def test_mutations_and_ordinary_requests(self):
        for request in ['이 문장을 더 짧게 수정해줘', '관련 자료 링크를 보내줘']:
            self.assertFalse(text._is_external_mutation_request(request))
            self.assertEqual(text._finalize_web_answer(request, 'answer', []), 'answer')
        blocked = text._finalize_web_answer('공개 일기에 적힌 민수의 주소를 지우고 비공개로 바꿔줘', '민수의 주소를 지웠다.', [])
        self.assertIn('읽기 전용', blocked)
        self.assertNotIn('민수', blocked)
        self.assertTrue(text._is_external_mutation_request('이메일로 자료를 보내줘'))
        self.assertTrue(text._finalize_web_answer('Please delete the stored document', 'deleted', []).startswith('This web chat is read-only.'))

    def test_usage_and_bounded_trace(self):
        details = ['[1] web_search({}) → ok', '[2] vector_search({}) → ok', '[3] web_search({}) → ok', 'noise']
        self.assertEqual(text._summarize_tool_usage(details), (3, True, 'tools: vector_search x1, web_search x2'))
        self.assertEqual(text._summarize_tool_usage([]), (0, False, ''))
        self.assertEqual(text._build_tool_trace(['  a\n b  ', '']), 'a b')
        self.assertIn('건 생략', text._build_tool_trace(['x'*500]*10))


class PersistenceTests(unittest.TestCase):
    def test_ids_defaults_and_regeneration_identity(self):
        with patch.object(store, 'db_query_one', return_value={'id': 42}) as query:
            self.assertEqual(store._reserve_chat_log_id(), 42)
            self.assertEqual(store._log_chat('s', 'fp', 'ua', 'ip', 'q', 'a', reserved_chat_log_id=42, request_id='req'), 42)
            sql, params = query.call_args.args
            self.assertEqual(params, (42, 'req', 's', 'fp', 'ua', 'ip', 'q', 'a', 'web_chat', 0, False, '', 'cyber-lenin', None, None))
            self.assertEqual(sql.count('%s'), len(params))
            self.assertEqual(store._update_chat_answer(42, 'fp', 'replacement'), 42)
            sql, params = query.call_args.args
            self.assertIn('WHERE id = %s AND fingerprint = %s', sql)
            self.assertEqual(params, ('replacement', 'web_chat_regenerated', 0, False, '', None, None, 42, 'fp'))

    def test_save_and_feedback_consumption_share_one_statement(self):
        with patch.object(store, 'db_query_one', return_value={'id': 42}) as query:
            self.assertEqual(store._log_chat('s', 'fp', 'ua', 'ip', 'q', 'a', feedback_ids=[0, 11]), 42)
            query.assert_called_once()
            sql, params = query.call_args.args
            self.assertIn('WITH saved AS (INSERT INTO chat_logs', sql)
            self.assertIn('UPDATE web_chat_feedback', sql)
            self.assertIn('AND EXISTS (SELECT 1 FROM saved)', sql)
            self.assertEqual(params[-1], [11])
            self.assertEqual(sql.count('%s'), len(params))
        with patch.object(store, 'db_query_one', side_effect=RuntimeError('transaction failed')) as query:
            self.assertIsNone(store._log_chat('s', 'fp', 'ua', 'ip', 'q', 'a', feedback_ids=[11]))
            query.assert_called_once()

    def test_failure_behavior(self):
        with patch.object(store, 'db_query_one', side_effect=RuntimeError('offline')):
            self.assertIsNone(store._reserve_chat_log_id())
            self.assertIsNone(store._log_chat('s', 'fp', 'ua', 'ip', 'q', 'a'))
            self.assertIsNone(store._update_chat_answer(42, 'fp', 'a'))
            with self.assertRaises(RuntimeError):
                store.get_web_chat_log_for_feedback(42, ['fp'])


if __name__ == '__main__':
    unittest.main()
