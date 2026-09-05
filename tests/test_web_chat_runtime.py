"""Web-chat preparation, provider adapters and persistence failures without I/O."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.web_chat_text import _normalize_web_result


class ResultTests(unittest.TestCase):
    def test_string_and_metadata_completion_defaults(self):
        tracker = {'rounds_used': 2, 'total_cost': 0.25}
        result = _normalize_web_result('answer', tracker)
        self.assertEqual(result, {
            'text': 'answer', 'complete': True, 'truncated': False,
            'finish_reason': None, 'continuations_used': 0, 'rounds': 2, 'cost_usd': 0.25,
        })
        self.assertEqual(_normalize_web_result({}, {})['text'], '')
        result = _normalize_web_result({
            'text': 'partial', 'complete': False, 'truncated': True, 'finish_reason': 'length',
            'continuations_used': 2, 'rounds': 4, 'cost_usd': 0.5, 'private_field': 'hidden',
        }, tracker)
        self.assertEqual(result['rounds'], 4)
        self.assertEqual(result['cost_usd'], 0.5)
        self.assertEqual(result['continuations_used'], 2)
        self.assertFalse(result['complete'])
        self.assertTrue(result['truncated'])
        self.assertNotIn('private_field', result)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch('db.query', return_value=[]))
        self.enterContext(patch('db.query_one', return_value=None))
        self.enterContext(patch('db.execute', return_value=None))
        from services import web_chat
        self.web = web_chat
        self.enterContext(patch.object(web_chat, 'db_query', return_value=[]))
        self.enterContext(patch('services.web_chat_store.db_query', return_value=[]))
        self.enterContext(patch('services.web_chat_store.db_query_one', return_value=None))
        self.enterContext(patch('services.web_chat_store.db_execute', return_value=None))

    def turn(self, provider):
        return self.web._PreparedWebChat(
            original_message='question', regeneration_source=None,
            history=[{'role': 'user', 'content': 'question'}],
            profile=SimpleNamespace(provider=provider, model_id='test-model', max_rounds=2,
                                    max_tokens=128, budget_usd=0.3, display_name='Test', tier='medium'),
            system_prompt='system', feedback_ids=[11], preflight_tool_detail='',
        )

    async def invoke(self, turn, tracker=None):
        return await self.web._invoke_web_model(
            turn, tools=[{'name': 'test'}], tool_handlers={'test': 'handler'},
            on_progress=AsyncMock(), budget_tracker=tracker if tracker is not None else {},
        )

    async def test_provider_specific_options_and_shared_limits(self):
        for provider in ('claude', 'openai', 'kimi', 'deepseek'):
            with self.subTest(provider=provider), \
                 patch.object(self.web, 'chat_with_tools', new_callable=AsyncMock, return_value='answer') as anth, \
                 patch('llm.openai_tool_loop.chat_with_tools', new_callable=AsyncMock, return_value={'text': 'answer'}) as oai, \
                 patch.object(self.web, '_kimi_client', object()), \
                 patch.object(self.web, '_deepseek_anthropic_client', object()), \
                 patch('llm.provider_failover.resolve_deepseek_failover_model', new_callable=AsyncMock, return_value=None), \
                 patch('llm.provider_registry.kimi_openai_tool_options', return_value={'reasoning_effort': 'max'}):
                turn = self.turn(provider)
                result = await self.invoke(turn)
                self.assertEqual(result['text'], 'answer')
                called = oai if provider in ('openai', 'kimi') else anth
                unused = anth if called is oai else oai
                called.assert_awaited_once()
                unused.assert_not_awaited()
                self.assertIs(called.call_args.args[0], turn.history)
                options = called.call_args.kwargs
                self.assertEqual((options['max_tokens'], options['max_rounds'], options['budget_usd']), (128, 2, 0.3))
                self.assertTrue(options['continue_on_length'])
                self.assertEqual(options['max_length_continuations'], 2)
                self.assertEqual(options['system_prompt'], 'system')
                if provider == 'deepseek':
                    self.assertEqual(options['thinking'], {'type': 'disabled'})
                if provider == 'kimi':
                    self.assertEqual(options['reasoning_effort'], 'max')
                if provider in ('openai', 'kimi'):
                    self.assertEqual(options['provider_label'], f'{provider}:web')
                    self.assertTrue(options['return_metadata'])

    async def test_missing_provider_clients_fail_before_invocation(self):
        with patch.object(self.web, '_kimi_client', None), patch.object(self.web, '_deepseek_anthropic_client', None), \
             patch.object(self.web, 'chat_with_tools', new_callable=AsyncMock) as anth, \
             patch('llm.openai_tool_loop.chat_with_tools', new_callable=AsyncMock) as oai:
            for provider in ('kimi', 'deepseek'):
                with self.assertRaises(RuntimeError):
                    await self.invoke(self.turn(provider))
            anth.assert_not_awaited()
            oai.assert_not_awaited()

    async def test_deepseek_transient_failure_uses_fallback(self):
        import httpx
        with patch.object(self.web, '_deepseek_anthropic_client', object()), \
             patch.object(self.web, 'chat_with_tools', new_callable=AsyncMock, side_effect=httpx.ConnectError('offline')) as anth, \
             patch('llm.openai_tool_loop.chat_with_tools', new_callable=AsyncMock, return_value={'text': 'fallback'}) as oai, \
             patch('llm.provider_failover.resolve_deepseek_failover_model', new_callable=AsyncMock, return_value='test-fallback'):
            turn = self.turn('deepseek')
            result = await self.invoke(turn)
            self.assertEqual(result['text'], 'fallback')
            anth.assert_awaited_once()
            oai.assert_awaited_once()
            self.assertIs(oai.call_args.args[0], turn.history)
            self.assertEqual(oai.call_args.kwargs['model'], 'test-fallback')
            self.assertEqual(oai.call_args.kwargs['provider_label'], 'openai:web-failover')

    async def test_preparation_scopes_history_and_local_feedback(self):
        spec = SimpleNamespace(id='cyber-lenin', provider_override=None, tier_override=None)
        with patch.object(self.web, '_load_web_history', return_value=[{'role': 'user', 'content': 'old'}]) as history, \
             patch.object(self.web, '_load_web_feedback_rows', return_value=[{'id': 11, 'note': 'use examples'}]) as notes, \
             patch.object(self.web, '_load_web_tone_policy', return_value=[{'tone_feedback': 'shorter', 'count': 3}]), \
             patch.object(self.web, 'resolve_runtime_profile', new_callable=AsyncMock, return_value=self.turn('openai').profile), \
             patch.object(self.web, 'render_system_prompt', return_value='system'), \
             patch('kg_runtime.recall.entity_gated_kg_block', return_value=''):
            turn = await self.web._prepare_web_chat(
                'new', spec=spec, session_id='s', fingerprint='fp', fps=['fp'],
                authenticated_user_id=7, regenerate_from_id=None, tone_feedback='', feedback_note='',
            )
            history.assert_called_once_with(['fp'], 's', 20, 'cyber-lenin', set(), account_user_id=7)
            notes.assert_called_once_with(['fp'], 's', 'cyber-lenin', 8, account_user_id=7)
            self.assertEqual(turn.history[0], {'role': 'user', 'content': 'old'})
            self.assertEqual(turn.feedback_ids, [11])
            self.assertIn('note=use examples', turn.history[-1]['content'])
            self.assertIn('selected 3 times recently', turn.history[-1]['content'])
            self.assertTrue(turn.history[-1]['content'].endswith('\n\nnew'))

    async def test_invalid_regeneration_stops_before_model_and_history(self):
        spec = SimpleNamespace(id='cyber-lenin')
        with patch.object(self.web, 'get_web_chat_log_for_feedback', return_value={'user_query_active': False}), \
             patch.object(self.web, '_load_web_history') as history, \
             patch.object(self.web, 'resolve_runtime_profile', new_callable=AsyncMock) as profile:
            result = await self.web._prepare_web_chat(
                'new', spec=spec, session_id='s', fingerprint='fp', fps=['fp'],
                authenticated_user_id=7, regenerate_from_id=42, tone_feedback='', feedback_note='',
            )
            self.assertIsNone(result)
            history.assert_not_called()
            profile.assert_not_awaited()

    async def test_attached_detached_and_failed_persistence(self):
        from scripts.smoke_webchat_disconnect import _check_detached_run_persists
        for regenerate in (False, True):
            for detach in (False, True):
                for save_fails in (False, True):
                    with self.subTest(regenerate=regenerate, detach=detach, save_fails=save_fails):
                        await _check_detached_run_persists(regenerate=regenerate, detach=detach, save_fails=save_fails)
