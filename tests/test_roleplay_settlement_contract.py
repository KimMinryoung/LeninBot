"""Regressions for omitted record decisions and clock/draft divergence."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from llm.call_registry import Decision, DecisionResult, GenerationResult
from roleplay import jev, turn
from roleplay.pacing import policy_for, turn_time_scope
from test_roleplay_holdouts import initial, held


class RecordContractTests(unittest.TestCase):
    def test_low_confidence_loss_is_not_silently_keep(self):
        before = initial(holdouts=held('증언 거부'), humiliation=60)
        verdict = {'labels': {'mode': 'scene', 'event': 'none', 'elapsed': '0'},
                   'uncertain': ['holdout_0'], 'answers': {'holdout_0': {
                       'choice': 'lost', 'confidence': .36, 'probabilities': {'lost': .68, 'keep': .32}}}}
        original = deepcopy(before)
        with turn_time_scope(policy_for('증언을 썼다')):
            with self.assertRaises(jev.PendingChoice) as caught:
                jev.project(before, '증언을 썼다', [], verdict, 'test')
            self.assertEqual(caught.exception.key, 'holdout_0')
            self.assertEqual(caught.exception.candidates[0][0], 'lost')
            verdict['labels']['holdout_0'] = 'lost'
            after, applied = jev.project(before, '증언을 썼다', [], verdict, 'test')
        self.assertEqual(before, original)
        self.assertEqual(applied['holdouts_lost'], ['증언 거부'])
        self.assertEqual(after['holdouts'][0]['status'], 'lost')

    def test_focused_record_retry_uses_latest_answer_and_accounts_cost(self):
        def answer(feature, payload, questions, **kwargs):
            answers = {k: {'choice': next(iter(q['criteria'])), 'confidence': .99} for k, q in questions.items()}
            if 'holdout_0' in questions:
                answers['holdout_0'] = {'choice': 'lost', 'confidence': .36, 'probabilities': {'lost': .68, 'keep': .32}}
                if kwargs['label'] == 'roleplay-record-review':
                    self.assertEqual(set(questions), {'holdout_0'})
                    self.assertEqual(payload['current']['holdouts'][0]['title'], '증언 거부')
                    answers['holdout_0'] = {'choice': 'lost', 'confidence': .96}
            return DecisionResult(decision=Decision(model='test', answers=answers, cost_usd=.001))
        profile = SimpleNamespace(extra={'enabled': True, 'thresholds': {'accept': .75}})
        with patch.object(jev, 'resolve', return_value=profile), patch.object(jev, 'decide_detailed', side_effect=answer):
            result = jev.classify('써라', initial(holdouts=held('증언 거부')), [], [], draft='증언을 썼다')
        self.assertEqual(result['labels']['holdout_0'], 'lost')
        self.assertNotIn('holdout_0', result['uncertain'])
        self.assertIn('roleplay-record-review', result['calls'])
        self.assertGreaterEqual(result['cost_usd'], .003)

    def test_clock_and_authorization_reach_duration_checker(self):
        before = initial()
        before['clock'].update(date='1939-04-28', time='17:50')
        verdict = {'labels': {}, 'draft': '21시 심문을 시작했다', 'duration_limit': 180,
                   'authorization': {'labels': {'time_scope': 'none'}}}
        result = GenerationResult(text=json.dumps({'elapsed_minutes': 85, 'reason': 'incorrect model arithmetic', 'within_scope': True, 'timeline_anchor': {'date': '1939-04-28', 'time': '21:00', 'quote': '21시'}}))
        with patch.object(jev, 'generate_detailed', return_value=result) as generate:
            with self.assertRaisesRegex(ValueError, '허가된 사건·시간'):
                jev.estimate_duration('밥 먹고 쉬어. 21시에 심문이 있다.', before, verdict)
        payload = json.loads(generate.call_args.args[1])
        self.assertEqual(payload['clock_before']['time'], '17:50')
        self.assertEqual(payload['authorization'], verdict['authorization'])

    def test_explicit_passage_cannot_skip_draft_validation(self):
        before = initial()
        text = '20분 쉰다'
        auth = {'user_text': text, 'labels': {'mode': 'scene', 'transition': 'current', 'time_scope': 'explicit', 'span': 'brief'}, 'duration_minutes': 20}
        verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'event': 'none'}}
        with patch.object(jev, 'classify', return_value=verdict), patch.object(jev, 'estimate_duration', side_effect=ValueError('초안 초과')):
            with self.assertRaisesRegex(ValueError, '초안 초과'):
                turn.prepare(text, before, [], [], 'test', '다음날까지 잤다', auth)


class SnapshotGatewayTests(unittest.TestCase):
    def test_reads_fresh_and_discarded_writes_have_no_durable_receipts(self):
        from tool_gateway.dispatcher import _roleplay_snapshot_call
        from roleplay.memory import MEMORY_OVERRIDE
        self.assertTrue(_roleplay_snapshot_call('roleplay_state', {'action': 'read'}))
        self.assertFalse(_roleplay_snapshot_call('roleplay_state', {'action': 'update'}))
        token = MEMORY_OVERRIDE.set('/tmp/disposable-test.sqlite3')
        try:
            self.assertTrue(_roleplay_snapshot_call('roleplay_state', {'action': 'update'}))
            self.assertTrue(_roleplay_snapshot_call('roleplay_person', {'action': 'save'}))
            self.assertFalse(_roleplay_snapshot_call('send_email', {}))
        finally:
            MEMORY_OVERRIDE.reset(token)


class SnapshotDispatchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_refresh_and_same_write_runs_in_each_disposable_draft(self):
        import tempfile
        from pathlib import Path
        from roleplay import memory
        from tool_gateway.dispatcher import execute_tool
        from tool_gateway.security import caller_scope, new_run_context
        decision = SimpleNamespace(denied=False, risk_class='write')
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory, 'MEMORY_PATH', Path(tmp) / 'state.sqlite3'), \
             patch('tool_gateway.security.authorize', return_value=decision), patch('tool_gateway.security.audit'), \
             patch('security_gateway.idempotency.lookup', side_effect=AssertionError('snapshot must not consult old receipts')), \
             caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True,
                                          session_id='same-chat', scope_type='telegram_message', scope_id='one')):
            with memory._connection() as conn:
                conn.execute('INSERT INTO character_state VALUES (?,?)', ('1', json.dumps(initial(scene='original'))))
            cache = {}
            for _ in range(2):
                with turn.staged_memory('1'):
                    old, err = await execute_tool('roleplay_state', {'action': 'read'}, {'roleplay_state': memory.roleplay_state}, idempotency_cache=cache)
                    self.assertFalse(err)
                    self.assertEqual(json.loads(old)['scene'], 'original')
                    _, err = await execute_tool('roleplay_state', {'action': 'update', 'changes': {'scene': 'draft'}, 'reason': 'test'}, {'roleplay_state': memory.roleplay_state}, idempotency_cache=cache)
                    self.assertFalse(err)
                    new, err = await execute_tool('roleplay_state', {'action': 'read'}, {'roleplay_state': memory.roleplay_state}, idempotency_cache=cache)
                    self.assertFalse(err)
                    self.assertEqual(json.loads(new)['scene'], 'draft')
                self.assertEqual(memory.load_state('1')['scene'], 'original')


class StatusRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_replies_with_incomplete_time_audit(self):
        from unittest.mock import AsyncMock
        from roleplay import bot
        async def inline_thread(func, *args):
            return func(*args)
        for interpretation, expected in (({'operation': 'repair', 'interpretation': '확정 장면 기준 시계 복구'}, '확정 장면 기준 시계 복구'),
                                          ({'source_quote': '20분 쉬어'}, '20분 쉬어'), ({}, '아직 없음')):
            with self.subTest(interpretation=interpretation):
                state = initial()
                state['clock'].update(time='21:27', certainty='estimated', last_interpretation=interpretation)
                message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='/status', answer=AsyncMock())
                with patch.object(bot, 'load_state', return_value=state), patch.object(bot, 'people_context', return_value={'index': []}), \
                     patch.object(bot.asyncio, 'to_thread', side_effect=inline_thread):
                    await bot.cmd_status(message)
                rendered = '\n'.join(call.args[0] for call in message.answer.await_args_list)
                self.assertIn('21:27', rendered)
                self.assertIn(expected, rendered)
                self.assertIn('예조프 상태', rendered)
