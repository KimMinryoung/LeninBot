"""Consequential outcomes, draft reuse, failure isolation and Telegram ordering."""
import asyncio
import json
import unittest
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from roleplay import jev, turn
from roleplay.decisions import pending_important, settle_general, DraftOutOfScope
from roleplay.actor import actor_state_view
from test_roleplay_holdouts import initial, held


def verdict(**labels):
    return {'status': 'classified', 'labels': {'mode': 'scene', **{k: 'none' for k in jev.FAMILY_KEYS},
            'elapsed': 'brief', 'activity': 'light', **labels}, 'uncertain': [], 'answers': {},
            'model': 'test', 'draft': '혐의를 인정하고 서명했다.', 'calls': {}}


def authorization():
    return {'user_text': '계속', 'labels': {'mode': 'scene', 'transition': 'current', 'span': 'brief'},
            'auto_general': True}


def proof(key='event_pressure', label='coerced_confession', target_id=None):
    return {'key': key, 'label': label, 'target_id': target_id, 'quote': '서명했다'}


class ConsequenceTests(unittest.TestCase):
    def test_general_choices_batch_without_losing_independent_families(self):
        v = verdict(event_intake='meal')
        for key in ('event_relief', 'intensity', 'activity'):
            v['labels'].pop(key, None)
        v['uncertain'] = ['event_relief', 'intensity', 'activity']
        v['answers'] = {'event_relief': {'probabilities': {'kindness': .7, 'none': .3}},
                        'activity': {'probabilities': {'rest': .8, 'light': .2}}}
        original = deepcopy(v)
        result = settle_general(initial(), [], v)
        self.assertEqual(set(result['labels']['events']), {'meal', 'kindness'})
        self.assertEqual(result['labels']['intensity'], 'moderate')
        self.assertEqual(result['labels']['activity'], 'rest')
        self.assertEqual(v, original)

    def test_low_confidence_loss_is_never_an_automatic_pick(self):
        state = initial(holdouts=held('이름을 쓰지 않기'))
        v = verdict()
        v['uncertain'] = ['holdout_0']
        v['answers']['holdout_0'] = {'confidence': .4, 'probabilities': {'lost': .8, 'keep': .2}}
        result = settle_general(state, [], v)
        self.assertNotIn('holdout_0', result['labels'])
        self.assertEqual(pending_important(state, [], result)[0]['label'], 'lost')
        # Missing record-service output also must not silently become keep.
        result['answers'] = {}
        self.assertEqual(pending_important(state, [], result)[0]['key'], 'holdout_0')

    def test_both_reliable_decision_and_exact_evidence_required(self):
        v = verdict(event_pressure='coerced_confession')
        v['answers']['event_pressure'] = {'confidence': .99}
        for evidence in ([], [{**proof(), 'quote': '없는 인용'}], [{**proof(), 'label': 'implicating_others'}]):
            v['duration_estimate'] = {'important_evidence': evidence}
            self.assertEqual(len(pending_important(initial(), [], v)), 1)
        v['duration_estimate'] = {'important_evidence': [proof()]}
        self.assertEqual(pending_important(initial(), [], v), [])
        v['answers']['event_pressure']['confidence'] = .7
        self.assertEqual(len(pending_important(initial(), [], v)), 1)

    def test_bargain_evidence_must_match_record_and_paid_is_not_kept(self):
        state = initial(bargains=[{'id': 'b1', 'request': '담요', 'price': '진술', 'status': 'open'}])
        v = verdict(bargain_0='kept')
        v['answers']['bargain_0'] = {'confidence': .99}
        for item in (proof('bargain_0', 'paid', 'b1'), proof('bargain_0', 'kept', 'other')):
            v['duration_estimate'] = {'important_evidence': [item]}
            self.assertEqual(len(pending_important(state, [], v)), 1)
        v['duration_estimate'] = {'important_evidence': [proof('bargain_0', 'kept', 'b1')]}
        self.assertEqual(pending_important(state, [], v), [])

    def test_ordinary_interrogation_needs_no_important_confirmation(self):
        self.assertEqual(pending_important(initial(), [], verdict(event_pressure='interrogation')), [])

    def test_pending_reuses_classifier_and_duration_and_changed_draft_invalidates(self):
        v = verdict(event_pressure='coerced_confession')
        before = initial()
        args = ('계속', before, [], [], 'scope', v['draft'], authorization())
        with patch.object(jev, 'classify', side_effect=lambda *a, **k: deepcopy(v)) as classify, \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 2}) as duration, \
             patch.object(turn, 'review_reply', return_value={'approved': None}):
            pending = turn.prepare_result(*args)
            self.assertEqual(pending['key'], 'important')
            cached = pending['verdict']
            cached['confirmed_important'] = [{k: item[k] for k in ('key', 'label', 'target_id')}
                                              for item in cached['pending_important']]
            self.assertEqual(turn.prepare_result(*args, verdict=cached)['status'], 'prepared')
            self.assertEqual(classify.call_count, 1)
            self.assertEqual(duration.call_count, 1)
            changed = (*args[:5], '다른 초안', args[6])
            self.assertEqual(turn.prepare_result(*changed, verdict=cached)['status'], 'pending')
            self.assertEqual(classify.call_count, 2)
            self.assertEqual(duration.call_count, 2)
        self.assertEqual(before, initial())

    def test_changed_revision_invalidates_previous_confirmation(self):
        v = verdict(event_pressure='coerced_confession')
        before = initial()
        with patch.object(jev, 'classify', side_effect=lambda *a, **k: deepcopy(v)) as classify, \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 2}):
            result = turn.prepare_result('계속', before, [], [], 's', v['draft'], authorization())
            before['revision'] += 1
            turn.prepare_result('계속', before, [], [], 's', v['draft'], authorization(), verdict=result['verdict'])
        self.assertEqual(classify.call_count, 2)

    def test_excluded_result_cannot_be_committed_even_with_evidence(self):
        v = verdict(event_pressure='coerced_confession')
        v['answers']['event_pressure'] = {'confidence': .99}
        auth = authorization()
        auth['excluded_outcomes'] = [{k: proof()[k] for k in ('key', 'label', 'target_id')}]
        with patch.object(jev, 'classify', return_value=v), \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 2, 'important_evidence': [proof()]}):
            with self.assertRaises(DraftOutOfScope):
                turn.prepare_result('계속', initial(), [], [], 's', v['draft'], auth)

    def test_duration_call_collects_evidence_without_an_extra_call(self):
        from llm.call_registry import GenerationResult
        v = verdict(event_pressure='coerced_confession')
        value = {'elapsed_minutes': 2, 'reason': '서명', 'within_scope': True,
                 'timeline_anchor': None, 'important_evidence': [proof()]}
        with patch.object(jev, 'generate_detailed', return_value=GenerationResult(text=json.dumps(value))) as generate:
            duration = jev.estimate_duration('계속', initial(), v)
        generate.assert_called_once()
        self.assertEqual(duration['important_evidence'], [proof()])
        payload = json.loads(generate.call_args.args[1])
        self.assertEqual(payload['important_candidates'][0]['label'], 'coerced_confession')

    def test_actor_context_carries_different_consequences_and_bounded_focus(self):
        for status, expected in [('kept', '이행됨'), ('broken', '파기됨')]:
            state = initial(goal='담요를 받기', next_action='약속을 묻기', holdouts=held('서명 거부'),
                bargains=[{'id': 'b1', 'status': status, 'request': '담요', 'price': '진술'}])
            original = deepcopy(state)
            view = actor_state_view(state, '약속은?')
            self.assertIn(expected, view['bargains']['최근 결과'][0])
            self.assertEqual(view['scene_focus']['가능한 시도'], '약속을 묻기')
            self.assertLessEqual(len(view['scene_focus']['관련 단서']), 3)
            self.assertEqual(state, original)


class TelegramDecisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from roleplay import bot
        self.bot = bot
        bot.PENDING_CHOICES.clear()
        bot.TURN_LOCKS.clear()
        self.addCleanup(bot.PENDING_CHOICES.clear)
        self.message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='계속', message_id=11,
            chat=SimpleNamespace(id=1), answer=AsyncMock(), bot=SimpleNamespace(send_chat_action=AsyncMock()))

    @contextmanager
    def harness(self, v):
        bot = self.bot
        before = initial()
        @contextmanager
        def stage(uid):
            yield {'people': [], 'state': before, 'records': {}, 'baseline': {}}
        async def thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        with patch.object(bot.asyncio, 'to_thread', side_effect=thread), \
             patch.object(bot, '_asks_player', return_value=False), \
             patch.object(bot, 'build_system_prompt', return_value='test'), \
             patch.object(turn, 'staged_memory', side_effect=stage), \
             patch.object(turn, 'review_reply', return_value={'approved': None}), \
             patch.object(jev, 'classify', side_effect=lambda *a, **k: deepcopy(v)) as classify, \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 2}) as duration, \
             patch.object(bot, '_make_progress_callback', return_value=SimpleNamespace(flush=AsyncMock())), \
             patch.object(bot, '_drop_unsettled') as drop, \
             patch.object(bot, '_deliver', new_callable=AsyncMock) as deliver, \
             patch.object(bot, 'chat_with_tools', new_callable=AsyncMock, return_value='혐의를 인정하고 서명했다.') as chat, \
             patch.object(bot, 'adjudicate_turn', return_value={'status': 'applied', 'reply': '혐의를 인정하고 서명했다.'}) as commit:
            yield {'user_text': '계속', 'scope_id': '11', 'chat_id': 1, 'history': [], 'state': before,
                   'notes': [], 'people': []}, classify, duration, chat, commit, deliver, drop

    async def test_important_confirmation_reuses_same_draft_and_stale_tap_cannot_commit(self):
        with self.harness(verdict(event_pressure='coerced_confession')) as (ctx, classify, duration, chat, commit, deliver, drop):
            await self.bot._draft_and_settle(self.message, 1, ctx, authorization())
            commit.assert_not_called()
            self.assertEqual(self.bot.PENDING_CHOICES[1]['key'], 'important')
            query = SimpleNamespace(from_user=SimpleNamespace(id=1), data='rp:11:important:confirm',
                answer=AsyncMock(), message=SimpleNamespace(edit_text=AsyncMock(), answer=AsyncMock()))
            await self.bot.on_choice(query)
            await self.bot.on_choice(query)
            commit.assert_called_once()
            deliver.assert_awaited_once()
            chat.assert_awaited_once()
            classify.assert_called_once()
            duration.assert_called_once()
            drop.assert_not_called()

    async def test_model_failure_does_not_regenerate_actor(self):
        with self.harness({'status': 'unavailable'}) as (ctx, classify, duration, chat, commit, deliver, drop):
            await self.bot._draft_and_settle(self.message, 1, ctx, authorization())
            chat.assert_awaited_once()
            commit.assert_not_called()
            duration.assert_not_called()
            drop.assert_called_once()
            self.assertIn('판정 서비스', self.message.answer.call_args.args[0])

    async def test_rewrite_discards_staged_people_and_keeps_original_history(self):
        with self.harness(verdict(event_pressure='coerced_confession')) as (ctx, *mocks):
            await self.bot._draft_and_settle(self.message, 1, ctx, authorization())
            self.bot.PENDING_CHOICES[1]['people'] = [{'person_id': 'uncommitted'}]
            query = SimpleNamespace(from_user=SimpleNamespace(id=1), data='rp:11:important:rewrite', answer=AsyncMock(),
                message=SimpleNamespace(edit_text=AsyncMock(), answer=AsyncMock()))
            with patch.object(self.bot, 'people_context', return_value={'index': [], 'present': []}), \
                 patch.object(self.bot, '_draft_and_settle', new_callable=AsyncMock) as generate:
                await self.bot.on_choice(query)
            rewritten = generate.call_args.args[2]
            self.assertEqual(rewritten['history'], [])
            self.assertEqual(rewritten['people'], {'index': [], 'present': []})
            self.assertEqual(rewritten['rewrite_without'][0]['label'], 'coerced_confession')

    async def test_same_user_serialized_other_user_independent(self):
        bot = self.bot
        entered = asyncio.Event()
        release = asyncio.Event()
        order = []
        async def handler(event, data):
            order.append(event.name)
            if event.name == 'first':
                entered.set()
                await release.wait()
        def event(uid, name):
            return SimpleNamespace(from_user=SimpleNamespace(id=uid), name=name)
        middleware = bot.OwnerOnlyMiddleware()
        with patch.object(bot, '_is_allowed', return_value=True):
            first = asyncio.create_task(middleware(handler, event(1, 'first'), {}))
            await entered.wait()
            second = asyncio.create_task(middleware(handler, event(1, 'second'), {}))
            await middleware(handler, event(2, 'other'), {})
            await asyncio.sleep(0)
            self.assertEqual(order, ['first', 'other'])
            release.set()
            await asyncio.gather(first, second)
        self.assertEqual(order, ['first', 'other', 'second'])
