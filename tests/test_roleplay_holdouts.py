"""Holdouts, delayed reactions, player-settled choices and the settlement line."""
import json
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from runtime_tools import roleplay_memory as memory, roleplay_jev as jev, roleplay_turn as turn
from runtime_tools.roleplay_actor import actor_state_view, actor_outcome_view
from runtime_tools.roleplay_dynamics import with_defaults, advance
from runtime_tools.roleplay_pacing import policy_for, turn_time_scope
from runtime_tools.roleplay_story import apply_story_updates
from tool_gateway.security import caller_scope, new_run_context


def initial(**values):
    return with_defaults({**memory.STATE_DEFAULTS, 'hunger': 40, 'fatigue': 40, 'pain': 20, 'tension': 50,
                          'resolve': 2, 'clarity': 55, 'humiliation': 100, 'conditions_initialized': True,
                          'activity': 'light', 'threat': 'threatening', 'participants': ['rodos'], **values})


def held(*titles, lost=()):
    items = [{'id': f'h{i + 1}', 'title': t, 'status': 'held', 'created_minute': 0} for i, t in enumerate(titles)]
    items += [{'id': f'h{len(titles) + i + 1}', 'title': t, 'status': 'lost', 'created_minute': 0, 'lost_minute': 0} for i, t in enumerate(lost)]
    return items


def project(text, state, **labels):
    verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'elapsed': '0', 'event': 'none', **labels},
               'uncertain': [], 'model': 'jev-test', 'answers': {}}
    with turn_time_scope(policy_for(text)):
        return jev.project(state, text, [{'person_id': 'rodos', 'name': '로도스'}], verdict, 'scope')


class HoldoutTests(unittest.TestCase):
    def test_questions_cover_held_items_and_alone_cues_only(self):
        state = initial(holdouts=held('빈 두 줄', lost=['이름 낭독 거부']),
                        story_events=[{'id': 'delayed-reaction-1', 'title': '반응', 'status': 'ready', 'when_alone': True, 'ready_minute': 0},
                                      {'id': 'visit', 'title': '방문', 'status': 'pending', 'due_minute': 500}])
        questions = jev.build_questions(state, [])
        self.assertEqual(set(questions['holdout_0']['criteria']), {'keep', 'lost'})
        self.assertNotIn('holdout_1', questions)
        self.assertIn('미뤄 둔 반응', questions['story_0']['instructions'])
        self.assertIn('예정 사건', questions['story_1']['instructions'])
        options = {k for key in jev.FAMILY_KEYS for k in questions[key]['criteria']}
        self.assertNotIn('holdout_lost', options)
        self.assertNotIn('sexual_coercion', options)
        self.assertNotIn('event', questions)

    def test_losing_a_holdout_is_discrete_bounded_and_once(self):
        before = initial(holdouts=held('빈 두 줄', '한 줄 유지'), humiliation=60)
        state, applied = project('로도스가 빈 줄을 채우게 했다', before, event='coerced_confession', intensity='moderate', holdout_0='lost')
        self.assertEqual(applied['holdouts_lost'], ['빈 두 줄'])
        by_title = {h['title']: h for h in state['holdouts']}
        self.assertEqual(by_title['빈 두 줄']['status'], 'lost')
        self.assertEqual(by_title['빈 두 줄']['lost_event'], 'coerced_confession')
        self.assertEqual(by_title['한 줄 유지']['status'], 'held')
        kinds = [e['kind'] for e in state['resolve_events']]
        self.assertEqual(kinds, ['coerced_confession', 'holdout_lost'])
        self.assertLess(state['resolve'], before['resolve'])
        self.assertGreater(state['humiliation'], 65)  # confession +5 plus holdout +5, both dampened above 60
        again, applied_again = project('같은 요구를 되풀이', state, event='none', holdout_0='lost')
        self.assertNotIn('holdouts_lost', applied_again)
        self.assertEqual(len(again['resolve_events']), 2)
        untouched, applied_none = project('요구만 받음', before, event='interrogation', holdout_0='keep')
        self.assertNotIn('holdouts_lost', applied_none)
        self.assertEqual([h['status'] for h in untouched['holdouts']], ['held', 'held'])

    def test_actor_registers_holdouts_but_never_loses_or_revives_them(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory, 'MEMORY_PATH', Path(tmp) / 'm.sqlite3'):
            with memory._connection() as conn:
                conn.execute('INSERT INTO character_state VALUES (?,?)', ('1', json.dumps(initial(revision=3, holdouts=held('빈 두 줄', lost=['이름 낭독 거부'])))))
            with caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True, scope_type='telegram_message', scope_id='s1')):
                result = json.loads(memory.roleplay_state('update', changes={'holdouts': ['빈 두 줄', "'인민의 지지' 한 줄"]}, reason='아직 지키는 것', expected_revision=3))
                self.assertEqual(result['holdouts']['아직 지키는 것'], ['빈 두 줄', "'인민의 지지' 한 줄"])
                self.assertEqual(result['holdouts']['이미 넘긴 것'], ['이름 낭독 거부'])
                with self.assertRaises(ValueError):
                    memory.roleplay_state('update', changes={'holdouts': ['이름 낭독 거부']}, reason='되살리기', expected_revision=4)
                with self.assertRaises(ValueError):
                    memory.roleplay_state('update', changes={'holdouts': ['a', 'b', 'c', 'd']}, reason='너무 많음', expected_revision=4)
                dropped = json.loads(memory.roleplay_state('update', changes={'holdouts': ["'인민의 지지' 한 줄"]}, reason='하나만 보냄', expected_revision=4))
                self.assertEqual(dropped['holdouts']['아직 지키는 것'], ['빈 두 줄', "'인민의 지지' 한 줄"])
                self.assertTrue(any('그대로 유지' in w for w in dropped['warnings']))
            stored = memory.load_state('1')
            self.assertEqual([h['status'] for h in stored['holdouts']], ['held', 'lost', 'held'])
            self.assertEqual(stored['revision'], 5)

    def test_actor_view_is_narrative(self):
        view = actor_state_view(initial(holdouts=held('빈 두 줄', lost=['이름 낭독 거부'])))
        self.assertEqual(view['holdouts']['아직 지키는 것'], ['빈 두 줄'])
        self.assertEqual(view['holdouts']['이미 넘긴 것'], ['이름 낭독 거부'])
        self.assertNotIn('lost_minute', json.dumps(view, ensure_ascii=False))
        outcome = actor_outcome_view({'status': 'applied', 'applied': {'event': 'coerced_confession', 'holdouts_lost': ['빈 두 줄'], 'delayed_reaction': 'scheduled'}})
        self.assertEqual(outcome['holdouts_lost'], ['빈 두 줄'])
        self.assertIn('혼자 남는 장면', outcome['delayed_reaction'])


class DelayedReactionTests(unittest.TestCase):
    def test_saturated_humiliation_defers_the_reaction_until_alone(self):
        before = initial(humiliation=95)
        state, applied = project('증인 앞에서 복종시켰다', before, event='public_submission', intensity='moderate')
        self.assertEqual(applied['delayed_reaction'], 'scheduled')
        event = state['story_events'][0]
        self.assertTrue(event['when_alone'])
        self.assertEqual(event['status'], 'pending')  # Rodos is still in the room
        # A second humiliation while one is pending does not stack another cue.
        stacked, applied_more = project('또 굴욕', state, event='public_submission', intensity='moderate')
        self.assertNotIn('delayed_reaction', applied_more)
        self.assertEqual(len(stacked['story_events']), 1)
        # Leaving the character alone readies the cue without stopping the clock.
        alone, _ = project('로도스가 나갔다', stacked, event='none', person_0='leave')
        self.assertEqual(alone['story_events'][0]['status'], 'ready')
        moved = advance({**alone, 'activity': 'rest'}, alone['scene_minute'] + 120, '두 시간 혼자')
        self.assertIsNone(moved['story_interrupt'])
        self.assertEqual(moved['scene_minute'], alone['scene_minute'] + 120)
        view = actor_state_view(alone)
        self.assertTrue(any(e['status'] == 'cue' for e in view['story_events']))
        # The reaction shown in a solitary scene releases some of what the slider could not hold.
        released, applied_release = project('혼자 남아 담요를 끌어안고 울었다', moved, event='none', story_0='complete')
        self.assertEqual(applied_release['delayed_reaction'], 'released')
        self.assertEqual(released['story_events'][0]['status'], 'completed')
        self.assertEqual(released['humiliation'], moved['humiliation'] - 4)
        self.assertEqual(released['tension'], round(moved['tension'] - 3, 4))

    def test_not_scheduled_below_saturation_and_expires_unused(self):
        state, applied = project('복종', initial(humiliation=70), event='public_submission', intensity='moderate')
        self.assertNotIn('delayed_reaction', applied)
        self.assertEqual(state['story_events'], [])
        scheduled = apply_story_updates(initial(participants=[]), [{'op': 'schedule', 'id': 'delayed-reaction-x', 'title': '반응', 'source': '포화', 'when_alone': True}])
        self.assertEqual(scheduled['story_events'][0]['status'], 'ready')
        from runtime_tools.roleplay_story import advance_to_event
        late = advance_to_event({**scheduled, 'activity': 'rest'}, 1440, '하루', advance)
        self.assertEqual(late['story_events'][0]['status'], 'cancelled')
        self.assertIsNone(late['story_interrupt'])
        with self.assertRaises(ValueError):
            apply_story_updates(initial(), [{'op': 'schedule', 'id': 'bad', 'title': 't', 'source': 's', 'when_alone': False}])


class PlayerChoiceTests(unittest.TestCase):
    def test_uncertain_event_or_intensity_becomes_a_pending_choice(self):
        verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'elapsed': '0', 'activity': 'light'}, 'uncertain': ['event'],
                   'answers': {'event': {'probabilities': {'interrogation': .4, 'coerced_confession': .35, 'none': .1, 'beating': .05, 'sexual_unspecified': .1}}}}
        with turn_time_scope(policy_for('심문')), self.assertRaises(jev.PendingChoice) as caught:
            jev.project(initial(), '심문', [], verdict, 'scope')
        self.assertEqual(caught.exception.key, 'event')
        self.assertEqual([k for k, _ in caught.exception.candidates], ['interrogation', 'coerced_confession', 'beating', 'none'])
        self.assertIsInstance(caught.exception, ValueError)
        with turn_time_scope(policy_for('구타')), self.assertRaises(jev.PendingChoice) as caught:
            jev.project(initial(), '구타', [], {**verdict, 'labels': {**verdict['labels'], 'event': 'beating'}}, 'scope')
        self.assertEqual(caught.exception.key, 'intensity')

    def test_unknown_activity_defaults_for_brief_intervals_and_asks_for_long_ones(self):
        verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'elapsed': 'brief', 'event': 'kindness', 'intensity': 'moderate'},
                   'uncertain': ['activity'], 'answers': {}, 'duration_estimate': {'elapsed_minutes': 4}}
        with turn_time_scope(policy_for('물을 건넸다')):
            state, applied = jev.project(initial(), '물을 건넸다', [], verdict, 'scope')
        self.assertEqual((state['activity'], state['scene_minute'], applied['activity_defaulted']), ('light', 4, 'light'))
        long = {**verdict, 'labels': {**verdict['labels'], 'elapsed': 'explicit'}}
        with turn_time_scope(policy_for('한 시간 쉬어')), self.assertRaises(jev.PendingChoice) as caught:
            jev.project(initial(), '한 시간 쉬어', [], long, 'scope')
        self.assertEqual(caught.exception.key, 'activity')
        self.assertEqual(caught.exception.candidates[0][0], 'light')
        # After one player answer, remaining gaps settle by default instead of a second round.
        settled = {**long, 'player_settled': True, 'labels': {k: v for k, v in long['labels'].items() if k != 'intensity'}}
        with turn_time_scope(policy_for('한 시간 쉬어')):
            state, applied = jev.project(initial(participants=[]), '한 시간 쉬어', [], settled, 'scope')
        self.assertEqual((state['activity'], applied['activity_defaulted'], applied['intensity'], state['scene_minute']), ('rest', 'rest', 'moderate', 60))

    def test_prepare_with_settled_verdict_skips_gate_and_classifier(self):
        before = initial()
        auth = {'user_text': '심문', 'labels': {'mode': 'scene', 'transition': 'current', 'span': 'brief'}}
        verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'activity': 'light', 'location': 'keep'}, 'uncertain': ['event'],
                   'answers': {}, 'model': 't', 'draft': '초안', 'authorization': auth, 'duration_estimate': {'elapsed_minutes': 3}}
        with patch.object(turn, 'decide') as gate, patch.object(jev, 'classify') as classify, patch.object(jev, 'estimate_duration') as estimate, \
             patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}):
            with self.assertRaises(jev.PendingChoice) as caught:
                turn.prepare('심문', before, [], [], '9', '초안', auth, None, verdict)
            self.assertEqual(caught.exception.verdict['labels']['elapsed'], 'brief')
            settled = dict(caught.exception.verdict)
            settled['labels'] = {**settled['labels'], 'event': 'interrogation'}
            prepared = turn.prepare('심문', before, [], [], '9', '초안', auth, None, settled)
        gate.assert_not_called(); classify.assert_not_called(); estimate.assert_not_called()
        self.assertEqual(prepared['applied']['event'], 'interrogation')
        self.assertEqual(prepared['state']['scene_minute'], 3)

    def test_feedback_line_names_what_the_engine_believed(self):
        line = turn.feedback_line({'status': 'applied', 'applied': {'event': 'implicating_others', 'minutes': 90, 'holdouts_lost': ['빈 두 줄'], 'delayed_reaction': 'scheduled'}},
                                  {'clock': {'time': '07:30', 'certainty': 'estimated'}})
        self.assertEqual(line, '⚙ 확정: 타인 연루 진술 · 90분 · 07:30 추정 · 넘긴 것: 빈 두 줄 · 미뤄 둔 반응 예약')
        self.assertIn('상담', turn.feedback_line({'status': 'unchanged', 'applied': {'no_change': True}}))
        self.assertIn('시간·장면 조건 보류', turn.feedback_line({'status': 'applied', 'applied': {'event': 'recognition', 'deferred_components': ['time']}}))
        self.assertNotIn('90', turn.feedback_line({'status': 'applied', 'applied': {'event': 'kindness', 'minutes': 0, 'narrative_only': True}}))


class BotChoiceFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_choice_offers_buttons_then_settles_the_same_draft(self):
        from telegram import roleplay_bot as bot
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='심문해', message_id=5, chat=SimpleNamespace(id=1),
                                  answer=AsyncMock(), bot=SimpleNamespace(send_chat_action=AsyncMock()))
        progress = SimpleNamespace(flush=AsyncMock())
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        @contextmanager
        def fake_stage(uid):
            yield {'people': []}
        auth = {'user_text': '심문해', 'labels': {'mode': 'scene', 'transition': 'current', 'span': 'brief'}}
        pending = jev.PendingChoice('event', [('interrogation', '집중 심문'), ('none', '뚜렷한 사건 없음')], '불확실')
        pending.verdict = {'labels': {'mode': 'scene'}, 'answers': {}}
        prepared = {'reply': '초안 본문'}
        with patch.object(bot.asyncio, 'to_thread', side_effect=inline_thread), \
             patch.object(bot.roleplay_turn, 'committed_reply', return_value=None), \
             patch.object(bot.roleplay_turn, 'authorize', return_value=auth), \
             patch.object(bot.roleplay_turn, 'staged_memory', side_effect=fake_stage), \
             patch.object(bot.roleplay_turn, 'prepare', side_effect=[pending, prepared]) as prepare, \
             patch.object(bot, 'adjudicate_turn', return_value={'status': 'applied', 'reply': '초안 본문', 'applied': {'event': 'interrogation', 'minutes': 3}}) as settle, \
             patch.object(bot, 'save_message') as save, patch.object(bot, 'load_history', return_value=[]), \
             patch.object(bot, 'load_notes', return_value=[]), patch.object(bot, 'load_state', return_value={'hunger': 25, 'clock': {'time': '21:00'}}), \
             patch.object(bot, 'get_preference', return_value='on'), \
             patch.object(bot, 'people_context', return_value={'index': [], 'present': []}), \
             patch.object(bot, 'build_system_prompt', return_value='sys'), \
             patch.object(bot, '_make_progress_callback', return_value=progress), \
             patch.object(bot, 'chat_with_tools', new_callable=AsyncMock, return_value='초안 본문'):
            await bot.handle_message(message)
            self.assertEqual(save.call_count, 1)  # only the user's message so far
            self.assertTrue(message.answer.await_args_list[0].args[0].startswith('【미확정 초안'))
            self.assertIn('초안 본문', message.answer.await_args_list[0].args[0])
            markup = message.answer.call_args.kwargs['reply_markup']
            self.assertEqual([b.callback_data for row in markup.inline_keyboard for b in row],
                             ['rp:5:event:interrogation', 'rp:5:event:none', 'rp:5:cancel:-'])
            self.assertEqual(bot.PENDING_CHOICES[1]['scope_id'], '5')
            query = SimpleNamespace(from_user=SimpleNamespace(id=1), data='rp:5:event:interrogation', answer=AsyncMock(),
                                    message=SimpleNamespace(edit_text=AsyncMock(), answer=AsyncMock()))
            await bot.on_choice(query)
            self.assertEqual(prepare.call_args.args[-2]['labels']['event'], 'interrogation')
            settle.assert_called_once()
            self.assertEqual(settle.call_args.kwargs['prepared'], prepared)
            self.assertNotIn(1, bot.PENDING_CHOICES)
            save.assert_called_with(1, 'assistant', '초안 본문')
            sent = [c.args[0] for c in query.message.answer.await_args_list]
            self.assertEqual(sent[0], '초안 본문')
            self.assertEqual(sent[1], '⚙ 확정: 집중 심문 · 3분 · 21:00')
            stale = SimpleNamespace(from_user=SimpleNamespace(id=1), data='rp:5:event:none', answer=AsyncMock(), message=SimpleNamespace(edit_text=AsyncMock()))
            await bot.on_choice(stale)
            self.assertTrue(stale.answer.await_args.kwargs.get('show_alert'))

    async def test_status_lists_holdouts(self):
        from telegram import roleplay_bot as bot
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='/status', answer=AsyncMock())
        state = {**memory.STATE_DEFAULTS, 'hunger': 50, 'holdouts': held('빈 두 줄', lost=['이름 낭독 거부'])}
        with patch.object(bot.asyncio, 'to_thread', new=AsyncMock(return_value=state)):
            await bot.cmd_status(message)
        self.assertIn('아직 지키는 것: 빈 두 줄 / 넘긴 것: 이름 낭독 거부', message.answer.call_args.args[0])


if __name__ == '__main__':
    unittest.main()


class AuthorizeToleranceTests(unittest.TestCase):
    """Short director commands must not lose the turn over secondary uncertainty."""

    def decision(self, answers):
        from llm.call_registry import Decision, DecisionResult
        return DecisionResult(decision=Decision(answers=answers, model='jev-test'))

    def test_unsure_span_and_transition_default_to_the_smaller_scene(self):
        sure = {'mode': {'choice': 'scene', 'confidence': .95, 'probabilities': {'scene': .95}},
                'transition': {'choice': 'current', 'confidence': .5}, 'span': {'choice': 'session', 'confidence': .55}}
        profile = SimpleNamespace(extra={'enabled': True, 'thresholds': {'accept': .75}})
        with patch.object(turn, 'resolve', return_value=profile), patch.object(turn, 'decide_detailed', side_effect=[self.decision(sure), self.decision(sure)]):
            result = turn.decide('compatibility', {}, {k: jev.choice(k, options) for k, options in {
                'mode': {'scene': 'scene'}, 'transition': {'current': 'current'},
                'span': {'brief': 'brief', 'session': 'session'}, 'time_scope': {'none': 'none'}}.items()},
                defaults={'transition': 'current', 'span': 'brief', 'time_scope': 'none'})
            result['user_text'] = '아침 배식이나 해라'
        self.assertEqual(result['labels'], {'mode': 'scene', 'transition': 'current', 'span': 'brief', 'time_scope': 'none'})
        self.assertEqual(result['defaulted'], {'transition': 'current', 'span': 'brief', 'time_scope': 'none'})
        self.assertEqual(turn.policy_for_authorization(result).max_minutes, 10)

    def test_unsure_mode_and_scope_become_player_choices(self):
        unsure = {'mode': {'choice': 'scene', 'confidence': .6, 'probabilities': {'scene': .6, 'discussion': .35, 'plan': .05}},
                  'transition': {'choice': 'current', 'confidence': .9}, 'span': {'choice': 'brief', 'confidence': .9}}
        profile = SimpleNamespace(extra={'enabled': True, 'thresholds': {'accept': .75}})
        with patch.object(turn, 'resolve', return_value=profile), patch.object(turn, 'decide_detailed', side_effect=[self.decision(unsure), self.decision(unsure)]):
            with self.assertRaises(jev.PendingChoice) as caught:
                turn.decide('compatibility', {}, {'mode': jev.choice('mode', {k: k for k in ('scene', 'discussion', 'plan', 'correction', 'reset')})})
        self.assertEqual(caught.exception.key, 'mode')
        self.assertEqual([k for k, _ in caught.exception.candidates], ['scene', 'discussion', 'plan', 'correction'])
        auth = {'user_text': '의사 재방문', 'labels': {'mode': 'scene', 'transition': 'current', 'span': 'brief'}}
        verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'event': 'treatment', 'activity': 'light', 'location': 'keep'}, 'uncertain': [], 'answers': {}, 'model': 't'}
        # The player's confirmation skips the gate; the classifier then runs as usual.
        with patch.object(turn, 'decide') as gate_call, patch.object(jev, 'classify', return_value=deepcopy(verdict)), \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 4}), patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}):
            prepared = turn.prepare('의사 재방문', initial(), [], [], '9', '초안', auth, None, None, True)
        gate_call.assert_not_called()
        self.assertEqual(prepared['applied']['event'], 'treatment')
        self.assertTrue(prepared['verdict']['scope_review']['player_confirmed'])


class BotAuthorizeChoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_mode_choice_runs_the_turn_with_the_players_answer(self):
        from telegram import roleplay_bot as bot
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='의사 재방문', message_id=8, chat=SimpleNamespace(id=1),
                                  answer=AsyncMock(), bot=SimpleNamespace(send_chat_action=AsyncMock()))
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        pending = jev.PendingChoice('mode', [('scene', 'Perform'), ('discussion', 'Discuss')], '불확정')
        with patch.object(bot.asyncio, 'to_thread', side_effect=inline_thread), \
             patch.object(bot.roleplay_turn, 'committed_reply', return_value=None), \
             patch.object(bot.roleplay_turn, 'authorize', side_effect=pending), \
             patch.object(bot, '_draft_and_settle', new_callable=AsyncMock) as run, \
             patch.object(bot, 'save_message'), patch.object(bot, 'load_history', return_value=[]), \
             patch.object(bot, 'get_preference', return_value='on'), \
             patch.object(bot, 'load_notes', return_value=[]), patch.object(bot, 'load_state', return_value={'hunger': 25}), \
             patch.object(bot, 'people_context', return_value={'index': [], 'present': []}):
            await bot.handle_message(message)
            run.assert_not_awaited()
            markup = message.answer.call_args.kwargs['reply_markup']
            self.assertEqual([b.text for row in markup.inline_keyboard for b in row], ['장면 실행', '질문·상담', '버리기'])
            query = SimpleNamespace(from_user=SimpleNamespace(id=1), data='rp:8:mode:scene', answer=AsyncMock(),
                                    message=SimpleNamespace(edit_text=AsyncMock(), answer=AsyncMock()))
            await bot.on_choice(query)
            run.assert_awaited_once()
            turn_ctx, authorization = run.await_args.args[2], run.await_args.args[3]
            self.assertEqual(turn_ctx['scope_id'], '8')
            self.assertEqual(authorization['labels'], {'mode': 'scene', 'transition': 'current', 'span': 'brief'})
            self.assertNotIn(1, bot.PENDING_CHOICES)


class AutoSettleTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsure_mode_plays_the_scene_when_buttons_are_off(self):
        from telegram import roleplay_bot as bot
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='(맘대로 쉬어라)', message_id=4, chat=SimpleNamespace(id=1),
                                  answer=AsyncMock(), bot=SimpleNamespace(send_chat_action=AsyncMock()))
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        pending = jev.PendingChoice('mode', [('discussion', 'Discuss'), ('scene', 'Perform')], '불확정')
        with patch.object(bot.asyncio, 'to_thread', side_effect=inline_thread), \
             patch.object(bot.roleplay_turn, 'committed_reply', return_value=None), \
             patch.object(bot.roleplay_turn, 'authorize', side_effect=pending), \
             patch.object(bot, '_draft_and_settle', new_callable=AsyncMock) as run, \
             patch.object(bot, 'save_message', return_value=1), patch.object(bot, 'load_history', return_value=[]), \
             patch.object(bot, 'get_preference', return_value='off'), \
             patch.object(bot, 'load_notes', return_value=[]), patch.object(bot, 'load_state', return_value={'hunger': 25}), \
             patch.object(bot, 'people_context', return_value={'index': [], 'present': []}):
            await bot.handle_message(message)
        authorization = run.await_args.args[3]
        self.assertEqual(authorization['labels']['mode'], 'scene')
        self.assertEqual(authorization['auto_settled'], {'mode': 'scene'})
        message.answer.assert_not_awaited()

    async def test_buttons_off_takes_the_most_probable_value_and_says_so(self):
        from telegram import roleplay_bot as bot
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='죽을 건넸다', message_id=9, chat=SimpleNamespace(id=1),
                                  answer=AsyncMock(), bot=SimpleNamespace(send_chat_action=AsyncMock()))
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        @contextmanager
        def fake_stage(uid):
            yield {'people': []}
        auth = {'user_text': '죽을 건넸다', 'labels': {'mode': 'scene', 'transition': 'current', 'span': 'brief'}}
        pending = jev.PendingChoice('event', [('kindness', '배려·양보'), ('none', '뚜렷한 사건 없음')], '불확실')
        pending.verdict = {'labels': {'mode': 'scene', 'event_intake': 'meal'}, 'answers': {}}
        prepared = {'reply': '초안', 'applied': {'event': 'kindness', 'events': ['kindness', 'meal'], 'minutes': 3}}
        with patch.object(bot.asyncio, 'to_thread', side_effect=inline_thread), \
             patch.object(bot.roleplay_turn, 'committed_reply', return_value=None), \
             patch.object(bot.roleplay_turn, 'authorize', return_value=auth), \
             patch.object(bot.roleplay_turn, 'staged_memory', side_effect=fake_stage), \
             patch.object(bot.roleplay_turn, 'prepare', side_effect=[pending, prepared]) as prepare, \
             patch.object(bot, 'adjudicate_turn', side_effect=lambda *a, **k: {'status': 'applied', 'reply': '초안', 'applied': k['prepared']['applied']}) as settle, \
             patch.object(bot, 'save_message', return_value=5), patch.object(bot, 'load_history', return_value=[]), \
             patch.object(bot, 'get_preference', side_effect=lambda uid, key, default='': 'off' if key == bot.ASK_PREFERENCE else 'on'), \
             patch.object(bot, 'load_notes', return_value=[]), patch.object(bot, 'load_state', return_value={'hunger': 25, 'clock': {}}), \
             patch.object(bot, 'people_context', return_value={'index': [], 'present': []}), \
             patch.object(bot, 'build_system_prompt', return_value='sys'), \
             patch.object(bot, '_make_progress_callback', return_value=SimpleNamespace(flush=AsyncMock())), \
             patch.object(bot, 'chat_with_tools', new_callable=AsyncMock, return_value='초안'):
            await bot.handle_message(message)
        self.assertEqual(prepare.call_count, 2)
        second = prepare.call_args_list[1].args
        self.assertEqual(second[8]['labels']['event'], 'kindness')
        self.assertTrue(second[8]['player_settled'])
        self.assertNotIn(1, bot.PENDING_CHOICES)
        sent = [c.args[0] for c in message.answer.await_args_list]
        self.assertEqual(sent[0], '초안')
        self.assertIn('애매해서 자동 처리: 사건=배려·양보', sent[1])
        self.assertFalse(any('reply_markup' in c.kwargs for c in message.answer.await_args_list))
