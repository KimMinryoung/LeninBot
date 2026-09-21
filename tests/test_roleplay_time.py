import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from runtime_tools import roleplay_time as timing, roleplay_turn as turn, roleplay_jev as jev
from runtime_tools.roleplay_pacing import duration_minutes, turn_time_scope
from test_roleplay_jev import initial
from llm.call_registry import Decision, DecisionResult


class TimeAuthorizationTests(unittest.TestCase):
    def test_evening_endpoint_uses_routine_and_passes_the_full_explicit_interval(self):
        before = initial()
        before['clock'].update(date='1939-04-28', time='14:35')
        before['routine'] = [{'id': 'dinner', 'time': '18:00', 'title': '저녁 배식', 'kind': 'meal'}]
        text = '(저녁까지 시간을 보낸다)'
        value = dict(mode='scene', transition='current', span='brief', time_scope='explicit',
                     duration_minutes=205, corrections={}, appointment=None, reason='18:00 저녁 배식까지')
        result = SimpleNamespace(text=json.dumps(value), error_kind=None, truncated=False, latency_ms=1)
        with patch.object(timing, 'generate_detailed', return_value=result) as generate:
            auth = turn.authorize(text, before, [])
        self.assertEqual(json.loads(generate.call_args.args[1])['current']['routine'], before['routine'])
        self.assertEqual(turn.policy_for_authorization(auth).max_minutes, 205)
        self.assertEqual(turn.expected_stop(auth, before), {'title': '18:00 저녁 배식', 'minutes': 205})
        verdict = {'status': 'classified', 'labels': {'mode': 'scene', 'event': 'none', 'activity': 'rest'}}
        with patch.object(jev, 'classify', return_value=verdict), patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 1}) as estimate, \
             patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}):
            prepared = turn.prepare(text, before, [], [], 'evening', '저녁 배식이 왔다.', auth)
        estimate.assert_called_once()
        self.assertEqual(prepared['applied']['minutes'], 205)
        self.assertEqual(prepared['state']['clock']['time'], '18:00')

    def test_open_rest_direction_cannot_reach_a_later_mentioned_appointment(self):
        auth = self.authorize('(밥 먹고 쉬어. 밤 9시에 심문이 있다.)', time_scope='open_ended', duration_minutes=0)
        before = initial()
        before['clock'].update(date='1939-04-28', time='17:35')
        direction = turn.direction(auth, before)
        self.assertIn('1939-04-28 17:35', direction)
        self.assertIn('1939-04-28 20:35', direction)

    def authorize(self, text, **changes):
        value = dict(mode='scene', transition='current', span='brief', time_scope='explicit',
                     duration_minutes=20, corrections={}, appointment=None, reason='Only the current rest is authorized')
        value.update(changes)
        result = SimpleNamespace(text=json.dumps(value), error_kind=None, truncated=False, latency_ms=1)
        with patch.object(timing, 'generate_detailed', return_value=result), \
             patch.object(turn, 'decide_detailed', side_effect=AssertionError('Jev must not authorize time')):
            return turn.authorize(text, initial(), [])

    def test_current_duration_is_not_sum_of_all_numbers(self):
        text = '어제 3시간 잤다. 지금 20분 쉬어.'
        auth = self.authorize(text)
        policy = turn.policy_for_authorization(auth)
        self.assertEqual(policy.max_minutes, 20)
        with turn_time_scope(policy):
            self.assertEqual(duration_minutes(text), 20)
            verdict = {'labels': {'mode': 'scene', 'event': 'none', 'elapsed': 'explicit', 'activity': 'rest'}}
            state, applied = jev.project(initial(), text, [], verdict, 'scope')
            self.assertEqual(applied['minutes'], 20)
            self.assertEqual(state['scene_minute'], 20)
        with patch.object(jev, 'classify', return_value={'status': 'classified', **deepcopy(verdict)}), \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 1}) as estimate, \
             patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}):
            prepared = turn.prepare(text, initial(), [], [], 'scope', '20분 쉬었다', auth)
        estimate.assert_called_once()
        self.assertEqual(prepared['applied']['minutes'], 20)

    def test_draft_jev_does_not_reclassify_llm_time(self):
        auth = self.authorize('지금 20분 쉬어')
        def answer(feature, payload, questions, **kwargs):
            self.assertNotIn('mode', questions)
            self.assertNotIn('elapsed', questions)
            answers = {key: {'choice': next(iter(q['criteria'])), 'confidence': .99}
                       for key, q in questions.items()}
            return DecisionResult(decision=Decision(model='test', answers=answers))
        with patch.object(jev, 'decide_detailed', side_effect=answer):
            verdict = jev.classify('지금 20분 쉬어', initial(), [], [], draft='쉬었다', authorization=auth)
        self.assertEqual(verdict['labels']['mode'], 'scene')
        self.assertEqual(verdict['labels']['elapsed'], 'explicit')

    def test_implicit_duration_and_morning_use_llm_labels(self):
        auth = self.authorize('아침에는 움직이지 마. 지금 여기 있어.', time_scope='none', duration_minutes=0)
        self.assertFalse(turn.policy_for_authorization(auth).calendar_skip)
        auth = self.authorize('다음 장면은 내일 아침.', transition='next_morning', time_scope='day_skip', duration_minutes=0)
        self.assertTrue(turn.policy_for_authorization(auth).calendar_skip)

    def test_plan_delay_does_not_advance_scene(self):
        text = '어제 3시간 잤다. 20분 뒤 방문 약속을 등록해.'
        auth = self.authorize(text, mode='plan', time_scope='none', appointment={'title': '방문'})
        with turn_time_scope(turn.policy_for_authorization(auth)):
            state, applied = jev.project(initial(), text, [], {'labels': {'mode': 'plan', 'plan_action': 'schedule'}}, 'plan')
        self.assertEqual(state['scene_minute'], 0)
        self.assertEqual(state['story_events'][0]['due_minute'], 20)

    def test_invalid_or_inconsistent_output_fails_without_regex_fallback(self):
        for changes in ({'duration_minutes': True}, {'duration_minutes': -1}, {'duration_minutes': 10081},
                        {'time_scope': 'none'}, {'transition': 'next_day'}, {'mode': 'discussion'},
                        {'reason': ''}, {'mode': 'invented'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.authorize('2시간 쉬어', **changes)
        for result in (SimpleNamespace(text='bad', error_kind=None, truncated=False),
                       SimpleNamespace(text='', error_kind='transport', truncated=False),
                       SimpleNamespace(text='{}', error_kind=None, truncated=True)):
            with patch.object(timing, 'generate_detailed', return_value=result), self.assertRaises(ValueError):
                turn.authorize('2시간 쉬어', initial(), [])

    def test_correction_uses_llm_final_values_and_preserves_other_state(self):
        text = '허기 60이 아니라 20으로 정정해. 피로는 영으로.'
        auth = self.authorize(text, mode='correction', time_scope='none', duration_minutes=0,
                              corrections={'hunger': 20, 'fatigue': 0})
        before = initial(metric_remainders={'hunger': .2, 'pain': .3})
        with patch.object(jev, 'classify', return_value={'status': 'classified', 'labels': {'mode': 'correction'}}), \
             patch.object(jev, 'estimate_duration') as estimate, \
             patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}):
            prepared = turn.prepare(text, before, [], [], 'fix', '정정했다.', auth)
        estimate.assert_not_called()
        self.assertEqual(prepared['applied']['corrected'], {'hunger': 20, 'fatigue': 0})
        expected = deepcopy(before)
        expected.update(hunger=20, fatigue=0, metric_remainders={'pain': .3})
        self.assertEqual(prepared['state'], expected)
        self.assertEqual(prepared['verdict']['authorization']['corrections'], auth['corrections'])

    def test_correction_contract_rejects_invalid_values_and_wrong_mode(self):
        for values in ({}, None, [], {'unknown': 5}, {'hunger': True}, {'hunger': '20'},
                       {'hunger': -1}, {'hunger': 101}, {'hunger': float('nan')}, {'hunger': float('inf')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.authorize('허기 20', mode='correction', time_scope='none', duration_minutes=0,
                               corrections=values)
        for mode in ('scene', 'discussion', 'plan', 'reset'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.authorize('허기 20?', mode=mode, time_scope='none', duration_minutes=0,
                               corrections={'hunger': 20})

    def test_no_regex_fallback_for_missing_or_stale_correction(self):
        text = '의지를 40으로 정정해'
        auth = self.authorize(text, mode='correction', time_scope='none', duration_minutes=0,
                              corrections={'resolve': 40})
        for authorization in (None, {**auth, 'corrections': {}}, {**auth, 'user_text': '다른 입력'},
                              {**auth, 'corrections': {'resolve': 200}}):
            with self.subTest(authorization=authorization), turn_time_scope(turn.policy_for_authorization(auth)), \
                 self.assertRaises(ValueError):
                jev.project(initial(), text, [], {'labels': {'mode': 'correction'},
                                                'authorization': authorization}, 'fix')

    def test_authorization_receives_current_metrics_for_relative_corrections(self):
        result = SimpleNamespace(text=json.dumps(dict(mode='correction', transition='current', span='brief',
            time_scope='none', duration_minutes=0, corrections={'resolve': 65.5}, appointment=None, reason='60에서 5.5 올림')),
            error_kind=None, truncated=False, latency_ms=1)
        with patch.object(timing, 'generate_detailed', return_value=result) as generate:
            auth = turn.authorize('의지 수치를 5.5 올려서 정정해', initial(), [])
        payload = json.loads(generate.call_args.args[1])
        self.assertEqual(payload['current']['resolve'], 60)
        self.assertEqual(auth['corrections'], {'resolve': 65.5})
