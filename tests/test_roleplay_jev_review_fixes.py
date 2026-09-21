"""Regressions for uncertain settlement and focused adjudication context."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm.call_registry import Decision, DecisionResult
from runtime_tools import roleplay_jev as jev
from runtime_tools.roleplay_pacing import policy_for, turn_time_scope
from test_roleplay_jev import initial, verdict


class SettlementTests(unittest.TestCase):
    def test_final_activity_rejects_recovery_even_after_defaulting(self):
        for event, act in [('sexual_assault', 'touch'), ('rape', 'penetration')]:
            for activity in (None, 'rest', 'sleep'):
                with self.subTest(event=event, activity=activity):
                    before = initial()
                    saved = deepcopy(before)
                    v = verdict(event=event, sexual_act=act, intensity='moderate', elapsed='explicit')
                    v['player_settled'] = True
                    if activity is None:
                        v['labels'].pop('activity')
                    else:
                        v['labels']['activity'] = activity
                    with turn_time_scope(policy_for('한 시간 진행')), self.assertRaisesRegex(ValueError, '휴식·수면'):
                        jev.project(before, '한 시간 진행', [], v, 'test')
                    self.assertEqual(before, saved)

    def test_pending_intensity_is_probability_ordered(self):
        v = verdict(event='beating', elapsed='0')
        v['answers'] = {'intensity': {'probabilities': {'mild': .05, 'moderate': .35, 'severe': .6}}}
        with turn_time_scope(policy_for('구타')), self.assertRaises(jev.PendingChoice) as caught:
            jev.project(initial(), '구타', [], v, 'test')
        self.assertEqual([k for k, _ in caught.exception.candidates], ['severe', 'moderate', 'mild'])
        v['player_settled'] = True
        with turn_time_scope(policy_for('구타')):
            _, applied = jev.project(initial(), '구타', [], v, 'test')
        self.assertEqual(applied['intensity'], 'severe')

    def test_pending_and_remaining_activity_use_latest_probabilities(self):
        v = verdict(elapsed='explicit')
        v['labels'].pop('activity')
        v['answers'] = {'activity': {'probabilities': {'restrained': .6, 'rest': .3, 'light': .1}}}
        with turn_time_scope(policy_for('한 시간 진행')), self.assertRaises(jev.PendingChoice) as caught:
            jev.project(initial(), '한 시간 진행', [], v, 'test')
        self.assertEqual(caught.exception.candidates[0][0], 'restrained')
        v['player_settled'] = True
        with turn_time_scope(policy_for('한 시간 진행')):
            state, _ = jev.project(initial(), '한 시간 진행', [], v, 'test')
        self.assertEqual(state['activity'], 'restrained')

    def test_invalid_probabilities_use_explicit_default(self):
        choices = [('mild', 'm'), ('moderate', 'n'), ('severe', 's')]
        for bad in (None, '0.9', True, -1, 2, float('nan'), float('inf')):
            v = {'answers': {'intensity': {'probabilities': {'mild': bad, 'severe': bad}}}}
            self.assertEqual(jev.ranked_candidates(v, 'intensity', choices, 'moderate')[0][0], 'moderate')


class ClassificationTests(unittest.TestCase):
    def test_retry_has_current_context_and_total_time_includes_all_groups(self):
        profile = SimpleNamespace(extra={'enabled': True, 'thresholds': {'accept': .65, 'secondary': .5}})
        requests = []
        def decide(feature, state, questions, **kwargs):
            group = kwargs['label']
            requests.append((group, state))
            answers = {}
            for key, question in questions.items():
                label = next(iter(question['criteria']))
                if key in jev.FAMILY_KEYS:
                    label = 'none'
                confidence = .99
                if key in ('activity', 'location') and group == 'roleplay-scene':
                    confidence = .1
                if group == 'roleplay-event-review':
                    label = 'keep' if key == 'location' else 'light'
                    self.assertEqual(state['current']['location'], '복도')
                    self.assertEqual(state['current']['activity'], 'light')
                    self.assertNotIn('injuries', state['current'])
                answers[key] = {'choice': label, 'confidence': confidence}
            return DecisionResult(decision=Decision(answers=answers, model='fixture', latency_ms=100, cost_usd=.001))
        with patch.object(jev, 'resolve', return_value=profile), patch.object(jev, 'decide_detailed', side_effect=decide), \
             patch.object(jev.time, 'monotonic', side_effect=[10, 11.25]):
            result = jev.classify('그곳에 있어', initial(), [{'person_id': 'guard', 'name': '간수'}], [],
                                  draft='그는 움직이지 않았다.', authorization={'labels': {'mode': 'scene'}})
        self.assertEqual([g for g, _ in requests], ['roleplay-scene', 'roleplay-people', 'roleplay-records', 'roleplay-event-review'])
        self.assertEqual(result['latency_ms'], 1250)
        self.assertEqual(result['calls']['roleplay-scene']['latency_ms'], 100)
        self.assertEqual(result['event_review']['latency_ms'], 100)
        self.assertEqual(result['labels']['location'], 'keep')
        self.assertAlmostEqual(result['cost_usd'], .004)

    def test_failed_scene_includes_elapsed_time(self):
        with patch.object(jev, 'decide_detailed', return_value=DecisionResult(error_kind='transport')), \
             patch.object(jev.time, 'monotonic', side_effect=[10, 12]):
            result = jev.classify('이동', initial(), [], [])
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['latency_ms'], 2000)
