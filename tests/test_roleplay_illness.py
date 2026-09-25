import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from llm.call_registry import Decision, DecisionResult
from roleplay import illness, jev, memory
from roleplay.actor import actor_state_view
from roleplay.dynamics import advance, with_defaults
from roleplay.pacing import policy_for, turn_time_scope
from tool_gateway.security import caller_scope, new_run_context
from test_roleplay_jev import initial, verdict


def diagnosed(**changes):
    record = dict(kind='pneumonia', status='active', severity='moderate', treatment='untreated',
                  fever='yes', elapsed_minutes=0, source='현재 폐렴이 확인됨', scope_id='diagnosis')
    return {**record, **changes}


class IllnessTests(unittest.TestCase):
    def test_legacy_defaults_are_isolated_and_do_not_infer_disease(self):
        old = {'body': '열이 난다', 'scene': '사료에서 폐렴 치료를 읽었다'}
        first, second = with_defaults(old), with_defaults(old)
        self.assertEqual(first['illnesses'], [])
        first['illnesses'].append(diagnosed())
        self.assertEqual(second['illnesses'], [])

    def test_lifecycle_treatment_is_not_cure_and_recurrence_resets_episode(self):
        records, _ = illness.settle([], {'disease_pneumonia': 'active'}, '현재 폐렴 진단', '1')
        self.assertEqual(records[0]['severity'], 'unknown')
        records, _ = illness.settle(records, {'disease_pneumonia_treatment': 'treated'}, '폐렴 치료를 받았다', '2')
        self.assertEqual(records[0]['status'], 'active')
        records = illness.advance_illnesses(records, 120)
        records, _ = illness.settle(records, {'disease_pneumonia': 'resolved'}, '완치 확인', '3')
        self.assertEqual(illness.rates(records), {'fatigue': 0, 'clarity': 0})
        self.assertEqual(illness.advance_illnesses(records, 60)[0]['elapsed_minutes'], 120)
        records, _ = illness.settle(records, {'disease_pneumonia': 'active'}, '다시 폐렴 발생', '4')
        self.assertEqual(records[0]['elapsed_minutes'], 0)
        self.assertEqual(records[0]['treatment'], 'unknown')

    def test_symptoms_and_treatment_alone_do_not_create_diagnosis(self):
        records, changes = illness.settle([], {'disease_pneumonia_fever': 'yes',
                                              'disease_pneumonia_treatment': 'treated'}, '상처를 처치받았다', '1')
        self.assertEqual((records, changes), ([], []))

    def test_fatigue_and_clarity_effects_split_equally_and_do_not_cure(self):
        base = initial(activity='light', threat='safe', illnesses=[])
        sick = {**deepcopy(base), 'illnesses': [diagnosed()]}
        healthy = advance(base, 60, 'one hour')
        whole = advance(sick, 60, 'one hour')
        split = advance(advance(sick, 30, 'half'), 60, 'half')
        self.assertAlmostEqual(whole['fatigue'] - healthy['fatigue'], 1, places=3)
        self.assertAlmostEqual(healthy['clarity'] - whole['clarity'], 1, places=3)
        for key in ('fatigue', 'clarity', 'illnesses'):
            self.assertEqual(whole[key], split[key])
        self.assertEqual(whole['illnesses'][0]['elapsed_minutes'], 60)
        self.assertEqual(whole['illnesses'][0]['status'], 'active')
        self.assertEqual(illness.rates([diagnosed(status='recovering')]), {'fatigue': .5, 'clarity': -.5})

    def test_onset_applies_at_endpoint_and_discussion_does_not_diagnose(self):
        text = '한 시간 뒤 현재 폐렴을 진단받았다'
        v = verdict(elapsed='explicit', activity='light', disease_pneumonia='active',
                    disease_pneumonia_severity='moderate', disease_pneumonia_fever='yes')
        with turn_time_scope(policy_for(text)):
            state, applied = jev.project(initial(), text, [], v, '1')
        self.assertEqual(state['illnesses'][0]['elapsed_minutes'], 0)
        self.assertEqual(applied['illness_changes'][0]['kind'], 'pneumonia')
        v['labels']['mode'] = 'discussion'
        state, _ = jev.project(initial(), '폐렴이면 어떻게 될까?', [], v, '2')
        self.assertEqual(state['illnesses'], [])

    def test_unknown_or_failed_jev_labels_preserve_existing_illness(self):
        before = initial(illnesses=[diagnosed()])
        def answer(feature, payload, questions, **kwargs):
            if kwargs['label'] == 'roleplay-records':
                self.assertEqual(payload['current']['illnesses'], before['illnesses'])
                return DecisionResult(error_kind='transport')
            return DecisionResult(decision=Decision(model='test', answers={
                key: {'choice': next(iter(question['criteria'])), 'confidence': .99}
                for key, question in questions.items()}))
        with patch.object(jev, 'decide_detailed', side_effect=answer):
            classified = jev.classify('현재 장면', before, [], [])
        self.assertIn('disease_pneumonia', classified['uncertain'])
        self.assertEqual(illness.settle(before['illnesses'], classified['labels'], '장면', '1')[0], before['illnesses'])

    def test_scheduled_interruption_does_not_apply_later_diagnosis(self):
        from roleplay.story import apply_story_updates
        before = apply_story_updates(initial(), [{'op': 'schedule', 'id': 'visit', 'title': '방문',
                                                   'source': '예약', 'due_minute': 5}])
        text = '20분 쉬어'
        with turn_time_scope(policy_for(text)):
            state, applied = jev.project(before, text, [], verdict(
                elapsed='explicit', activity='rest', disease_pneumonia='active'), 'stop')
        self.assertTrue(applied['interrupted'])
        self.assertEqual(state['scene_minute'], 5)
        self.assertEqual(state['illnesses'], [])

    def test_scene_reset_clears_illness(self):
        text = '새 장면으로 초기화해'
        with turn_time_scope(policy_for(text, mode='reset')):
            state, _ = jev.project(initial(illnesses=[diagnosed()]), text, [], verdict(mode='reset'), 'reset')
        self.assertEqual(state['illnesses'], [])

    def test_actor_gets_qualitative_state_not_clocks_or_rates(self):
        view = actor_state_view(initial(illnesses=[diagnosed(elapsed_minutes=120)]))['illnesses'][0]
        self.assertEqual(view['name'], '폐렴')
        self.assertEqual(view['fever'], '발열 있음')
        self.assertNotIn('elapsed_minutes', view)
        self.assertNotIn('source', view)
        self.assertNotIn('fatigue', view)

    def test_validation_rejects_invalid_records(self):
        for changes in ({'kind': 'made_up'}, {'elapsed_minutes': True}, {'elapsed_minutes': -1},
                        {'severity': 100}, {'status': 'dead'}, {'treatment': 'magic'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                illness.validate_illnesses([diagnosed(**changes)])
        with self.assertRaises(ValueError):
            illness.validate_illnesses([diagnosed(), diagnosed()])

    def test_persistence_replay_and_actor_write_protection(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(memory, 'MEMORY_PATH', Path(temp) / 'state.db'):
            with memory._connection() as conn:
                conn.execute('INSERT INTO character_state VALUES (?,?)', ('1', json.dumps(initial())))
            ctx = new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True,
                                  scope_type='telegram_message', scope_id='1')
            v = verdict(elapsed='0', disease_pneumonia='active', disease_pneumonia_severity='moderate')
            with caller_scope(ctx), patch.object(jev, 'classify', return_value=v), turn_time_scope(policy_for('현재 폐렴 진단')):
                outcome = jev.adjudicate_turn(1, '현재 폐렴 진단', [], '1')
                self.assertEqual(outcome['status'], 'applied')
                saved = memory.load_state(1)
                self.assertEqual(saved['illnesses'][0]['kind'], 'pneumonia')
                self.assertTrue(jev.adjudicate_turn(1, '현재 폐렴 진단', [], '1')['replayed'])
                self.assertEqual(memory.load_state(1), saved)
                with self.assertRaises(PermissionError):
                    memory.roleplay_state('update', changes={'illnesses': []}, reason='임의 완치')
