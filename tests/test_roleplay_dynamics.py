import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_tools import roleplay_memory as memory
from runtime_tools.roleplay_dynamics import advance, with_defaults
from tool_gateway.security import caller_scope, new_run_context


class DynamicsTests(unittest.TestCase):
    def initial(self, **conditions):
        return with_defaults({"hunger": 30, "fatigue": 60, "pain": 40, "tension": 50,
                              "conditions_initialized": True, **conditions})

    def test_activity_sleep_and_no_wall_clock(self):
        rest = advance(self.initial(activity='rest'), 60, '한 시간 휴식')
        exertion = advance(self.initial(activity='strenuous'), 60, '한 시간 활동')
        sleep = advance(self.initial(activity='sleep', sleep_quality='good'), 60, '한 시간 수면')
        self.assertEqual((rest['hunger'], rest['fatigue']), (33, 58))
        self.assertEqual(exertion['fatigue'], 70)
        self.assertEqual(sleep['fatigue'], 50)
        self.assertEqual(advance(rest, 60, '같은 시점'), rest)
        self.assertEqual(rest['last_calculation']['before']['fatigue'], 60)

    def test_injury_trends_and_threat_adaptation(self):
        injury = {'id': 'arm', 'description': '팔 부상', 'severity': 2, 'trend': 'stable', 'treated': False}
        self.assertEqual(advance(self.initial(injuries=[injury]), 120, '두 시간')['pain'], 40)
        injury['trend'] = 'worsening'
        self.assertEqual(advance(self.initial(injuries=[injury]), 120, '두 시간')['pain'], 42)
        injury['treated'] = True
        self.assertEqual(advance(self.initial(injuries=[injury]), 120, '두 시간')['pain'], 41)
        injury['trend'] = 'recovering'
        self.assertEqual(advance(self.initial(injuries=[injury]), 120, '두 시간')['pain'], 38)
        self.assertEqual(advance(self.initial(threat='safe'), 120, '두 시간')['tension'], 38)
        self.assertEqual(advance(self.initial(threat='threatening'), 120, '두 시간')['tension'], 62)

    def test_partial_intervals_bounds_and_unknowns(self):
        state = self.initial()
        for minute in range(1, 61):
            state = advance(state, minute, '같은 활동의 다음 1분')
        whole = advance(self.initial(), 60, '한 시간')
        for key in ('hunger', 'fatigue', 'pain', 'tension'):
            self.assertAlmostEqual(state[key], whole[key], delta=.01)
        bounded = advance(self.initial(hunger=99, fatigue=1, activity='sleep'), 120, '두 시간')
        self.assertEqual((bounded['hunger'], bounded['fatigue']), (100, 0))
        self.assertIsNone(advance(self.initial(pain=None), 120, '두 시간')['pain'])
        for target in (-1, True, 1441):
            with self.assertRaises(ValueError):
                advance(self.initial(), target, '잘못된 시간')
        with self.assertRaises(ValueError):
            advance(self.initial(conditions_initialized=False), 60, '한 시간')


class StateTransactionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patcher = patch.object(memory, 'MEMORY_PATH', Path(temp.name) / 'memory.sqlite3')
        patcher.start()
        self.addCleanup(patcher.stop)
        scope = caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True))
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        memory.roleplay_state('update', {'hunger': 30, 'fatigue': 60, 'pain': 40, 'tension': 50,
            'activity': 'rest', 'sleep_quality': 'normal', 'threat': 'safe', 'injuries': []},
            '출발 상태', expected_revision=0, adjustment='initialize', event_id='initial',
            metric_reasons={k: '현재 장면의 기준값' for k in ('hunger', 'fatigue', 'pain', 'tension')})

    def time(self, *, reason, expected_revision, target_minute, time_basis):
        return memory.roleplay_state('time', reason=reason, expected_revision=expected_revision,
            event_id=f"passage-{target_minute}", temporal={
                'relation': 'current', 'certainty': 'explicit', 'source_quote': time_basis,
                'interpretation': time_basis, 'operation': 'advance',
                'elapsed_minutes': max(1, target_minute - memory.load_state(1)['scene_minute']),
            })

    def test_retry_conflict_history_and_reset(self):
        result = json.loads(self.time( reason='휴식', expected_revision=1,
                                               target_minute=60, time_basis='한 시간 휴식'))
        replay = json.loads(self.time( reason='재시도', expected_revision=1,
                                               target_minute=60, time_basis='한 시간 휴식'))
        self.assertEqual(result, replay)
        with self.assertRaises(ValueError):
            self.time( reason='오래된 상태', expected_revision=1, target_minute=120, time_basis='두 시간')
        self.assertEqual(len(json.loads(memory.roleplay_state('history'))), 2)
        with self.assertRaises(ValueError):
            memory.roleplay_state('update', {'pain': 60}, '그냥 상승', expected_revision=2)
        self.assertEqual(memory.load_state(1)['pain'], 40)
        memory.roleplay_state('reset', reason='다른 장면', expected_revision=2)
        self.assertEqual(memory.load_state(1)['last_calculated_minute'], 0)
        self.assertIsNone(memory.load_state(1)['pain'])
        self.assertFalse(memory.load_state(1)['conditions_initialized'])

    def test_immediate_meal_dedup_and_changed_conditions(self):
        memory.roleplay_state('update', {'hunger': 10}, '식사', expected_revision=1,
                             adjustment='event', event_id='meal-1', metric_reasons={'hunger': '식사 완료'})
        memory.roleplay_state('update', {'hunger': 0}, '재시도', expected_revision=1,
                             adjustment='event', event_id='meal-1', metric_reasons={'hunger': '식사 완료'})
        self.assertEqual(memory.load_state(1)['hunger'], 10)
        memory.roleplay_state('update', {'activity': 'strenuous'}, '활동 시작', expected_revision=2)
        self.time( reason='활동 지속', expected_revision=3, target_minute=60, time_basis='한 시간 활동')
        self.assertEqual(memory.load_state(1)['fatigue'], 70)
        records = json.loads(memory.roleplay_state('history'))
        self.assertEqual(records[0]['before']['fatigue'], 60)
        self.assertEqual(records[0]['after']['fatigue'], 70)
        with caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='2', is_owner=True)):
            self.assertEqual(json.loads(memory.roleplay_state('history')), [])


    def test_persisted_time_interpretation_and_event_timestamp(self):
        temporal = {'relation': 'current', 'certainty': 'explicit', 'operation': 'anchor',
                    'source_quote': '1939년 12월 31일 23시 50분', 'interpretation': '현재 시각 설정',
                    'date': '1939-12-31', 'time': '23:50'}
        memory.roleplay_state('time', reason='시간 설정', expected_revision=1, event_id='clock-start', temporal=temporal)
        memory.roleplay_state('time', reason='과거 회상', expected_revision=2, event_id='yesterday', temporal={
            'relation': 'past', 'certainty': 'explicit', 'operation': 'reference',
            'source_quote': '어제 한 시간 잤다', 'interpretation': '과거 사건 설명'})
        self.assertEqual(memory.load_state(1)['clock']['time'], '23:50')
        memory.roleplay_state('update', {'hunger': 10}, '식사', expected_revision=3,
            adjustment='event', event_id='dinner', event_type='meal', metric_reasons={'hunger': '식사 완료'})
        event = memory.load_state(1)['event_timestamps'][-1]
        self.assertEqual(event['type'], 'meal')
        self.assertEqual(event['clock']['time'], '23:50')
        record = json.loads(memory.roleplay_state('history'))[1]
        self.assertEqual(record['temporal']['relation'], 'past')
        self.assertEqual(record['before']['clock']['time'], record['after']['clock']['time'])
        memory.roleplay_state('reset', reason='새 장면', expected_revision=4)
        self.assertIsNone(memory.load_state(1)['clock']['date'])
