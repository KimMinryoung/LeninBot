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
            event_id=f"passage-{target_minute}", interval_conditions={"activity": memory.load_state(1)["activity"]},
            person_updates=[], person_review="인물의 새 정보 없음", temporal={
                'relation': 'current', 'certainty': 'explicit', 'source_quote': time_basis,
                'interpretation': time_basis, 'operation': 'advance',
                'elapsed_minutes': max(1, target_minute - memory.load_state(1)['scene_minute']),
            })

    def test_retry_conflict_history_and_reset(self):
        result = json.loads(self.time( reason='휴식', expected_revision=1,
                                               target_minute=60, time_basis='한 시간 휴식'))
        replay = json.loads(self.time( reason='재시도', expected_revision=1,
                                               target_minute=60, time_basis='한 시간 휴식'))
        self.assertTrue(replay.pop('replayed'))
        replay.pop('note')
        self.assertEqual(result, replay)
        with self.assertRaises(ValueError):
            self.time( reason='오래된 상태', expected_revision=1, target_minute=120, time_basis='두 시간')
        self.assertEqual(len(json.loads(memory.roleplay_state('history'))), 2)
        # A bare numeric change is an event with the reason as its evidence, not a failure.
        result = json.loads(memory.roleplay_state('update', {'pain': 60}, '갑작스런 타격', expected_revision=2))
        self.assertEqual(memory.load_state(1)['pain'], 60)
        self.assertTrue(any('adjustment' in w for w in result['warnings']))
        self.assertTrue(any('event_id' in w for w in result['warnings']))
        self.assertEqual(memory.load_state(1)['event_timestamps'][-1]['reason'], '갑작스런 타격')
        memory.roleplay_state('reset', reason='다른 장면', expected_revision=3)
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

    def test_explicit_interval_and_atomic_person_review(self):
        memory.roleplay_person('save', 'rodos', changes={'name': '로도스', 'observed': '이전 사건'})
        memory.roleplay_state('update', {'activity': 'strenuous'}, '격한 활동', expected_revision=1)
        temporal = {'relation': 'current', 'certainty': 'estimated', 'operation': 'advance',
                    'elapsed_minutes': 3, 'source_quote': '물을 마셨다', 'interpretation': '물 마시는 3분'}
        kwargs = dict(reason='물 마심', expected_revision=2, event_id='water', temporal=temporal)
        with self.assertRaises(ValueError):
            memory.roleplay_state('time', **kwargs)
        with self.assertRaises(ValueError):
            memory.roleplay_state('time', **kwargs, interval_conditions={'activity': 'light'},
                person_updates=[{'person_id': 'rodos', 'changes': {'observed': '갱신'}},
                                {'person_id': 'missing', 'changes': {'observed': '실패'}}])
        self.assertEqual(memory.load_state(1)['scene_minute'], 0)
        self.assertEqual(memory.load_people(1)[0]['observed'], '이전 사건')
        memory.roleplay_state('time', **kwargs, interval_conditions={'activity': 'light'},
            person_updates=[{'person_id': 'rodos', 'changes': {'observed': '이전 사건. 물을 건넴'}}])
        self.assertEqual(memory.load_state(1)['fatigue'], 60.1)
        self.assertEqual(memory.load_state(1)['activity'], 'light')
        self.assertIn('물을 건넴', memory.load_people(1)[0]['observed'])
        before = memory.load_state(1)
        memory.roleplay_state('time', **kwargs, interval_conditions={'activity': 'light'},
            person_updates=[{'person_id': 'rodos', 'changes': {'observed': '재시도'}}])
        self.assertEqual(memory.load_state(1), before)
        self.assertIn('물을 건넴', memory.load_people(1)[0]['observed'])
        # Skipping the people review is a reminder in the result, not a lost turn.
        memory.roleplay_state('update', {'last_event': '새 사건', 'participants': ['rodos']}, '새 사건', expected_revision=3)
        result = json.loads(memory.roleplay_state('update', {'last_event': '또 다른 사건'}, '사건', expected_revision=4))
        self.assertIn('로도스', result['people_reminder'])
        result = json.loads(memory.roleplay_state('update', {'last_event': '검토한 사건'}, '사건', expected_revision=5,
                                                  person_updates=[], person_review='로도스는 말없이 서 있었음'))
        self.assertNotIn('people_reminder', result)

    def test_one_call_interval_then_changes(self):
        memory.roleplay_person('save', 'rodos', changes={'name': '로도스', 'aliases': ['보리스']})
        temporal = {'relation': 'current', 'certainty': 'explicit', 'operation': 'advance', 'elapsed_minutes': 60,
                    'source_quote': '한 시간 뒤 식사가 왔다', 'interpretation': '벽에 기대 쉰 한 시간, 그 뒤 식사',
                    'date': '1939-04-20', 'time': '10:00'}
        # The interval is rest; what follows it (a meal, a visitor, a new activity) rides in changes.
        result = json.loads(memory.roleplay_state('time', {
            'hunger': 5, 'activity': 'light', 'threat': 'threatening', 'last_event': '간수가 식사를 두고 갔다',
            'participants': ['보리스'], 'reason': '식사 도착',
        }, expected_revision=1, event_id='meal-arrives', event_type='meal', temporal=temporal,
            interval_conditions={'activity': 'rest'}, person_updates=[{'person_id': '보리스', 'observed': '식사를 가져옴'}]))
        state = memory.load_state(1)
        self.assertEqual(state['scene_minute'], 60)
        self.assertEqual(state['fatigue'], 58)  # one hour of rest, then the meal event
        self.assertEqual(state['hunger'], 5)
        self.assertEqual(state['activity'], 'light')
        self.assertEqual(state['threat'], 'threatening')
        self.assertEqual(state['participants'], ['rodos'])
        self.assertEqual(state['last_calculation']['conditions']['activity'], 'rest')
        self.assertEqual(state['event_timestamps'][-1]['type'], 'meal')
        self.assertEqual(state['event_timestamps'][-1]['scene_minute'], 60)
        self.assertIsNone(state['clock']['date'])  # advance never takes a second target
        self.assertEqual(memory.load_people(1)[0]['observed'], '식사를 가져옴')
        self.assertEqual(state['reason'], '식사 도착')
        joined = ' '.join(result['warnings'])
        for fragment in ('reason', 'participants', 'advance', 'adjustment'):
            self.assertIn(fragment, joined)
        self.assertNotIn('recent_events', result)
        self.assertEqual(result['hunger'], 5)
        # Without interval_conditions, the activity in changes is taken for the interval too, with a warning.
        result = json.loads(memory.roleplay_state('time', {'activity': 'strenuous'}, '강요가 이어짐', expected_revision=2,
            event_id='assault', temporal={**temporal, 'elapsed_minutes': 30}))
        self.assertEqual(memory.load_state(1)['fatigue'], 63)
        self.assertTrue(any('interval_conditions' in w for w in result['warnings']))
        with self.assertRaises(ValueError):
            memory.roleplay_state('time', {'mood': '지침'}, '활동 불명', expected_revision=3,
                                  event_id='no-activity', temporal={**temporal, 'elapsed_minutes': 5})

    def test_same_turn_revision_tolerance_and_partial_initialize(self):
        with caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True, scope_id='msg-9')):
            memory.roleplay_state('update', {'mood': '경계'}, '첫 변경', expected_revision=1)
            result = json.loads(memory.roleplay_state('update', {'mood': '긴장'}, '같은 턴', expected_revision=1))
            self.assertEqual(result['revision'], 3)
            self.assertTrue(any('같은 턴' in w for w in result['warnings']))
            result = json.loads(memory.roleplay_state('update', {'mood': '누락'}, '같은 턴, 버전 누락'))
            self.assertEqual(result['revision'], 4)
        with caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True, scope_id='msg-10')):
            with self.assertRaises(ValueError):
                memory.roleplay_state('update', {'mood': '다른 턴'}, '오래된 버전', expected_revision=3)
        memory.roleplay_state('update', {'pain': 55}, '정정', expected_revision=4, adjustment='correction', event_id='fix')
        result = json.loads(memory.roleplay_state('update', {'pain': 10, 'hunger': 20}, '초기화 시도', expected_revision=5,
                                                  adjustment='initialize', event_id='init-again'))
        self.assertEqual((memory.load_state(1)['pain'], memory.load_state(1)['hunger']), (55, 30))
        self.assertTrue(any('initialize' in w for w in result['warnings']))
        memory.roleplay_state('reset', reason='새 장면', expected_revision=6)
        result = json.loads(memory.roleplay_state('update', {'pain': 10, 'hunger': 20}, '새 장면 기준', expected_revision=7,
                                                  adjustment='initialize', event_id='init-fresh'))
        self.assertEqual((result['pain'], result['hunger']), (10, 20))

    def test_reset_with_initial_scene_and_injury_upsert(self):
        result = json.loads(memory.roleplay_state('reset', {
            'hunger': 20, 'fatigue': 10, 'pain': 0, 'tension': 30, 'activity': 'light', 'sleep_quality': 'normal',
            'threat': 'safe', 'injuries': [], 'location': '집무실', 'goal': '보고서 마무리',
        }, '1937년 집무실 장면으로 전환', expected_revision=1, event_id='new-scene'))
        state = memory.load_state(1)
        self.assertEqual((state['hunger'], state['location'], state['scene_minute']), (20, '집무실', 0))
        self.assertTrue(state['conditions_initialized'])
        self.assertEqual(result['revision'], 2)
        result = json.loads(memory.roleplay_state('update', {
            'hand_cut': {'description': '왼손 베임', 'severity': 1},
            'injuries': [{'id': 'bruise', 'description': '멍', 'severity': 1, 'trend': 'stable', 'treated': False}],
        }, '유리에 베임', expected_revision=2))
        self.assertEqual([i['id'] for i in memory.load_state(1)['injuries']], ['bruise', 'hand_cut'])
        self.assertEqual(memory.load_state(1)['injuries'][1]['trend'], 'stable')
        self.assertTrue(any('injuries' in w for w in result['warnings']))
        with self.assertRaises(ValueError):
            memory.roleplay_state('update', {'unknown_field': 'x'}, '모르는 필드', expected_revision=3)
        with self.assertRaises(ValueError):
            memory.roleplay_state('update', {'mood': '기타'}, '모르는 인자', expected_revision=3, bogus=1)
