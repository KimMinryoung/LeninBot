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
        # Pain above the floor (8 for this wound) is acute and halves every two resting hours.
        stable = advance(self.initial(injuries=[injury]), 120, '두 시간')['pain']
        self.assertEqual(stable, 24)
        injury['trend'] = 'worsening'
        worse = advance(self.initial(injuries=[injury]), 120, '두 시간')['pain']
        self.assertAlmostEqual(worse, stable + 2, delta=0.3)  # +1/h drift on top of the decay
        injury['treated'] = True
        treated = advance(self.initial(injuries=[injury]), 120, '두 시간')['pain']
        self.assertLess(treated, worse)
        injury['trend'] = 'recovering'
        self.assertLess(advance(self.initial(injuries=[injury]), 120, '두 시간')['pain'], treated)
        self.assertEqual(advance(self.initial(threat='safe', participants=['x']), 120, '두 시간')['tension'], 38)
        self.assertEqual(advance(self.initial(threat='threatening', participants=['x']), 120, '두 시간')['tension'], 62)

    def test_acute_pain_subsides_by_activity(self):
        wounds = [{'id': 'cheek', 'description': '뺨 타박', 'severity': 1, 'trend': 'stable', 'treated': False},
                  {'id': 'glute', 'description': '볼기 멍', 'severity': 2, 'trend': 'stable', 'treated': True}]
        floor = 9.76  # noisy-or of 4 and 6
        beaten = self.initial(pain=68, injuries=wounds, participants=['guard'])
        # A night of poor sleep after a beating brings pain back close to what the wounds imply.
        night = advance({**beaten, 'activity': 'sleep', 'sleep_quality': 'poor', 'participants': []}, 450, '밤')
        self.assertLess(night['pain'], floor + 5)
        self.assertGreaterEqual(night['pain'], floor)
        # Rest halves the excess every two hours; light activity every three; exertion holds it.
        self.assertAlmostEqual(advance(beaten, 120, '두 시간 휴식')['pain'], floor + (68 - floor) / 2, delta=0.01)
        self.assertAlmostEqual(advance({**beaten, 'activity': 'light'}, 180, '세 시간')['pain'], floor + (68 - floor) / 2 + 0.9, delta=0.2)  # plus 0.3/h of movement
        self.assertEqual(advance({**beaten, 'activity': 'strenuous'}, 60, '강요')['pain'], 68 + 1.5 * 3)
        # Decay never goes below the floor, and a floor-level pain stays put under stable wounds.
        self.assertEqual(advance(self.initial(pain=floor, injuries=wounds), 600, '열 시간')['pain'], floor)
        # Long passages are stepped hourly, so a drain keyed to pain > 60 stops once pain has eased.
        stepped = advance({**beaten, 'threat': 'uncertain', 'resolve': 40, 'clarity': 60, 'humiliation': 50}, 480, '여덟 시간')
        self.assertGreater(stepped['resolve'], 40 - 8 + 0.5 * 8 - 3)  # at most a few hours of the pain drain
        self.assertEqual(stepped['last_calculation']['from_minute'], 0)
        self.assertEqual(stepped['last_calculation']['before']['pain'], 68)

    def test_alone_at_rest_relieves_threat(self):
        mental = dict(resolve=11, clarity=58, humiliation=100)
        alone = self.initial(threat='threatening', activity='sleep', sleep_quality='poor', pain=45, participants=[], **mental)
        night = advance(alone, 448, '홀로 보낸 밤')
        self.assertEqual(night['threat'], 'uncertain')
        self.assertTrue(night['last_calculation']['threat_relieved'])
        self.assertEqual(night['last_calculation']['conditions']['threat'], 'uncertain')
        self.assertGreater(night['resolve'], 11)  # no threat drain while nobody is there
        self.assertEqual(night['calm_minutes'], 448)
        # Someone present, a short absence, or an interval of exertion keeps the threat as given.
        with_guard = advance({**alone, 'participants': ['guard']}, 448, '간수와 밤')
        self.assertEqual((with_guard['threat'], with_guard['resolve']), ('threatening', 0))
        self.assertFalse(with_guard['last_calculation']['threat_relieved'])
        self.assertEqual(advance(alone, 45, '잠깐')['threat'], 'threatening')
        self.assertEqual(advance({**alone, 'activity': 'strenuous'}, 120, '강요')['threat'], 'threatening')
        self.assertEqual(advance({**alone, 'threat': 'immediate', 'activity': 'rest'}, 60, '혼자 휴식')['threat'], 'uncertain')

    def test_pain_floor_and_healing_timeline(self):
        from runtime_tools.roleplay_dynamics import injury_pain_floor, progress_injuries, carry_injury_progress
        burn = {'id': 'burn', 'description': '화상', 'severity': 3, 'trend': 'recovering', 'treated': True}
        cut = {'id': 'cut', 'description': '열상', 'severity': 2, 'trend': 'stable', 'treated': False}
        self.assertEqual(injury_pain_floor([burn, cut]), 17.66)  # 10.5 and 8, noisy-or
        self.assertEqual(injury_pain_floor([]), 0)
        # Recovery drifts down but stops at the floor the wounds imply.
        long_rest = advance(self.initial(pain=30, injuries=[burn, cut]), 1440, '하루 휴식')
        self.assertEqual(long_rest['pain'], 17.66)
        # After an event left pain below the floor, it climbs back at the approach rate.
        after_event = advance(self.initial(pain=5, injuries=[burn, cut]), 120, '두 시간')
        self.assertEqual(after_event['pain'], 11)
        self.assertEqual(advance(self.initial(pain=5, injuries=[burn, cut]), 600, '열 시간')['pain'], 17.66)
        # Severe pain halves what rest and sleep recover.
        self.assertEqual(advance(self.initial(pain=60, activity='rest'), 60, '아픈 휴식')['fatigue'], 59)
        self.assertEqual(advance(self.initial(pain=60, activity='sleep'), 60, '아픈 수면')['fatigue'], 56)
        self.assertEqual(advance(self.initial(pain=40, activity='sleep'), 60, '수면')['fatigue'], 52)
        # Healing clock: a treated recovering wound loses a severity step per 24h and disappears at 0.
        day = advance(self.initial(injuries=[burn, cut]), 1440, '하루')
        self.assertEqual([(i['id'], i['severity']) for i in day['injuries']], [('burn', 2), ('cut', 2)])
        self.assertEqual(day['last_calculation']['injury_changes'], [{'id': 'burn', 'from': 3, 'to': 2}])
        self.assertEqual(day['injuries'][1]['progress_minutes'], 0)
        state = day
        for _ in range(2):
            state = advance(state, state['scene_minute'] + 1440, '하루 더')
        self.assertEqual([i['id'] for i in state['injuries']], ['cut'])
        self.assertEqual(state['last_calculation']['healed'], ['burn'])
        # Untreated worsening gains a step per 24h and stops at 3; untreated recovery takes 48h a step.
        wound = {'id': 'w', 'description': '상처', 'severity': 1, 'trend': 'worsening', 'treated': False, 'progress_minutes': 1439}
        worse, changes = progress_injuries([wound], 1)
        self.assertEqual((worse[0]['severity'], worse[0]['progress_minutes'], changes), (2, 0, [{'id': 'w', 'from': 1, 'to': 2}]))
        capped, _ = progress_injuries([{**wound, 'severity': 3}], 5000)
        self.assertEqual((capped[0]['severity'], capped[0]['progress_minutes']), (3, 1440))
        slow, _ = progress_injuries([{**wound, 'trend': 'recovering', 'progress_minutes': 0}], 2879)
        self.assertEqual(slow[0]['severity'], 1)
        # Records saved before the clock existed carry no progress_minutes at all.
        legacy = {'id': 'old', 'description': '옛 기록', 'severity': 2, 'trend': 'recovering', 'treated': True}
        self.assertEqual(carry_injury_progress([legacy], [dict(legacy)])[0]['progress_minutes'], 0)
        from runtime_tools.roleplay_dynamics import reconcile_injuries
        self.assertEqual(reconcile_injuries([legacy], [dict(legacy)], [dict(legacy)])[0]['progress_minutes'], 0)
        # A model-sent list keeps the server clock for unchanged trends and restarts it on a change.
        carried = carry_injury_progress([{**burn, 'progress_minutes': 700}, {**cut, 'progress_minutes': 300}],
                                        [dict(burn), {**cut, 'trend': 'recovering'}, {'id': 'new', 'description': '새 상처', 'severity': 1, 'trend': 'stable', 'treated': False}])
        self.assertEqual([i['progress_minutes'] for i in carried], [700, 0, 0])

    def test_tension_eases_with_calm_resolve_sleep_and_rises_with_pain(self):
        from runtime_tools.roleplay_dynamics import tension_target
        base = self.initial(threat='uncertain', tension=40)
        self.assertEqual(tension_target(base), 40)
        self.assertEqual(tension_target({**base, 'calm_minutes': 8 * 60}), 32)
        self.assertEqual(tension_target({**base, 'calm_minutes': 40 * 60}), 30)
        self.assertEqual(tension_target({**base, 'resolve': 100}), 32)
        self.assertEqual(tension_target({**base, 'resolve': 0}), 48)
        self.assertEqual(tension_target({**base, 'pain': 60}), 45)
        self.assertEqual(tension_target({**base, 'activity': 'sleep'}), 30)
        self.assertEqual(tension_target({**base, 'threat': 'safe', 'calm_minutes': 40 * 60, 'resolve': 100, 'activity': 'sleep'}), 5)
        # A quiet day in the cell: tension settles well below the bare threat level.
        state = base
        for _ in range(12):
            state = advance(state, state['scene_minute'] + 60, '조용한 한 시간')
        self.assertEqual(state['calm_minutes'], 720)
        self.assertLess(state['tension'], 32)
        self.assertEqual(state['last_calculation']['tension_target'], 30)
        # A threatening interval ends the calm streak.
        shaken = advance({**state, 'threat': 'threatening'}, state['scene_minute'] + 30, '방문자')
        self.assertEqual(shaken['calm_minutes'], 0)
        self.assertGreater(shaken['tension'], state['tension'])

    def test_isolation_stages_and_contact(self):
        from runtime_tools.roleplay_dynamics import isolation_stage, isolation_after, tension_target, mental_rates
        self.assertIsNone(isolation_stage(23 * 60))
        self.assertEqual([isolation_stage(h * 60)['label'] for h in (24, 72, 168)], ['단절', '침식', '왜곡'])
        alone = self.initial(threat='uncertain', resolve=60, clarity=60, humiliation=40)
        self.assertEqual(isolation_after(alone, 600), 600)
        self.assertEqual(isolation_after({**alone, 'isolation_minutes': 3000, 'participants': ['guard']}, 10), 2760)
        self.assertEqual(isolation_after({**alone, 'isolation_minutes': 3000, 'participants': ['guard']}, 30), 0)
        self.assertEqual(isolation_after({**alone, 'isolation_minutes': 100, 'participants': ['guard']}, 5), 0)
        self.assertEqual(mental_rates(alone)['clarity'], 0.75)  # quiet rest sharpens the mind, less so near the ceiling
        self.assertEqual(mental_rates({**alone, 'isolation_minutes': 30 * 60})['clarity'], -0.25)  # not in solitary
        day3 = mental_rates({**alone, 'isolation_minutes': 80 * 60})
        self.assertEqual((day3['clarity'], day3['resolve']), (-0.5, 0.5 * 0.75 - 0.25))  # gain thinned, drain in full
        self.assertLess(mental_rates({**alone, 'isolation_minutes': 80 * 60, 'resolve': 88})['resolve'], 0)
        self.assertEqual(tension_target({**alone, 'isolation_minutes': 30 * 60}) - tension_target(alone), 5)
        # Two quiet days alone: after the first day clarity erodes despite rest.
        state = alone
        for _ in range(48):
            state = advance(state, state['scene_minute'] + 60, '홀로 한 시간')
        self.assertEqual(state['isolation_minutes'], 48 * 60)
        self.assertEqual(state['last_calculation']['isolation_stage'], '단절')
        day_one = alone
        for _ in range(24):
            day_one = advance(day_one, day_one['scene_minute'] + 60, '홀로 한 시간')
        self.assertLess(day_one['clarity'], 90)
        self.assertAlmostEqual(state['clarity'], day_one['clarity'] - 6, places=3)
        self.assertGreater(state['tension'], advance({**state, 'isolation_minutes': 0}, state['scene_minute'] + 60, '비교')['tension'] - 6)
        visited = advance({**state, 'participants': ['guard']}, state['scene_minute'] + 45, '간수와 45분')
        self.assertEqual(visited['isolation_minutes'], 0)

    def test_mental_axes_drift(self):
        mental = dict(resolve=50, clarity=50, humiliation=50)
        safe_rest = advance(self.initial(threat='safe', **mental), 60, '안전한 휴식 한 시간')
        self.assertEqual((safe_rest['resolve'], safe_rest['clarity'], safe_rest['humiliation']), (52, 51, 49.5))
        coerced = advance(self.initial(threat='immediate', activity='strenuous', fatigue=75, pain=65, **mental), 60, '강요 한 시간')
        self.assertEqual((coerced['resolve'], coerced['clarity'], coerced['humiliation']), (45, 47, 50))
        # Hourly steps thin the second hour's gain a little (see _diminish), so two hours fall just short of 2x.
        slept = advance(self.initial(activity='sleep', sleep_quality='good', threat='uncertain', **mental), 120, '두 시간 숙면')
        self.assertAlmostEqual(slept['resolve'], 53, delta=0.1)
        self.assertAlmostEqual(slept['clarity'], 60, delta=0.7)
        poor = advance(self.initial(activity='sleep', sleep_quality='poor', threat='uncertain', **mental), 120, '두 시간 선잠')
        self.assertAlmostEqual(poor['resolve'], 51, delta=0.1)
        self.assertAlmostEqual(poor['clarity'], 52, delta=0.1)
        # Unset mental values stay unset; physical drift is unaffected by them.
        unset = advance(self.initial(threat='immediate'), 60, '정신 수치 미설정')
        self.assertIsNone(unset['resolve'])
        self.assertEqual(unset['hunger'], 33)
        bounded = advance(self.initial(resolve=1, clarity=99, activity='sleep', sleep_quality='good', threat='immediate', participants=['x']), 120, '경계값')
        self.assertEqual(bounded['resolve'], 0)
        self.assertEqual(bounded['clarity'], 99)  # drift never lifts a mind past the ceiling
        # Recovery thins out near the ceiling: a resolute mind gains little from a quiet hour.
        self.assertEqual(advance(self.initial(threat='safe', resolve=90), 60, '한 시간')['resolve'], 90)
        self.assertEqual(advance(self.initial(threat='safe', resolve=70), 60, '한 시간')['resolve'], 71)

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

    def test_injury_clock_survives_model_resend(self):
        burn = {'id': 'burn', 'description': '화상', 'severity': 2, 'trend': 'recovering', 'treated': True}
        memory.roleplay_state('update', {'injuries': [burn]}, '처치', expected_revision=1)
        temporal = {'relation': 'current', 'certainty': 'explicit', 'operation': 'advance', 'elapsed_minutes': 720,
                    'source_quote': '반나절', 'interpretation': '반나절 휴식'}
        memory.roleplay_state('time', reason='반나절', expected_revision=2, event_id='half', temporal=temporal,
                              interval_conditions={'activity': 'rest'}, person_updates=[], person_review='없음')
        self.assertEqual(memory.load_state(1)['injuries'][0]['progress_minutes'], 720)
        # The model resends the list without the server clock; the clock is kept.
        result = json.loads(memory.roleplay_state('time', {'injuries': [{**burn, 'description': '화상, 딱지 앉음'}]}, '반나절 더',
                            expected_revision=3, event_id='half2', temporal=temporal,
                            interval_conditions={'activity': 'rest', 'injuries': [burn]}, person_updates=[], person_review='없음'))
        state = memory.load_state(1)
        self.assertEqual(state['injuries'][0]['severity'], 1)
        self.assertEqual(state['injuries'][0]['description'], '화상, 딱지 앉음')
        self.assertEqual(state['injuries'][0]['progress_minutes'], 0)
        self.assertEqual(result['last_calculation']['injury_changes'], [{'id': 'burn', 'from': 2, 'to': 1}])
        self.assertEqual(result['pain_floor'], 3)

    def test_alone_interval_warning_and_inner_metric_reasons(self):
        temporal = {'relation': 'current', 'certainty': 'estimated', 'operation': 'advance', 'elapsed_minutes': 448,
                    'source_quote': '밤을 넘겼다', 'interpretation': '홀로 보낸 밤'}
        result = json.loads(memory.roleplay_state('time', reason='밤', expected_revision=1, event_id='night', temporal=temporal,
                            interval_conditions={'activity': 'sleep', 'sleep_quality': 'poor', 'threat': 'threatening'},
                            person_updates=[], person_review='없음'))
        self.assertEqual(memory.load_state(1)['threat'], 'uncertain')
        self.assertTrue(any('uncertain으로 계산' in w for w in result['warnings']))
        result = json.loads(memory.roleplay_state('update', {'pain': 30, 'metric_reasons': {'pain': '뺨을 맞음'}}, '구타',
                                                  expected_revision=2, adjustment='event', event_id='slap'))
        self.assertEqual(memory.load_state(1)['pain'], 30)
        self.assertTrue(any('metric_reasons' in w for w in result['warnings']))
        self.assertEqual(json.loads(memory.roleplay_state('history'))[0]['metric_reasons'], {'pain': '뺨을 맞음'})

    def test_tension_event_resets_calm(self):
        temporal = {'relation': 'current', 'certainty': 'explicit', 'operation': 'advance', 'elapsed_minutes': 600,
                    'source_quote': '열 시간', 'interpretation': '조용한 열 시간'}
        memory.roleplay_state('time', reason='조용', expected_revision=1, event_id='quiet', temporal=temporal,
                              interval_conditions={'activity': 'rest', 'threat': 'uncertain'}, person_updates=[], person_review='없음')
        self.assertEqual(memory.load_state(1)['calm_minutes'], 600)
        memory.roleplay_state('update', {'tension': 80}, '문이 열림', expected_revision=2, adjustment='event', event_id='door')
        self.assertEqual(memory.load_state(1)['calm_minutes'], 0)
        self.assertEqual(json.loads(memory.roleplay_state('read'))['calm_hours'], 0)

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
