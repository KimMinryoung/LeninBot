"""Behavioral regressions for contact, time partitioning, and story interruptions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_tools import roleplay_memory as memory
from runtime_tools.roleplay_dynamics import METRICS, advance, with_defaults
from runtime_tools.roleplay_story import apply_story_updates
from tool_gateway.security import caller_scope, new_run_context


class ProgressionTests(unittest.TestCase):
    def initial(self, **changes):
        return with_defaults(dict(hunger=10, fatigue=50, pain=68, tension=70,
                                  resolve=60, clarity=60, humiliation=10,
                                  conditions_initialized=True, threat='threatening',
                                  isolation_mode='solitary', participants=[], **changes))

    def test_partition_invariance_across_thresholds(self):
        initial = self.initial()
        for key, value in [('alone_rest_minutes', 50), ('isolation_minutes', 23 * 60 + 40),
                           ('wakefulness_minutes', 15 * 60 + 30)]:
            initial[key] = value
        whole = advance(initial, 180, '세 시간')
        split = initial
        for target in (13, 30, 55, 60, 97, 180):
            split = advance(split, target, '나누어 진행')
        for key in (*METRICS, 'alone_rest_minutes', 'isolation_minutes', 'wakefulness_minutes', 'threat'):
            self.assertEqual(whole[key], split[key], key)
        half = advance(self.initial(), 30, '30분')
        hour = advance(half, 60, '30분 더')
        self.assertEqual(hour['threat'], 'uncertain')
        self.assertEqual(hour['resolve'], advance(self.initial(), 60, '한 시간')['resolve'])
        continued = advance({**hour, 'threat': 'threatening'}, 90, '같은 위협 조건을 되풀이')
        self.assertEqual(continued['resolve'], advance(self.initial(), 90, '90분')['resolve'])

    def test_meals_and_interrogation_do_not_erase_isolation(self):
        state = self.initial()
        for day in range(3):
            # 23 hours alone, three brief meals, and one 45-minute interrogation.
            for duration, people, contact in [(1380, [], 'none'), (5, ['guard'], 'incidental'),
                                               (5, ['guard'], 'incidental'), (5, ['guard'], 'incidental'),
                                               (45, ['guard'], 'hostile')]:
                state = advance({**state, 'participants': people, 'social_contact': contact},
                                state['scene_minute'] + duration, '독방 생활')
        self.assertEqual(state['isolation_minutes'], 3 * 1440)
        self.assertEqual(state['last_calculation']['isolation_stage'], '침식')
        contact = {**state, 'participants': ['friend'], 'social_contact': 'meaningful'}
        whole = advance(contact, state['scene_minute'] + 60, '지지적인 대화')
        split = advance(contact, state['scene_minute'] + 10, '대화')
        split = advance(split, state['scene_minute'] + 60, '대화 계속')
        self.assertEqual(whole['isolation_minutes'], 3 * 1440 - 120)
        self.assertEqual(split['isolation_minutes'], whole['isolation_minutes'])
        departed = advance({**whole, 'participants': []}, whole['scene_minute'] + 60, '다시 혼자')
        self.assertEqual(departed['isolation_minutes'], whole['isolation_minutes'] + 60)

    def test_ordinary_solitude_and_sleep(self):
        state = {**self.initial(), 'isolation_mode': 'ordinary', 'isolation_minutes': 300,
                 'fatigue': 0, 'pain': 0, 'threat': 'safe'}
        day = advance(state, 1440, '잠들지 않고 집에서 쉼')
        self.assertEqual(day['isolation_minutes'], 0)
        self.assertGreater(day['fatigue'], 10)
        slept = advance({**day, 'activity': 'sleep'}, 1920, '여덟 시간 잠')
        self.assertLess(slept['wakefulness_minutes'], day['wakefulness_minutes'])
        self.assertLess(slept['fatigue'], day['fatigue'])


class StoryTransactionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patcher = patch.object(memory, 'MEMORY_PATH', Path(temp.name) / 'memory.sqlite3')
        patcher.start()
        self.addCleanup(patcher.stop)
        scope = caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1', is_owner=True))
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        self.call('update', changes=dict(hunger=30, fatigue=60, pain=0, tension=40, resolve=60,
                  activity='rest', sleep_quality='normal', threat='safe', injuries=[]), adjustment='initialize')
        self.call('time', temporal=self.temporal('anchor', date='1939-09-20', time='10:00'))

    def temporal(self, operation='advance', **fields):
        return dict(relation='current', certainty='explicit', source_quote='장면 시간',
                    interpretation='장면의 시간 근거', operation=operation, **fields)

    def call(self, action, **kwargs):
        return json.loads(memory.roleplay_state(action, reason='장면 근거',
                          expected_revision=memory.load_state(1)['revision'], **kwargs))

    def schedule(self, eid, **fields):
        return self.call('update', story_updates=[dict(op='schedule', id=eid, title=eid,
                                                       source='사용자와 합의한 약속', **fields)])

    def advance(self, minutes, **kwargs):
        return self.call('time', temporal=self.temporal(elapsed_minutes=minutes),
                         interval_conditions={'activity': 'rest'}, **kwargs)

    def test_interrupt_clock_deferred_effects_and_resume(self):
        self.schedule('visit', due_minute=60)
        self.schedule('reply', due_minute=120, after_event='visit')
        result = self.advance(180, changes={'hunger': 0, 'last_event': '저녁을 먹음'},
                              resolve_event={'kind': 'kindness', 'intensity': 2},
                              person_updates=[{'person_id': 'not-present', 'changes': {'observed': '아직 일어나지 않음'}}],
                              event_id='wait')
        self.assertEqual(result['scene_minute'], 60)
        self.assertEqual(result['clock']['time'], '11:00')
        self.assertEqual(result['hunger'], 33)
        self.assertEqual(result['last_event'], '')
        self.assertNotIn('last_resolve_event', result)
        self.assertEqual(result['story_interrupt']['remaining_minutes'], 120)
        self.assertEqual(result['story_events'][0]['status'], 'ready')
        again = self.advance(180, event_id='wait')
        self.assertTrue(again['replayed'])
        self.assertEqual(again['scene_minute'], 60)
        blocked = self.advance(120)
        self.assertEqual(blocked['scene_minute'], 60)
        self.call('update', story_updates=[dict(op='complete', id='visit', outcome='방문해 소식을 전함')],
                  changes={'last_event': '방문해 소식을 전함'})
        result = self.advance(120, event_id='after-visit')
        self.assertEqual(result['scene_minute'], 120)
        self.assertEqual(result['clock']['time'], '12:00')
        self.call('update', story_updates=[dict(op='cancel', id='reply', outcome='답신 약속 취소')])
        self.assertEqual(self.advance(60)['scene_minute'], 180)

    def test_until_same_time_and_multiple_events(self):
        self.schedule('a', due_minute=20)
        self.schedule('b', due_minute=20)
        result = self.call('time', temporal=self.temporal('until', date='1939-09-20', time='12:00'),
                           interval_conditions={'activity': 'rest'})
        self.assertEqual(result['clock']['time'], '10:20')
        self.assertEqual(result['story_interrupt']['event_ids'], ['a', 'b'])
        self.call('update', story_updates=[dict(op='complete', id='a', outcome='만남')])
        self.assertEqual(self.advance(60)['scene_minute'], 20)

    def test_dependency_cancellation_atomicity_and_idempotency(self):
        self.schedule('a', due_minute=60)
        self.schedule('b', after_event='a')
        before = memory.load_state(1)
        with self.assertRaises(ValueError):
            self.call('update', story_updates=[dict(op='cancel', id='a', outcome='취소')],
                      person_updates=[dict(person_id='missing', changes={'observed': '실패'})])
        self.assertEqual(memory.load_state(1), before)
        self.call('update', story_updates=[dict(op='cancel', id='a', outcome='취소')])
        self.assertEqual(self.call('read')['story_events'], [])
        self.schedule('a', due_minute=60)  # does not revive cancelled event
        self.assertEqual(self.call('read')['story_events'], [])
        with self.assertRaises(ValueError):
            self.schedule('a', due_minute=70)
        self.assertEqual(memory.load_state(2)['story_events'], [])
        self.call('reset')
        self.assertEqual(memory.load_state(1)['story_events'], [])

    def test_invalid_inputs_and_unquantified_skip(self):
        for fields in ({}, {'due_minute': -1}, {'due_minute': True}, {'after_event': 'unknown'}, {'after_event': []}):
            with self.assertRaises(ValueError):
                self.schedule('bad', **fields)
        self.schedule('future', due_minute=60)
        with self.assertRaises(ValueError):
            self.call('update', story_updates=[dict(op='complete', id='future', outcome='미래 완료 주장')])
        with self.assertRaises(ValueError):
            self.call('time', temporal=self.temporal('next_day'))
        for relation in ('past', 'plan'):
            temporal = {**self.temporal(elapsed_minutes=180), 'relation': relation}
            self.call('time', temporal=temporal)
            self.assertEqual(memory.load_state(1)['scene_minute'], 0)
        with self.assertRaises(ValueError):
            self.call('time', temporal=self.temporal(elapsed_minutes=60),
                      story_updates=[], interval_conditions={'activity': 'rest'})

    def test_active_event_bound_and_contact_reset(self):
        state = with_defaults({})
        for i in range(20):
            state = apply_story_updates(state, [dict(op='schedule', id=str(i), title='방문', source='약속', due_minute=60)])
        with self.assertRaises(ValueError):
            apply_story_updates(state, [dict(op='schedule', id='overflow', title='방문', source='약속', due_minute=60)])
        memory.roleplay_person('save', 'friend', changes={'name': '친구'})
        self.call('update', changes={'participants': ['friend'], 'social_contact': 'meaningful'})
        self.call('update', changes={'participants': []})
        self.assertEqual(memory.load_state(1)['social_contact'], 'none')
        self.assertEqual(memory.load_state(1)['alone_rest_minutes'], 0)
