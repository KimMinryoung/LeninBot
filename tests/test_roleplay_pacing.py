import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_tools import roleplay_memory as memory
from runtime_tools.roleplay_pacing import policy_for, turn_time_scope
from tool_gateway.security import caller_scope, new_run_context


class PacingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(memory, 'MEMORY_PATH', Path(tmp.name) / 'memory.sqlite3')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scope = caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1',
                                                  is_owner=True, scope_type='offline_pacing_test', scope_id='1467'))
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        memory.roleplay_state('update', changes={'hunger': 10, 'fatigue': 50, 'pain': 0, 'tension': 30,
                              'activity': 'light', 'threat': 'uncertain', 'sleep_quality': 'normal', 'injuries': []},
                              reason='초기 상태', expected_revision=0, adjustment='initialize')
        memory.roleplay_state('time', temporal=self.temporal('anchor', date='1939-04-27', time='19:40'),
                              reason='초기 시각', expected_revision=1)

    def temporal(self, op='advance', quote='위로 올려보내', **fields):
        return dict(operation=op, relation='current', certainty='estimated', source_quote=quote,
                    interpretation='현재 장면의 이동', **fields)

    def time(self, minutes, quote='위로 올려보내', **kwargs):
        return json.loads(memory.roleplay_state('time', temporal=self.temporal(quote=quote, elapsed_minutes=minutes),
                          interval_conditions={'activity': 'light'}, reason='장면 진행',
                          expected_revision=memory.load_state(1)['revision'], **kwargs))

    def test_original_incident_and_fabricated_sources_are_atomic(self):
        text = '(위로 올려보내. 이제 상태가 좀 나아져야 다시 끌어내리는 재미가 있을텐데, 어떻게 할까?)'
        before = memory.load_state(1)
        with turn_time_scope(policy_for(text)):
            for minutes, quote in [(100, '위로 올려보내.'), (640, '(아무도 오지 않은 밤)'),
                                   (780, '(하루 종일 잤다 깼다를 되풀이했다)'), (630, '(밤에도 아무도 오지 않았다)')]:
                with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                    self.time(minutes, quote=quote, changes={'hunger': 0, 'last_event': '다음 날'})
                self.assertEqual(memory.load_state(1), before)

    def test_arrival_is_the_turn_endpoint(self):
        with turn_time_scope(policy_for('위로 올려보내')):
            self.time(3, event_id='arrival', changes={'last_event': '감방에 돌아왔다'})
            arrived = memory.load_state(1)
            # Even under the total minute cap, another elapsed event is forbidden.
            with self.assertRaisesRegex(ValueError, '추가 사건'):
                self.time(2, event_id='wash')
            self.assertEqual(memory.load_state(1), arrived)
            replay = self.time(3, event_id='arrival')
            self.assertTrue(replay['replayed'])
        self.assertEqual(arrived['clock']['time'], '19:43')

    def test_explicit_passage_split_budget_and_new_message(self):
        text = '한 시간 쉬어'
        self.assertEqual(policy_for(text).max_minutes, 60)
        with turn_time_scope(policy_for(text)):
            self.time(30, quote=text, event_id='first')
            self.time(30, quote=text, event_id='second')
            with self.assertRaises(ValueError):
                self.time(1, quote=text, event_id='excess')
        with caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id='1',
                                         is_owner=True, scope_type='offline_pacing_test', scope_id='1468')):
            with turn_time_scope(policy_for('한 시간 쉬어')):
                self.time(60, quote=text, event_id='new-turn')
        self.assertEqual(memory.load_state(1)['scene_minute'], 120)

    def test_discussion_and_clock_bypasses(self):
        for text in ['하루 쉬게 하면 어떨까?', '두 시간 지나면 나을까?', '어제 한 시간 쉬었다. 어떻게 할까?', '이야기를 리드해봐']:
            self.assertEqual(policy_for(text).max_minutes, 10)
        for op, fields in [('next_day', {}), ('correct', {'date': '1939-04-30'}),
                           ('until', {'date': '1939-04-29', 'time': '07:30'})]:
            with turn_time_scope(policy_for('위로 올려보내')), self.assertRaises(ValueError):
                memory.roleplay_state('time', temporal=self.temporal(op, **fields), reason='시도',
                                      interval_conditions={'activity': 'rest'} if op == 'until' else None,
                                      expected_revision=memory.load_state(1)['revision'])
        self.assertEqual(memory.load_state(1)['clock']['date'], '1939-04-27')

    def test_explicit_next_day_and_past_reference(self):
        text = '다음 날 아침으로 넘겨'
        with turn_time_scope(policy_for(text)):
            memory.roleplay_state('time', temporal=self.temporal('next_day', quote=text), reason='명시적 진행',
                                  event_id='tomorrow', expected_revision=memory.load_state(1)['revision'])
            with self.assertRaises(ValueError):
                memory.roleplay_state('time', temporal=self.temporal('next_day', quote=text), reason='또 하루',
                                      event_id='another', expected_revision=memory.load_state(1)['revision'])
        self.assertEqual(memory.load_state(1)['clock']['date'], '1939-04-28')
        with turn_time_scope(policy_for('어제는 하루 쉬었다')):
            memory.roleplay_state('time', temporal={**self.temporal(elapsed_minutes=1440), 'relation': 'past'},
                                  reason='회상', expected_revision=memory.load_state(1)['revision'])
        self.assertEqual(memory.load_state(1)['scene_minute'], 0)

    def test_reset_cannot_bypass_and_retracted_audit_is_not_context(self):
        with turn_time_scope(policy_for('위로 올려보내')), self.assertRaises(ValueError):
            memory.roleplay_state('reset', reason='시간 제한 우회', expected_revision=memory.load_state(1)['revision'])
        self.time(3, event_id='old-passage')
        with memory._connection() as conn:
            conn.execute('INSERT INTO turn_retractions VALUES (?, ?, ?)', ('1', '1467', '사용자 철회'))
        self.assertEqual(json.loads(memory.roleplay_state('history')), [])
        with turn_time_scope(policy_for('새 장면으로 초기화해')):
            memory.roleplay_state('reset', reason='새 장면', expected_revision=memory.load_state(1)['revision'])
        self.assertEqual(memory.load_state(1)['scene_minute'], 0)

    def test_excluded_history_is_scoped(self):
        with memory._connection() as conn:
            conn.execute('INSERT INTO history_exclusions VALUES (?, ?, ?)', ('1', 976, '사용자가 철회한 턴'))
        self.assertEqual(memory.excluded_history_ids(1), [976])
        self.assertEqual(memory.excluded_history_ids(2), [])
        from telegram import roleplay_bot as bot
        with patch.object(bot, '_clear_after_id', return_value=0), \
             patch.object(bot, '_query', side_effect=[[{'n': 1}], [{'role': 'user', 'content': '유효한 대화'}]]) as query:
            self.assertEqual(bot.load_history(1), [{'role': 'user', 'content': '유효한 대화'}])
        self.assertIn('id = ANY(%s)', query.call_args.args[0])
        self.assertEqual(query.call_args.args[1][-2], [976])


class OpenEndedRestTests(unittest.TestCase):
    def test_rest_left_to_the_character_allows_a_session_budget(self):
        from runtime_tools.roleplay_pacing import policy_for
        free = policy_for('(예조프야 맘대로 쉬어라)')
        self.assertEqual((free.max_minutes, free.explicit_passage, free.calendar_skip), (180, False, False))
        plain = policy_for('감방에서 쉬어')
        self.assertEqual((plain.max_minutes, plain.explicit_passage), (10, False))
        timed = policy_for('한 시간 푹 쉬어')
        self.assertEqual((timed.max_minutes, timed.explicit_passage), (60, True))
