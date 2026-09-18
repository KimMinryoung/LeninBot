import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_tools import roleplay_memory as memory
from tool_gateway.security import caller_scope, new_run_context
from telegram.roleplay_bot import repeated_phrases


class RoleplayMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path_patch = patch.object(memory, 'MEMORY_PATH', Path(self.temp.name) / 'notes.sqlite3')
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def owner(self, user='1'):
        return caller_scope(new_run_context(interface='telegram', agent_name='roleplay', user_id=user, is_owner=True))

    def state(self, action, changes=None, reason=""):
        from tool_gateway.security import get_caller
        from uuid import uuid4
        current = memory.load_state(get_caller().user_id or "1")
        numeric = set(changes or {}) & {"hunger", "fatigue", "pain", "tension"}
        return memory.roleplay_state(action, changes, reason, expected_revision=current["revision"],
            adjustment="correction" if any(current[k] is not None for k in numeric) else "initialize",
            event_id=str(uuid4()), metric_reasons={k: reason for k in numeric},
            person_updates=[], person_review="인물의 새 정보 없음")

    def test_persistence_update_delete_and_isolation(self):
        with self.owner():
            memory.roleplay_memory('save', '관계', '친구')
            memory.roleplay_memory('save', '관계', '오랜 친구')
        self.assertEqual(memory.load_notes(1), [{'key': '관계', 'content': '오랜 친구'}])
        with self.owner('2'):
            self.assertEqual(json.loads(memory.roleplay_memory('list')), [])
            memory.roleplay_memory('delete', '관계')
        self.assertEqual(len(memory.load_notes(1)), 1)
        with self.owner():
            memory.roleplay_memory('delete', '관계')
        self.assertEqual(memory.load_notes(1), [])

    def test_context_and_limits(self):
        with self.assertRaises(PermissionError):
            memory.roleplay_memory('list')
        with self.assertRaises(PermissionError):
            self.state('read')
        with self.owner():
            for i in range(30):
                memory.roleplay_memory('save', str(i), '내용')
            with self.assertRaises(ValueError):
                memory.roleplay_memory('save', 'overflow', '내용')
            memory.roleplay_memory('save', '0', '수정')
            with self.assertRaises(ValueError):
                memory.roleplay_memory('save', '0', 'x' * 801)

    def test_state_partial_update_validation_and_reset(self):
        self.assertIsNone(memory.load_state(1)['hunger'])
        with self.owner():
            self.state('update', {'hunger': 70, 'mood': '초조함'}, '끼니를 거름')
            self.state('update', {'hunger': 10}, '식사를 마침')
            self.assertEqual(memory.load_state(1)['mood'], '초조함')
            self.assertEqual(memory.load_state(1)['hunger'], 10)
            for changes in ({'hunger': 101}, {'tension': True}, {'unknown': 1}):
                with self.assertRaises(ValueError):
                    self.state('update', changes, '이유')
            self.assertEqual(memory.load_state(1)['hunger'], 10)
            self.assertIsNone(memory.load_state(2)['hunger'])
            self.state('reset', reason='새 장면')
        self.assertIsNone(memory.load_state(1)['hunger'])

    def test_repetition_uses_assistant_turns(self):
        phrase = '나는 천천히 고개를 돌려 창밖을 바라보았다'
        self.assertEqual(repeated_phrases([{'role': 'assistant', 'content': phrase * 4}]), [])
        history = [{'role': 'assistant', 'content': phrase} for _ in range(3)]
        self.assertIn(phrase, repeated_phrases(history))
        self.assertEqual(repeated_phrases([{'role': 'user', 'content': phrase}] * 3), [])

    def test_people_alias_ambiguity_and_user_isolation(self):
        with self.owner():
            memory.roleplay_person('save', 'ivan-a', changes={'name': '이반', 'aliases': ['Vanya'], 'identity': '직장 동료'})
            memory.roleplay_person('save', 'ivan-b', changes={'name': '이반', 'aliases': ['Vanya'], 'identity': '이웃'})
            result = json.loads(memory.roleplay_person('read', query='  VANYA  '))
            self.assertTrue(result['ambiguous'])
            self.assertEqual(len(result['matches']), 2)
            memory.roleplay_person('save', 'ivan-a', changes={'relationship': '경계하는 동료'})
            result = json.loads(memory.roleplay_person('read', 'ivan-a'))['matches'][0]
            self.assertEqual(result['identity'], '직장 동료')
            self.assertEqual(result['aliases'], ['Vanya'])
        with self.owner('2'):
            self.assertEqual(json.loads(memory.roleplay_person('read', 'ivan-a'))['matches'], [])
            memory.roleplay_person('delete', 'ivan-a')
        self.assertEqual(len(memory.load_people(1)), 2)
        with self.assertRaises(PermissionError):
            memory.roleplay_person('list')

    def test_scene_participants_goals_context_and_deletion(self):
        with self.owner():
            memory.roleplay_person('save', 'ivan', changes={'name': '이반', 'observed': '어제 함께 식사함'})
            memory.roleplay_person('save', 'anna', changes={'name': '안나', 'reported': '이반이 안나의 부재를 알림'})
            self.state('update', {'participants': ['ivan'], 'goal': '함께 쉬기', 'unresolved': '저녁 약속'}, '이반을 만남')
            with self.assertRaises(ValueError):
                self.state('update', {'participants': ['missing'], 'goal': '잘못된 갱신'}, '장면 전환')
            state = memory.load_state(1)
            self.assertEqual(state['goal'], '함께 쉬기')
            context = memory.people_context(1, state['participants'])
            self.assertEqual(len(context['index']), 2)
            self.assertNotIn('observed', context['index'][0])
            self.assertEqual([p['person_id'] for p in context['present']], ['ivan'])
            self.state('update', {'unresolved': ''}, '약속을 정함')
            self.assertEqual(memory.load_state(1)['unresolved'], '')
            memory.roleplay_person('delete', 'ivan')
            self.assertEqual(memory.load_state(1)['participants'], [])
            self.state('reset', reason='별개의 장면')
            self.assertEqual(memory.load_state(1)['goal'], '')
            self.assertEqual(len(memory.load_people(1)), 1)

    def test_legacy_state_and_person_limits(self):
        with memory._connection() as conn:
            conn.execute('INSERT INTO character_state VALUES (?, ?)', ('1', '{"hunger": 40}'))
        self.assertEqual(memory.load_state(1)['hunger'], 40)
        self.assertEqual(memory.load_state(1)['participants'], [])
        with self.owner():
            for changes in ({'aliases': 'alias'}, {'unknown': 'value'}, {'name': ''}):
                with self.assertRaises(ValueError):
                    memory.roleplay_person('save', 'ivan', changes=changes)
            for i in range(30):
                memory.roleplay_person('save', f'p{i}', changes={'name': str(i)})
            with self.assertRaises(ValueError):
                memory.roleplay_person('save', 'overflow', changes={'name': '초과'})
            memory.roleplay_person('save', 'p0', changes={'name': '수정'})

    def test_person_id_normalization_and_review_shapes(self):
        with self.owner():
            result = json.loads(memory.roleplay_person('save', 'Young Guard', changes={'name': '젊은 간수', 'reason': '등장'}))
            self.assertEqual(result['person_id'], 'young_guard')
            self.assertTrue(result['warnings'])
            self.assertEqual(json.loads(memory.roleplay_person('read', 'Young Guard'))['matches'][0]['name'], '젊은 간수')
            with self.assertRaises(ValueError):
                memory.roleplay_person('save', '간수', changes={'name': '간수'})
            self.state('update', {'participants': ['젊은 간수']}, '간수 입장')
            self.assertEqual(memory.load_state(1)['participants'], ['young_guard'])
            with self.assertRaises(ValueError) as caught:
                self.state('update', {'participants': ['nobody']}, '미등록')
            self.assertIn('young_guard', str(caught.exception))
            current = memory.load_state(1)['revision']
            result = json.loads(memory.roleplay_state('update', {'last_event': '간수가 물을 줌'}, '물', expected_revision=current,
                person_updates=[{'person_id': 'young_guard', 'observed': '물을 줌'}, '다른 인물의 변화 없음'],
                person_review_note='검토함'))
            self.assertEqual(memory.load_people(1)[0]['observed'], '물을 줌')
            self.assertNotIn('people_reminder', result)
            self.assertIn('warnings', result)
            with self.assertRaises(ValueError):
                memory.roleplay_state('update', {'last_event': 'x'}, '잘못된 필드', expected_revision=current + 1,
                                      person_updates=[{'person_id': 'young_guard', 'changes': {'name': '이름 변경'}}])

    def test_memory_similar_key_hint(self):
        with self.owner():
            self.assertNotIn('similar_keys', json.loads(memory.roleplay_memory('save', '장면-구금방', '첫 메모')))
            result = json.loads(memory.roleplay_memory('save', '장면-구금방(1939-04-21)', '둘째 메모'))
            self.assertEqual(result['similar_keys'], ['장면-구금방'])
            self.assertNotIn('similar_keys', json.loads(memory.roleplay_memory('save', '화법', '무관')))

    def test_state_view_is_compact(self):
        with self.owner():
            self.state('update', {'hunger': 33.333, 'activity': 'rest', 'sleep_quality': 'normal', 'threat': 'safe', 'injuries': []}, '기준')
            view = json.loads(memory.roleplay_state('read'))
        self.assertEqual(view['hunger'], 33.3)
        self.assertNotIn('recent_events', view)
        self.assertNotIn('event_timestamps', view)
        self.assertNotIn('last_scope_id', view)
        self.assertNotIn('period', view)
        self.assertEqual(view['clock']['daypart'], 'unknown')
        self.assertEqual(view['unset_metrics'], ['fatigue', 'pain', 'tension', 'resolve', 'clarity', 'humiliation'])
        self.assertEqual(memory.load_state(1)['hunger'], 33.333)

    def test_dictionary_link_validation_and_removal(self):
        with self.owner():
            memory.roleplay_person('save', 'rodos', changes={'name': '로도스', 'observed': '역할극 사건'})
            with patch('db.query_one', return_value={'id': 'boris-rodos'}) as query:
                result = json.loads(memory.roleplay_person('save', 'rodos', changes={'commulingo_id': 'boris-rodos'}))
                self.assertEqual(query.call_args.args[1], ('boris-rodos',))
            self.assertEqual(result['commulingo_url'], 'https://cyber-lenin.com/commulingo/people/boris-rodos')
            self.assertEqual(memory.people_context(1, [])['index'][0]['commulingo_id'], 'boris-rodos')
            with patch('db.query_one', return_value=None):
                with self.assertRaises(ValueError):
                    memory.roleplay_person('save', 'rodos', changes={'commulingo_id': 'missing', 'observed': '덮어쓰기'})
            saved = memory.load_people(1)[0]
            self.assertEqual(saved['observed'], '역할극 사건')
            self.assertEqual(saved['commulingo_id'], 'boris-rodos')
            memory.roleplay_person('save', 'rodos', changes={'commulingo_id': ''})
            self.assertEqual(memory.load_people(1)[0]['commulingo_url'], '')
