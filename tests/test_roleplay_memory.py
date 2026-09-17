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
            event_id=str(uuid4()), metric_reasons={k: reason for k in numeric})

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
