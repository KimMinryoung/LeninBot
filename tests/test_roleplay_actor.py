import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_tools import roleplay_memory as memory
from runtime_tools.roleplay_actor import actor_state_view, actor_outcome_view
from runtime_tools.roleplay_dynamics import with_defaults
from tool_gateway.security import caller_scope, new_run_context


class ActorContextTests(unittest.TestCase):
    def setUp(self):
        self.state = with_defaults({**memory.STATE_DEFAULTS, 'revision': 7, 'hunger': 61, 'pain': 80,
            'resolve': 20, 'clarity': 40, 'last_calculation': {'before': {'resolve': 80}, 'tension_target': 75},
            'resolve_events': [{'kind': 'kindness', 'delta': 3, 'factors': {'base': 3}}],
            'injuries': [{'id':'arm','description':'팔 부상','severity':2,'trend':'recovering','treated':True,'progress_minutes':999}],
            'story_events': [{'id':'visit','title':'방문 약속','status':'pending','due_minute':1000}],
        })

    def assert_narrative_only(self, value):
        forbidden = {'hunger','fatigue','pain','resolve','clarity','tension','humiliation','last_calculation',
                     'last_resolve_event','factors','delta','before','after','scene_minute','due_minute',
                     'progress_minutes','pain_floor','decision','uncertain','duration_estimate','max_minutes'}
        def check(item):
            if isinstance(item, dict):
                self.assertFalse(set(item) & forbidden)
                for key, child in item.items():
                    if key != 'revision':
                        self.assertNotIsInstance(child, (int, float))
                    check(child)
            elif isinstance(item, list):
                for child in item: check(child)
        check(value)

    def test_state_and_outcome_are_allowlisted(self):
        result = actor_state_view(self.state)
        self.assert_narrative_only(result)
        self.assertIn('부인할 여지', result['acting_cues']['의지'])
        self.assertIn('심한 통증', result['acting_cues']['통증'])
        self.assertEqual(result['injuries'][0]['severity'], '뚜렷함')
        outcome = actor_outcome_view({'status':'applied','model':'secret-model','duration_estimate':{'elapsed_minutes':3},
            'applied':{'event':'recognition','minutes':3,'deferred_components':['time']}})
        self.assert_narrative_only(outcome)
        self.assertIn('확정되지 않았다', outcome['direction'])
        self.assertNotIn('model', outcome)

    def test_all_actor_state_tool_responses_hide_calculation_details(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory, 'MEMORY_PATH', Path(tmp)/'memory.sqlite3'):
            with memory._connection() as conn:
                conn.execute('INSERT INTO character_state VALUES (?,?)',('1',json.dumps(self.state)))
                conn.execute('INSERT INTO state_history(user_id,revision,payload) VALUES (?,?,?)',('1',7,json.dumps({
                    'before':self.state,'after':self.state,'jev':{'answers':{'secret':999}},'resolve_event':{'delta':3}})))
            with caller_scope(new_run_context(interface='telegram',agent_name='roleplay',user_id='1',is_owner=True,scope_type='telegram_message',scope_id='actor-test')):
                self.assert_narrative_only(json.loads(memory.roleplay_state('read')))
                self.assert_narrative_only(json.loads(memory.roleplay_state('history')))
                self.assert_narrative_only(json.loads(memory.roleplay_state('update',changes={'mood':'조심스러움'},reason='상대의 말을 조심스럽게 들음',expected_revision=7)))
            persisted = memory.load_state('1')
            self.assertEqual(persisted['hunger'],61)
            self.assertEqual(persisted['last_calculation'],self.state['last_calculation'])
