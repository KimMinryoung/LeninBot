import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from llm.call_registry import Decision, DecisionResult

from runtime_tools import roleplay_memory as memory, roleplay_turn as turn, roleplay_jev as jev
from runtime_tools.roleplay_dynamics import with_defaults
from tool_gateway.security import caller_scope, new_run_context


class PostDraftTests(unittest.TestCase):
    def test_pending_candidates_use_latest_retry_probabilities(self):
        results = [DecisionResult(decision=Decision(model='test', answers={'mode': {
            'choice': label, 'confidence': .4, 'probabilities': probabilities}}))
            for label, probabilities in [('scene', {'scene': .6, 'discussion': .4}),
                                          ('discussion', {'scene': .3, 'discussion': .7})]]
        with patch.object(turn, 'resolve', return_value=SimpleNamespace(extra={'enabled': True})), \
             patch.object(turn, 'decide_detailed', side_effect=results), \
             self.assertRaises(jev.PendingChoice) as caught:
            turn.decide('test', {}, {'mode': jev.choice('mode', {'scene': 'scene', 'discussion': 'discussion'})})
        self.assertEqual(caught.exception.candidates[0][0], 'discussion')

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        p=patch.object(memory,'MEMORY_PATH',Path(temp.name)/'state.sqlite3');p.start();self.addCleanup(p.stop)
        self.before=with_defaults({**memory.STATE_DEFAULTS,'hunger':30,'fatigue':30,'pain':10,'tension':40,'resolve':60,'clarity':60,'humiliation':30,
            'conditions_initialized':True,'clock':{'date':'1939-04-27','time':'20:00','daypart':'evening'},'activity':'light'})
        with memory._connection() as conn:conn.execute('INSERT INTO character_state VALUES (?,?)',('1',json.dumps(self.before)))
        ctx=caller_scope(new_run_context(interface='telegram',agent_name='roleplay',user_id='1',is_owner=True,scope_type='telegram_message',scope_id='77'))
        ctx.__enter__();self.addCleanup(ctx.__exit__,None,None,None)
        self.text='아침에 집중 심문을 진행하고 66명 명단 진술을 재현한다'
        self.auth={'user_text':self.text,'labels':{'mode':'scene','transition':'next_morning','span':'session'}}
        self.verdict={'status':'classified','labels':{'mode':'scene','event':'implicating_others','intensity':'moderate','activity':'light'},'uncertain':[],'model':'test'}

    def prepare(self,stage=None,final=True):
        with patch.object(turn,'decide',return_value={'labels':{'within_scope':'yes'}}), patch.object(turn,'review_reply',return_value={'approved':final,'issues':[] if final else ['contradiction']}), \
             patch.object(jev,'classify',return_value=deepcopy(self.verdict)) as classified, \
             patch.object(jev,'estimate_duration',return_value={'elapsed_minutes':90}):
            prepared=turn.prepare(self.text,self.before,stage['people'] if stage else [],[], '77','아침에 명단 진술을 마쳤다',self.auth,stage)
        self.assertEqual(classified.call_args.kwargs['draft'],'아침에 명단 진술을 마쳤다')
        return prepared

    def stage(self):
        with turn.staged_memory('1') as staged:
            memory.roleplay_memory('save',key='새 장면',content='아침 명단 진술')
            memory.roleplay_person('save',person_id='investigator',changes={'name':'조사관'})
            memory.roleplay_state('update',changes={'scene':'아침 심문 종료'},reason='초안',expected_revision=0)
            self.assertEqual(memory.load_state('1')['scene'],'아침 심문 종료')
        return staged

    def test_staged_writes_publish_only_with_approved_reply_and_replay_once(self):
        stage=self.stage()
        self.assertEqual(memory.load_state('1'),self.before)
        self.assertEqual(memory.load_notes('1'),[])
        prepared=self.prepare(stage)
        result=jev.adjudicate_turn('1',self.text,[],'77',prepared=prepared)
        self.assertEqual(result['reply'],'아침에 명단 진술을 마쳤다')
        state=memory.load_state('1')
        self.assertEqual(state['clock']['date'],'1939-04-28')
        self.assertEqual(state['clock']['daypart'],'morning')
        self.assertEqual(state['clock']['time'],'07:30')
        self.assertEqual(state['clock']['uncalculated_minutes'],600)
        self.assertEqual(state['hunger'],34.5)
        self.assertEqual(state['scene_minute'],690) # Timeline gap retained, no fabricated overnight sleep effects.
        self.assertEqual(state['resolve_events'][-1]['kind'],'implicating_others')
        self.assertEqual(len(state['resolve_events']),1) # Not 66 penalties, nor 3 stacked event types.
        self.assertEqual(state['scene'],'아침 심문 종료')
        self.assertTrue(memory.load_notes('1'));self.assertTrue(memory.load_people('1'))
        again=jev.adjudicate_turn('1',self.text,[],'77',prepared=prepared)
        self.assertTrue(again['replayed']);self.assertEqual(memory.load_state('1'),state)
        self.assertEqual(turn.committed_reply('1','77'),result['reply'])

    def test_review_findings_are_advisory_and_nothing_is_saved_before_commit(self):
        stage=self.stage()
        prepared=self.prepare(stage,final=False)
        self.assertEqual(prepared['applied']['review_issues'],['contradiction'])
        self.assertFalse(prepared['verdict']['final_review']['approved'])
        self.assertIn('서술 검토 지적: contradiction',turn.feedback_line({'status':'applied','applied':prepared['applied']},prepared['state']))
        # Preparing commits nothing; the staged records and state wait for adjudicate_turn.
        self.assertEqual(memory.load_state('1'),self.before)
        self.assertEqual(memory.load_notes('1'),[]);self.assertEqual(memory.load_people('1'),[])
        self.assertIsNone(turn.committed_reply('1','77'))

    def test_no_scope_gate_the_classifier_settles_the_draft(self):
        with patch.object(turn,'decide') as gate, patch.object(jev,'classify',return_value=deepcopy(self.verdict)), \
             patch.object(jev,'estimate_duration',return_value={'elapsed_minutes':90}), patch.object(turn,'review_reply',return_value={'approved':True,'issues':[]}):
            prepared=turn.prepare(self.text,self.before,[],[],'77','감방에서 자고 다음 날 식사했다',self.auth)
        gate.assert_not_called(); self.assertTrue(prepared['verdict']['scope_review']['skipped'])

    def test_review_failure_does_not_discard_or_falsely_approve_turn(self):
        with patch.object(jev, 'classify', return_value=deepcopy(self.verdict)), \
             patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 5}), \
             patch.object(turn, 'review_reply', side_effect=ValueError('최종 서술 검토 응답 형식 오류')):
            prepared = turn.prepare(self.text, self.before, [], [], 'review-down', '초안', self.auth)
        self.assertIsNone(prepared['verdict']['final_review']['approved'])
        self.assertTrue(prepared['applied']['review_unavailable'])
        self.assertEqual(memory.load_state('1'), self.before)
        committed = jev.adjudicate_turn('1', self.text, [], 'review-down', prepared=prepared)
        self.assertEqual(committed['status'], 'applied')
        self.assertEqual(committed['reply'], '초안')

    def test_ready_narrative_cue_does_not_block_short_scene_or_next_morning(self):
        from runtime_tools.roleplay_story import apply_story_updates
        before = apply_story_updates({**self.before, 'participants': []}, [{
            'op': 'schedule', 'id': 'delayed-reaction-1565', 'when_alone': True,
            'title': '혼자 있을 때 반응', 'source': '포화'}])
        for transition in ('current', 'next_morning'):
            auth = {'user_text': '몸을 살핀다', 'labels': {'mode': 'scene', 'transition': transition,
                    'time_scope': 'none' if transition == 'current' else 'day_skip', 'span': 'brief'}}
            v = {'status': 'classified', 'labels': {'mode': 'scene', 'event': 'none', 'activity': 'light'},
                 'uncertain': [], 'model': 'test'}
            with self.subTest(transition=transition), patch.object(jev, 'classify', return_value=v), \
                 patch.object(jev, 'estimate_duration', return_value={'elapsed_minutes': 5}), \
                 patch.object(turn, 'review_reply', return_value={'approved': True, 'issues': []}):
                prepared = turn.prepare('몸을 살핀다', before, [], [], 'cue', '몸을 살폈다', auth)
            self.assertFalse(prepared['applied'].get('interrupted'))
            self.assertEqual(prepared['applied']['minutes'], 5)
        self.assertEqual(memory.load_state('1'),self.before)

    def test_concurrent_notes_conflict_rolls_back_state_and_records(self):
        stage=self.stage();prepared=self.prepare(stage)
        memory.roleplay_memory('save',key='다른 기록',content='별도 변경')
        with self.assertRaises(ValueError):jev.adjudicate_turn('1',self.text,[],'77',prepared=prepared)
        self.assertEqual(memory.load_state('1'),self.before)
        self.assertEqual(memory.load_people('1'),[])
        self.assertIsNone(turn.committed_reply('1','77'))

    def test_legacy_cached_turn_cannot_publish_unsettled_new_draft(self):
        prepared=self.prepare()
        with memory._connection() as c:c.execute('INSERT INTO automatic_turns VALUES (?,?,?)',('1','77',json.dumps({'status':'unchanged'})))
        self.assertEqual(jev.adjudicate_turn('1',self.text,[],'77',prepared=prepared)['status'],'deferred')
        self.assertEqual(memory.load_state('1'),self.before)

    def test_review_rejects_malformed_empty_truncated_and_inconsistent_results(self):
        profile=SimpleNamespace(provider='deepseek_anthropic',extra={'enabled':True})
        import bot_config
        connection=SimpleNamespace(base_url=bot_config.DEEPSEEK_ANTHROPIC_BASE_URL)
        for text,truncated,error in [('not json',False,None),('',False,None),('{}',False,None),('{"approved":true,"issues":["conflict"]}',False,None),('{"approved":true,"issues":[]}',True,None),('',False,'transport')]:
            with self.subTest(text=text,truncated=truncated,error=error), patch.object(turn,'resolve',return_value=profile), patch.object(turn,'resolve_provider_connection',return_value=connection), patch.object(turn,'generate_detailed',return_value=SimpleNamespace(text=text,truncated=truncated,error_kind=error)):
                with self.assertRaises(ValueError):turn.review_reply('draft',self.before,self.before)

    def test_disabled_review_skips_both_models_without_warning_or_false_approval(self):
        with patch.object(turn,'resolve',return_value=SimpleNamespace(extra={'enabled':False})), \
             patch.object(turn,'generate_detailed') as generate, patch.object(turn,'screen_reply') as screen, \
             patch.object(jev,'classify',return_value=deepcopy(self.verdict)), \
             patch.object(jev,'estimate_duration',return_value={'elapsed_minutes':5}):
            prepared=turn.prepare(self.text,self.before,[],[],'review-off','초안',self.auth)
        generate.assert_not_called()
        screen.assert_not_called()
        self.assertEqual(prepared['verdict']['final_review'],
                         {'approved':None,'issues':[],'status':'skipped','reason':'disabled'})
        self.assertNotIn('review_unavailable',prepared['applied'])
        self.assertNotIn('review_issues',prepared['applied'])
        committed=jev.adjudicate_turn('1',self.text,[],'review-off',prepared=prepared)
        self.assertEqual(committed['status'],'applied')
        self.assertEqual(committed['reply'],'초안')

    def test_review_requires_actor_endpoint_and_only_sends_changed_records(self):
        import bot_config
        profile=SimpleNamespace(provider='deepseek_anthropic',extra={'enabled':True})
        connection=SimpleNamespace(base_url=bot_config.DEEPSEEK_ANTHROPIC_BASE_URL)
        staged={'baseline':{'notes':[('old','old content')],'people':[]},'records':{'notes':[('old','old content'),('new','new content')],'people':[]}}
        with patch.object(turn,'resolve',return_value=profile),patch.object(turn,'resolve_provider_connection',return_value=connection),patch.object(turn,'generate_detailed',return_value=SimpleNamespace(text='{"approved":true,"issues":[]}',truncated=False,error_kind=None)) as generate:
            self.assertTrue(turn.review_reply('draft',self.before,self.before,staged)['approved'])
            sent=json.loads(generate.call_args.args[1]);self.assertEqual(sent['new_records']['notes'],[['new','new content']]);self.assertNotIn('resolve',sent['settled_state'])
        with patch.object(turn,'resolve',return_value=profile),patch.object(turn,'resolve_provider_connection',return_value=SimpleNamespace(base_url='https://unexpected.invalid')),patch.object(turn,'generate_detailed') as generate:
            with self.assertRaises(ValueError):turn.review_reply('draft',self.before,self.before)
        generate.assert_not_called()
