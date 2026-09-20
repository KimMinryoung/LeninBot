import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from llm.call_registry import Decision, DecisionResult, GenerationResult
from runtime_tools import roleplay_memory as memory
from runtime_tools import roleplay_jev as jev
from runtime_tools.roleplay_dynamics import with_defaults
from runtime_tools.roleplay_pacing import policy_for, turn_time_scope
from tool_gateway.security import caller_scope, new_run_context


def initial(**changes):
    return with_defaults({**memory.STATE_DEFAULTS, 'hunger':60, 'fatigue':50, 'pain':10, 'tension':40,
                          'resolve':60, 'clarity':70, 'humiliation':10, 'conditions_initialized':True,
                          'activity':'light', 'location':'복도', **changes})


def verdict(**labels):
    return {'status':'classified', 'labels':{'mode':'scene','elapsed':'brief','event':'none','activity':'light', **labels},
            'duration_estimate':{'elapsed_minutes':3}, 'uncertain':[], 'model':'jev-test', 'answers':{}, 'rules_version':1}


class ProjectionTests(unittest.TestCase):
    def project(self, text, state=None, **labels):
        with turn_time_scope(policy_for(text)):
            return jev.project(state or initial(), text, [], verdict(**labels), 'test')

    def test_move_and_advice_only_reach_destination(self):
        state, applied = self.project('감방으로 보내. 회복하려면 어떻게 할까?', location='감방')
        self.assertEqual((state['location'],state['scene_minute']),('감방',3))
        self.assertEqual(state['hunger'],60.15)
        self.assertEqual(state['injuries'],[])
        self.assertEqual(applied['event'],'none')
        with self.assertRaises(ValueError):
            self.project('감방으로 보내. 회복하려면 어떻게 할까?', elapsed='explicit')

    def test_meal_rule_and_delivery_not_eating(self):
        state, _ = self.project('빵과 수프를 다 먹었다', event='meal', intensity='mild')
        self.assertEqual(state['hunger'],25.15)  # elapsed +0.15, fixed full meal -35
        delivery, _ = self.project('식사가 도착했다', event='none')
        self.assertEqual(delivery['hunger'],60.15)

    def test_explicit_short_duration_not_inflated_to_budget(self):
        state, _ = self.project('2분 쉬어', elapsed='explicit', activity='rest')
        self.assertEqual(state['scene_minute'],2)
        for mode in ('discussion','past'):
            state, applied = self.project('하루 쉬면 어떨까?', mode=mode)
            self.assertEqual(state,initial())
            self.assertTrue(applied['no_change'])

    def test_injury_treatment_and_code_owned_progress(self):
        injury = dict(id='arm',description='팔 타박',severity=2,trend='stable',treated=False,progress_minutes=20)
        state, _ = self.project('팔을 처치받았다', initial(injuries=[injury]), event='treatment', injury_0='treated')
        self.assertEqual((state['injuries'][0]['treated'],state['injuries'][0]['trend'],state['injuries'][0]['progress_minutes']), (True,'recovering',0))
        self.assertLess(state['pain'],10)
        new, _ = self.project('넘어져 팔을 다쳤다', event='injury', intensity='moderate', new_injury='arm', new_severity='mild')
        self.assertEqual(new['injuries'][0]['severity'],1)

    def test_schedule_and_interruption_preserve_endpoint_effects(self):
        planned, _ = self.project('한 시간 뒤 방문 약속을 등록해', mode='plan', plan_action='schedule')
        self.assertEqual(planned['story_events'][0]['due_minute'],60)
        after, applied = self.project('두 시간 쉬어', planned, elapsed='explicit',activity='rest',event='meal')
        self.assertEqual(after['scene_minute'],60)
        self.assertTrue(applied['interrupted'])
        self.assertEqual(after['hunger'],63)
        self.assertEqual(after['story_events'][0]['status'],'ready')
        ended, _ = self.project('방문이 끝났다', after, elapsed='0', story_0='complete')
        self.assertEqual(ended['story_events'][0]['status'],'completed')

    def test_low_confidence_does_not_guess_activity_or_event(self):
        for field in ('mode','elapsed','event'):
            v=verdict();v['labels'].pop(field)
            with turn_time_scope(policy_for('이동해')), self.assertRaises(ValueError):
                jev.project(initial(),'이동해',[],v,'test')
        with self.assertRaises(ValueError):
            self.project('한 시간 쉬어',elapsed='explicit',activity='keep')

    def test_recognition_survives_uncertain_time_without_movement(self):
        text = '"내일 아침은 성실하게 임해야 한다. 잘해 줬으니 오늘 소원을 들어주지."'
        v = verdict(event='recognition', intensity='moderate', location='감방', activity='rest')
        v['labels'].pop('elapsed')
        before = initial()
        with turn_time_scope(policy_for(text)):
            state, applied = jev.project(before, text, [], v, 'recognition')
        self.assertEqual(state['resolve'], before['resolve'] + 5)
        self.assertEqual(state['tension'], before['tension'] - 3)
        for field in ('scene_minute', 'clock', 'location', 'activity', 'hunger', 'fatigue', 'injuries'):
            self.assertEqual(state[field], before[field])
        self.assertEqual(applied['deferred_components'], ['time', 'scene_conditions'])
        self.assertEqual(state['resolve_events'][-1]['kind'], 'recognition')
        self.assertEqual(len(state['resolve_events']), len(before['resolve_events']) + 1)
        v['labels'].pop('intensity')
        with turn_time_scope(policy_for(text)):
            fixed, _ = jev.project(before, text, [], v, 'fixed')
        self.assertEqual(fixed['resolve'], state['resolve'])

    def test_sexual_event_subtypes_have_distinct_effects(self):
        results = []
        for event, act, delta in [('sexual_harassment','verbal',1),('sexual_assault','touch',3),('rape','penetration',8)]:
            state, _ = self.project('현재 사건', event=event, sexual_act=act, intensity='moderate', elapsed='0')
            self.assertEqual(state['resolve'], initial()['resolve'] - delta)
            results.append(state['tension'])
        self.assertLess(results[0], results[1]); self.assertLess(results[1], results[2])
        self.assertNotIn('sexual_coercion', jev.build_questions(initial(),[])['event']['criteria'])

    def test_no_penetration_and_unknown_act_cannot_be_rape(self):
        for text, event, act in [('삽입 없음','rape','penetration'),('삽입은 하지 않았다','rape','penetration'),('추행했다','rape','touch'),('성적 가해','sexual_unspecified','unknown')]:
            with self.assertRaises(ValueError): self.project(text, event=event, sexual_act=act, intensity='moderate', elapsed='0')

    def test_confirmed_nonpenetrative_event_survives_unknown_time(self):
        v = verdict(event='sexual_assault', sexual_act='touch', intensity='moderate', activity='restrained')
        v['labels'].pop('elapsed')
        before = initial()
        with turn_time_scope(policy_for('삽입 없는 추행이 있었다')):
            state, applied = jev.project(before, '삽입 없는 추행이 있었다', [], v, 'touch')
        self.assertEqual(state['resolve'], before['resolve'] - 3)
        self.assertEqual(state['scene_minute'], before['scene_minute'])
        self.assertIn('time', applied['deferred_components'])
        v['labels'].pop('sexual_act')
        with self.assertRaises(ValueError): jev.project(before, '추행이 있었다', [], v, 'unknown')

    def test_restrained_subject_gets_no_rest_or_sleep_recovery(self):
        state, _ = self.project('붙잡혀 움직이지 못했다. 삽입 없음', event='sexual_assault', sexual_act='touch', intensity='moderate', activity='restrained')
        self.assertGreater(state['fatigue'], initial()['fatigue'])
        self.assertEqual(state['pain'], initial()['pain'])
        self.assertEqual(state['injuries'], [])
        with self.assertRaises(ValueError): self.project('가해가 계속됐다',event='rape',sexual_act='penetration',intensity='moderate',activity='rest')

    def test_correction_and_unknown_initial_values(self):
        fixed, _ = self.project('의지를 40으로 정정해',mode='correction')
        self.assertEqual(fixed['resolve'],40)
        with self.assertRaises(ValueError):
            self.project('의지 40이면 어떻게 돼?',mode='correction')
        empty=initial(hunger=None,pain=None)
        initialized, _ = self.project('잠깐 이동해',empty,initial_hunger='50',initial_pain='unknown')
        self.assertEqual(initialized['hunger'],50.15)
        self.assertIsNone(initialized['pain'])


class AutomaticTransactionTests(unittest.TestCase):
    def setUp(self):
        estimator=patch.object(jev,'estimate_duration',return_value={'elapsed_minutes':3});estimator.start();self.addCleanup(estimator.stop)
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        patcher=patch.object(memory,'MEMORY_PATH',Path(tmp.name)/'state.sqlite3');patcher.start();self.addCleanup(patcher.stop)
        with memory._connection() as conn:
            conn.execute('INSERT INTO character_state VALUES (?,?)',('1',json.dumps(initial())))
        scope=caller_scope(new_run_context(interface='telegram',agent_name='roleplay',user_id='1',is_owner=True,
                                           scope_type='telegram_message',scope_id='100'))
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def run_turn(self, value, text='빵을 다 먹었다', scope='100'):
        with patch.object(jev,'classify',return_value=value),turn_time_scope(policy_for(text)):
            return jev.adjudicate_turn(1,text,[],scope)

    def test_automatic_once_and_agent_cannot_override(self):
        result=self.run_turn(verdict(event='meal'))
        self.assertEqual(result['status'],'applied')
        after=memory.load_state(1)
        self.assertEqual(after['hunger'],25.15)
        replay=self.run_turn(verdict(event='meal'))
        self.assertTrue(replay['replayed']);self.assertNotIn('decision',replay)
        self.assertEqual(memory.load_state(1),after)
        for args in [dict(action='time',temporal={}),dict(action='update',changes={'hunger':0}),
                     dict(action='reset'),dict(action='update',changes={'activity':'sleep'}),
                     dict(action='update',changes={'mood':'차분'},resolve_event={'kind':'kindness','intensity':3})]:
            with self.assertRaises(PermissionError):memory.roleplay_state(reason='임의 선택',expected_revision=after['revision'],**args)
        with self.assertRaises(PermissionError): memory.roleplay_person('delete', person_id='guard')
        memory.roleplay_state('update',changes={'goal':'소식 묻기'},reason='현재 목적',expected_revision=after['revision'])
        self.assertEqual(memory.load_state(1)['goal'],'소식 묻기')
        schema=memory.ROLEPLAY_STATE_TOOL['input_schema']
        self.assertNotIn('time',schema['properties']['action']['enum'])
        self.assertNotIn('hunger',schema['properties']['changes']['properties'])

    def test_failure_or_partial_decision_preserves_state(self):
        before=memory.load_state(1)
        for i,v in enumerate([{'status':'unavailable','reason':'transport'},verdict(event='injury')]):
            outcome=self.run_turn(v,scope=str(i))
            self.assertEqual(outcome['status'],'deferred')
            self.assertEqual(memory.load_state(1),before)
        self.assertEqual(memory.load_state(2)['revision'],0)

    def test_duration_failure_preserves_state_and_explicit_bypasses_estimator(self):
        before = memory.load_state(1)
        with patch.object(jev, 'estimate_duration', side_effect=ValueError('bad duration')) as estimator:
            result = self.run_turn(verdict(), scope='duration-failed')
            self.assertEqual(result['status'], 'deferred')
            self.assertEqual(memory.load_state(1), before)
            self.assertEqual(estimator.call_count, 1)
            result = self.run_turn(verdict(elapsed='explicit', activity='rest'), text='한 시간 쉬어', scope='explicit')
            self.assertEqual(result['status'], 'applied')
            self.assertEqual(memory.load_state(1)['scene_minute'], 60)
            self.assertEqual(estimator.call_count, 1)
            self.assertTrue(self.run_turn(verdict(), scope='explicit')['replayed'])
            self.assertEqual(estimator.call_count, 1)

    def test_conflict_is_not_overwritten(self):
        def concurrent(*args):
            with memory._connection() as conn:
                s=initial(revision=1,hunger=42)
                conn.execute('UPDATE character_state SET payload=? WHERE user_id=?',(json.dumps(s),'1'))
            return verdict(event='meal')
        with patch.object(jev,'classify',side_effect=concurrent),turn_time_scope(policy_for('먹었다')):
            result=jev.adjudicate_turn(1,'먹었다',[],'c')
        self.assertEqual(result['status'],'deferred')
        self.assertEqual(memory.load_state(1)['hunger'],42)

    def test_focused_jev_retry_preserves_confident_event(self):
        def answer(label, confidence): return {'choice': label, 'confidence': confidence}
        first = Decision(answers={'mode':answer('scene',.99), 'event':answer('public_submission',.9), 'intensity':answer('moderate',.5)}, model='jev')
        second = Decision(answers={'intensity':answer('moderate',.9)}, model='jev')
        with patch.object(jev, 'decide_detailed', side_effect=[DecisionResult(decision=first),DecisionResult(decision=second)]) as decide:
            result = jev.classify('잘해 줬으니 소원을 들어주지', initial(), [], [])
        self.assertEqual(result['labels']['event'], 'public_submission')
        self.assertEqual(result['labels']['intensity'], 'moderate')
        self.assertEqual(set(decide.call_args_list[1].args[2]), {'intensity'})
        self.assertEqual(result['event_review']['answers'], second.answers)

    def test_real_adapter_rejects_malformed_and_low_confidence(self):
        d=Decision(answers={'mode':{'choice':'scene','confidence':.2},'event':{'choice':'invented','confidence':1}},model='jev')
        with patch.object(jev,'decide_detailed',return_value=DecisionResult(decision=d)):
            result=jev.classify('이동해',initial(),[],[])
        self.assertNotIn('mode',result['labels']);self.assertNotIn('event',result['labels'])
        self.assertIn('mode',result['uncertain'])
        with patch.object(jev,'decide_detailed',return_value=DecisionResult(error_kind='transport')):
            self.assertEqual(jev.classify('이동해',initial(),[],[])['status'],'unavailable')


class DurationTests(unittest.TestCase):
    def test_duration_only_valid_integer_within_single_event_cap(self):
        for value in (True, -1, 11, 1.5, '3', None):
            with patch.object(jev, 'generate_detailed', return_value=GenerationResult(text=json.dumps({'elapsed_minutes':value,'reason':'arrival'}))):
                with self.assertRaises(ValueError): jev.estimate_duration('감방으로 보내',initial(),verdict())
        with patch.object(jev, 'generate_detailed', return_value=GenerationResult(text='{"elapsed_minutes": 4, "reason": "arrival"}')):
            self.assertEqual(jev.estimate_duration('감방으로 보내',initial(),verdict())['elapsed_minutes'],4)

    def test_failed_or_truncated_output_cannot_advance(self):
        for result in (GenerationResult(error_kind='transport'),GenerationResult(text='{}'),GenerationResult(text='[]'),GenerationResult(text='not json'),GenerationResult(text='{"elapsed_minutes": 4, "reason": "arrival"}',truncated=True)):
            with patch.object(jev,'generate_detailed',return_value=result),self.assertRaises(ValueError):
                jev.estimate_duration('감방으로 보내',initial(),verdict())
