import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from commulingo_pipeline.evidence import snapshot, compile_evidence
from commulingo_pipeline.engine import Engine, Result
from commulingo_pipeline.store import Store, LostLease, BudgetUnavailable


class EvidenceTests(unittest.TestCase):
    def test_person_create_rejects_update_only_fields_in_tool_schema(self):
        from jsonschema import Draft202012Validator
        from runtime_tools.commulingo_people import COMMULINGO_PERSON_CREATE_TOOL, COMMULINGO_PERSON_UPDATE_TOOL
        for field,value in [('aliasEdits',[]),('careerEdits',[]),('sceneEdits',[]),('expectedRevision','revision')]:
            with self.subTest(field=field):
                create = COMMULINGO_PERSON_CREATE_TOOL['input_schema']['properties']['fields']
                update = COMMULINGO_PERSON_UPDATE_TOOL['input_schema']['properties']['fields']
                errors = list(Draft202012Validator(create).iter_errors({field:value}))
                self.assertTrue(any(e.validator=='additionalProperties' and field in e.message for e in errors))
                self.assertIn(field,update['properties'])

    def test_source_nul_normalization_precedes_hash_and_ranges(self):
        source = snapshot('https://example.org/source','Historical\x00document with enough context.')
        self.assertNotIn('\x00',source['body'])
        same = snapshot(source['url'],source['body'])
        self.assertEqual(source['id'],same['id'])

    def test_displayed_chunks_resolve_to_exact_unicode_source_ranges(self):
        from commulingo_pipeline.evidence import resolve_claim_chunks
        source = snapshot('https://example.org/source', '한글 근거 ' * 100)
        claim = {'field':'bio','claim':'Supported fact','source_id':source['id'],'chunks':[1,2]}
        resolved = resolve_claim_chunks([claim], {source['id']:source})
        self.assertEqual(resolved[0]['start'],240)
        self.assertEqual(resolved[0]['end'],len(source['body']))
        compiled = compile_evidence(resolved,{source['id']:source},{'bio'})
        self.assertEqual(compiled[0]['excerpt'],source['body'][240:])
        separate = resolve_claim_chunks([{**claim,'chunks':[0,2]}],{source['id']:source})
        self.assertEqual([(c['start'],c['end']) for c in separate],[(0,240),(480,600)])
        singular = {k:v for k,v in claim.items() if k!='chunks'}
        singular['chunk']=1
        self.assertEqual(resolve_claim_chunks([singular],{source['id']:source})[0]['start'],240)
        for invalid in ([99], [-1], [True], []):
            with self.assertRaises(ValueError):
                resolve_claim_chunks([{**claim,'chunks':invalid}],{source['id']:source})

    def test_exact_range_and_expiry(self):
        source = snapshot('https://example.org/source','A documented event happened in 1917.')
        claim = {'field':'bio','claim':'The event happened in 1917',
                 'source_id':source['id'],'start':2,'end':35}
        evidence = compile_evidence([claim],{source['id']:source},{'bio'})
        self.assertEqual(evidence[0]['excerpt'],source['body'][2:35])
        with self.assertRaises(ValueError):
            compile_evidence([claim],{source['id']:source},{'moment'})
        source['expires_at'] = datetime.now(timezone.utc)-timedelta(seconds=1)
        with self.assertRaises(ValueError):
            compile_evidence([claim],{source['id']:source},{'bio'})


class PlannerSelectionTests(unittest.TestCase):
    def candidates(self, ordinary, *, gaps=(), jobs=(), limit=40):
        from commulingo_pipeline.planner import Planner
        cur = Mock()
        cur.fetchall.side_effect = [ordinary, gaps, jobs]
        store = Mock()
        @contextmanager
        def transaction():
            yield cur
        store.transaction = transaction
        with patch('commulingo_pipeline.planner.report_mentions_by_term', return_value={}):
            return Planner(store, overlap_allow=[], exclude=[]).candidates(limit)

    def row(self, target):
        return dict(kind='person', action='update', target=target, topic='basics',
                    priority=20, baseline='new', reason='Missing facts')

    def test_person_commissions_use_importance_cap_and_grace(self):
        from commulingo_pipeline.planner import Planner
        from commulingo_pipeline.store import GRACE_PARAMS
        cur = Mock()
        cur.fetchall.side_effect = [[], [], []]
        store = Mock()
        @contextmanager
        def transaction():
            yield cur
        store.transaction = transaction
        with patch('commulingo_pipeline.planner.report_mentions_by_term', return_value={}):
            Planner(store).candidates(5)
        sql, params = cur.execute.call_args_list[0].args
        self.assertIn("100 - LEAST((SELECT count(DISTINCT e.event_id)", sql)  # importance ordering
        self.assertIn("CASE WHEN (SELECT count(DISTINCT e.event_id) FROM commulingo_history_event_people e WHERE e.person_id=p.id)>=6 THEN 12", sql)
        self.assertIn("a.stage='submit' AND a.value->>'status'='approved'", sql)  # grace after applied edits
        self.assertEqual({k:params[k] for k in GRACE_PARAMS}, GRACE_PARAMS)
        self.assertEqual((params['grace_important'], params['grace_other']), (14, 90))
        # Terms: grace after applied edits, substantial bodies and event twins stay out.
        self.assertIn("g.kind='term' AND g.target=t.id AND a.stage='submit'", sql)
        self.assertIn("length(t.body_ko) >= %(body_ko)s", sql)
        self.assertIn("t.id <> ALL(%(term_exclude)s::text[])", sql)
        self.assertEqual((params['term_grace'], params['body_ko'], params['body_en']), (90, 2000, 4500))
        self.assertEqual(params['term_exclude'], ['doctors-plot', 'kronstadt-rebellion-1921', 'leningrad-affair', 'volga-famine'])
        # New registrations, not existing entries, are checked against event titles.
        gap_sql, gap_params = cur.execute.call_args_list[1].args
        self.assertIn("lower(ev.title_ko)=lower(g.label_ko)", gap_sql)
        self.assertEqual(gap_params[0], ['battle-of-lake-khasan'])

    def test_term_candidates_are_ordered_by_body_and_report_mentions(self):
        from commulingo_pipeline.planner import Planner
        cur = Mock()
        rows = [dict(kind='term', action='update', target=t, topic='history', priority=40, baseline='b',
                     reason='r', body_empty=empty) for t, empty in
                (('quiet', True), ('popular', True), ('written', False), ('written-popular', False))]
        cur.fetchall.side_effect = [rows, [], []]
        store = Mock()
        @contextmanager
        def transaction():
            yield cur
        store.transaction = transaction
        with patch('commulingo_pipeline.planner.report_mentions_by_term',
                   return_value={'popular': 18, 'written-popular': 40}):
            selected = Planner(store, overlap_allow=[], exclude=[]).candidates(10)
        self.assertEqual([r['target'] for r in selected], ['popular', 'quiet', 'written-popular', 'written'])
        self.assertEqual([r['priority'] for r in selected], [32, 50, 51, 80])
        self.assertNotIn('body_empty', selected[0])

    def test_active_jobs_with_changed_baselines_do_not_consume_limit(self):
        for kind in ('person', 'term'):
            for action in ('create', 'update'):
                for status in ('ready', 'running', 'deferred', 'escalated'):
                    with self.subTest(kind=kind, action=action, status=status):
                        blocked = {**self.row('blocked'), 'kind':kind, 'action':action}
                        jobs = [{**blocked, 'baseline':'old', 'status':status}]
                        selected = self.candidates([blocked,self.row('next')],jobs=jobs,limit=1)
                        self.assertEqual([r['target'] for r in selected], ['next'])

    def test_completed_jobs_suppress_only_the_same_baseline_and_topic(self):
        row = self.row('person')
        complete = {**row, 'status':'complete'}
        self.assertEqual(self.candidates([row], jobs=[complete]), [])
        self.assertEqual(self.candidates([row], jobs=[{**complete,'baseline':'old'}])[0]['payload']['topics'], ['basics'])
        self.assertEqual(self.candidates([row], jobs=[{**complete,'topic':'bio'}])[0]['payload']['topics'], ['basics'])

    def test_explicit_gap_keeps_payload_without_duplicate_commission(self):
        gap = {**self.row('person'), 'priority':10, 'baseline':'',
               'gap_id':7, 'label_ko':'인물', 'label_en':'Person'}
        selected = self.candidates([self.row('person'),self.row('next')],gaps=[gap],limit=2)
        self.assertEqual([r['target'] for r in selected], ['person','next'])
        self.assertEqual(selected[0]['payload']['gap_id'], 7)

    def test_available_groups_still_alternate(self):
        rows = [self.row('p1'),self.row('p2'),
                {**self.row('t1'),'kind':'term'}, {**self.row('t2'),'kind':'term'}]
        self.assertEqual([r['target'] for r in self.candidates(rows)], ['p1','t1','p2','t2'])

    def test_topics_are_bundled_before_limit(self):
        rows = [self.row('p1'), {**self.row('p1'),'topic':'bio'},
                {**self.row('p1'),'topic':'sections'},self.row('p2')]
        selected = self.candidates(rows,limit=2)
        self.assertEqual([r['target'] for r in selected], ['p1','p2'])
        self.assertEqual(selected[0]['topic'], 'enrichment')
        self.assertEqual(selected[0]['payload']['topics'], ['basics','bio','sections'])

    def test_active_bundle_blocks_new_topics_and_complete_bundle_covers_members(self):
        row = self.row('p1')
        bundle = {**row,'topic':'enrichment','status':'ready','payload':{'topics':['basics','bio']}}
        self.assertEqual(self.candidates([row],jobs=[bundle]), [])
        self.assertEqual(self.candidates([row],jobs=[{**bundle,'status':'complete'}]), [])


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_follows_same_job_then_returns_to_scheduler(self):
        engine = Engine(Mock(),{})
        engine.run_one = AsyncMock(side_effect=[
            {'status':'ready','job_id':7,'stage':'draft'},
            {'status':'complete','job_id':7,'stage':'complete'},
            {'status':'idle'}])
        results = await engine.run_batch(limit=12,draft_only=False)
        self.assertEqual(len(results),3)
        self.assertEqual([c.kwargs['job_id'] for c in engine.run_one.call_args_list],[None,7,None])

    async def test_batch_stops_at_budget_draft_and_lease_boundaries(self):
        for status in ('draft_ready','lease_lost'):
            engine = Engine(Mock(),{})
            engine.run_one = AsyncMock(return_value={'status':status,'job_id':7})
            await engine.run_batch()
            engine.run_one.assert_awaited_once()

    async def test_batch_limits_and_explicit_job(self):
        engine = Engine(Mock(),{})
        engine.run_one = AsyncMock(return_value={'status':'ready','job_id':7})
        self.assertEqual(len(await engine.run_batch(limit=2)),2)
        engine.run_one.reset_mock()
        self.assertEqual(await engine.run_batch(max_seconds=0),[])
        engine.run_one.assert_not_awaited()
        engine.run_one.return_value = {'status':'complete','job_id':7}
        self.assertEqual(len(await engine.run_batch(job_id=7)),1)

    async def test_batch_finishes_approved_submit_at_limit_only_in_publish_mode(self):
        for draft_only in (True,False):
            engine=Engine(Mock(),{})
            engine.run_one=AsyncMock(side_effect=[{'status':'ready','stage':'submit','job_id':7},
                                                {'status':'complete','stage':'complete','job_id':7}])
            results=await engine.run_batch(limit=1,draft_only=draft_only)
            self.assertEqual(len(results),1 if draft_only else 2)
            if not draft_only:
                self.assertEqual(engine.run_one.await_args_list[-1].kwargs,
                                 {'job_id':7,'draft_only':False,'expected_stage':'submit'})
        engine.run_one=AsyncMock(return_value={'status':'ready','stage':'research','job_id':7})
        self.assertEqual(len(await engine.run_batch(limit=1,draft_only=False)),1)

    async def test_final_submit_does_not_run_changed_stage(self):
        store=Mock()
        store.claim.return_value={'id':7,'stage':'research'}
        research=AsyncMock()
        result=await Engine(store,{'research':research}).run_one(
            job_id=7,draft_only=False,expected_stage='submit')
        self.assertEqual(result['status'],'stage_changed')
        research.assert_not_awaited()
        store.reserve.assert_not_called()
        self.assertFalse(store.defer.call_args.kwargs['failed'])

    async def test_bundle_judges_each_card_topic_then_advances_to_sections(self):
        from commulingo_pipeline.stages import judge
        from commulingo_pipeline.engine import Usage
        from commulingo_pipeline.bundles import work_topics
        job = {'id':7,'kind':'person','target':'p','topic':'enrichment',
               'payload':{'topics':['basics','bio','sections']}}
        research = {'current':{'revision':'r1'},'baseline':'r1','status':'complete',
                    'reason':'All card facts already supported','inspected_sources':['source']}
        with patch('commulingo_pipeline.stages.service.call') as rpc:
            result = await judge(job,[{'stage':'research','value':research}],Usage(),.2)
        self.assertEqual([c.args[0]['topic'] for c in rpc.call_args_list],['basics','bio'])
        self.assertEqual({c.args[0]['expectedRevision'] for c in rpc.call_args_list},{'r1'})
        self.assertEqual(result.next_stage,'research')
        self.assertEqual(result.value['remaining_topics'],['sections'])
        self.assertEqual(work_topics({**job,'payload':{**job['payload'],**result.value}}),['sections'])

    async def test_bundle_unavailable_does_not_skip_topics(self):
        from commulingo_pipeline.stages import judge
        from commulingo_pipeline.engine import Usage
        job = {'id':7,'kind':'person','target':'p','topic':'enrichment',
               'payload':{'topics':['bio','sections']}}
        with patch('commulingo_pipeline.stages.service.call'):
            result = await judge(job,[{'stage':'research','value':{'status':'sources_unavailable'}}],Usage(),.2)
        self.assertEqual(result.status,'deferred')
        self.assertNotIn('remaining_topics',result.value)

    async def test_approved_card_advances_without_reusing_its_draft(self):
        from commulingo_pipeline.stages import submit, latest
        from commulingo_pipeline.engine import Usage
        job = {'id':7,'kind':'person','action':'update','target':'p','topic':'enrichment',
               'payload':{'topics':['bio','sections']}}
        artifacts = [{'stage':'draft','value':{'fields':{'bio':{'ko':'소개','en':'Bio'}},'sources':['source']}},
                     {'stage':'review','value':{'decision':'approve','reason':'Verified','checks':[]}}]
        with patch('commulingo_pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo_pipeline.stages.service.call',return_value={'status':'approved','suggestionId':9}), \
             patch.object(Store,'publication_slot') as slot:
            result = await submit(job,artifacts,Usage(),.2)
        slot.assert_not_called()
        self.assertEqual(result.next_stage,'research')
        self.assertEqual(result.status,'ready')
        self.assertEqual(result.value['remaining_topics'],['sections'])
        artifacts.append({'stage':'submit','value':result.value})
        self.assertEqual(latest(artifacts,'draft'),{})


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_gap_contract_is_single_requested_entry(self):
        from commulingo_pipeline.stages import Discover
        from commulingo_pipeline.engine import Usage
        from jsonschema import validate, ValidationError
        payload = {'material_id':'gap:2007','requested_kind':'term','label':'독립',
                   'body':'독립과 주변 인물 및 개념들'}
        candidate = {'kind':'term','target':'independence','label':'독립','mention':'독립',
                     'reason':'A historically important concept missing from the dictionary.'}
        async def model(**kwargs):
            schema = kwargs['tool']['input_schema']
            validate({'candidates':[candidate]},schema)
            for candidates in ([candidate,candidate],[{**candidate,'kind':'person'}],
                               [{**candidate,'mention':'주변'}], [{**candidate,'label':'다른 항목'}]):
                with self.assertRaises(ValidationError):
                    validate({'candidates':candidates},schema)
            self.assertIn('zero or one', kwargs['prompt'])
            self.assertNotIn('up to four', kwargs['prompt'])
            await kwargs['handler']({'candidates':[candidate]})
        with patch('commulingo_pipeline.stages.model_call',side_effect=model), patch('db.query_one',return_value=None):
            result = await Discover()({'id':1,'payload':payload},[],Usage(),.2)
        self.assertEqual(result.value['candidates'],[candidate])

    async def test_canary_wait_is_not_reported_as_daily_budget_failure(self):
        store = Mock()
        store.claim.return_value={'id':1,'stage':'submit','attempts':1}
        store.detail.return_value={'artifacts':[]}
        async def submit(*args):
            raise BudgetUnavailable('canary publication slots exhausted')
        result = await Engine(store,{'submit':submit}).run_one(draft_only=False)
        self.assertEqual(result['reason'],'canary publication slots exhausted')
        self.assertEqual(store.defer.call_args.args[1],result['reason'])
        self.assertFalse(store.defer.call_args.kwargs['failed'])

    async def test_model_stage_uses_real_dispatcher_for_artifact_schema(self):
        from commulingo_pipeline.stages import model_call, result_tool
        from commulingo_pipeline.prompts import spec
        from commulingo_pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        saved = []
        tool = result_tool({'type':'object','additionalProperties':False,
            'properties':{'reason':{'type':'string','minLength':5}},'required':['reason']})
        async def handler(value):
            saved.append(value)
            return 'OK: saved artifact'
        async def chat(*args,**kwargs):
            self.assertEqual([t['name'] for t in kwargs['tools']],['commulingo_pipeline_result'])
            with patch('tool_gateway.security.audit'):
                _, failed = await execute_tool(tool['name'],{'reason':2},kwargs['tool_handlers'],tool_schema=tool['input_schema'])
                self.assertTrue(failed)
                result, failed = await execute_tool(tool['name'],{'reason':'A sourced conclusion'},kwargs['tool_handlers'],tool_schema=tool['input_schema'])
                self.assertFalse(failed,result)
            kwargs['budget_tracker']['total_cost']=.01
        binding = SimpleNamespace(chat=chat,client=None,model='fixture',render_provider='deepseek',reasoning={})
        with patch('bot_config.resolve_agent_tool_loop',return_value=binding):
            await model_call(spec=spec('research'),prompt='Fixture',tool=tool,handler=handler,
                             reads=set(),usage=Usage(),budget=.2)
        self.assertEqual(saved,[{'reason':'A sourced conclusion'}])

    async def test_term_draft_compiles_evidence_and_keeps_original_revision(self):
        from commulingo_pipeline.stages import Draft
        from commulingo_pipeline.engine import Usage
        store = Mock()
        source = snapshot('https://example.org/definition','The archive defines the concept and its historical use.')
        store.sources.return_value = {source['id']:source}
        claim = {'field':'definition','claim':'Documented definition','source_id':source['id'],'start':0,'end':len(source['body'])}
        async def model(**kwargs):
            schema = kwargs['tool']['input_schema']['properties']['fields']['properties']
            self.assertNotIn('expectedRevision',schema)
            self.assertNotIn('evidence',schema)
            from jsonschema import Draft7Validator
            lookup_schema = kwargs['read_tools']['commulingo_people']['input_schema']
            validator = Draft7Validator(lookup_schema)
            self.assertNotIn('search_people',str(lookup_schema))
            for action,key in [('get_person','person_id'),('get_sections','person_id'),
                               ('get_term','term_id'),('get_office','office_id'),('get_event','event_id')]:
                self.assertTrue(validator.is_valid({'action':action,key:'fixture'}))
                self.assertFalse(validator.is_valid({'action':action}))
                self.assertFalse(validator.is_valid({'action':action,'q':'fixture'}))
            self.assertFalse(validator.is_valid({'action':'search','q':'fixture'}))
            from tool_gateway.results import ToolRejection
            call = AsyncMock(return_value='Registry entry')
            lookup = kwargs['read_wrap']('commulingo_people',call)
            with self.assertRaises(ToolRejection):
                await lookup(action='list_terms')
            for _ in range(3):
                await lookup(action='get_term',term_id='fixture')
            with self.assertRaises(ToolRejection):
                await lookup(action='get_term',term_id='fixture')
            self.assertEqual(call.await_count,3)
            await kwargs['handler']({'fields':{'definition':{'ko':'검증한 정의','en':'Verified definition'}}})
        with patch('commulingo_pipeline.stages.model_call',side_effect=model), patch('commulingo_pipeline.stages.service.call',return_value={}):
            result = await Draft(store)({'kind':'term','action':'update','topic':'definition','target':'fixture'},
                [{'stage':'research','value':{'claims':[claim],'baseline':'original-revision'}}],Usage(),.2)
        self.assertEqual(result.value['fields']['expectedRevision'],'original-revision')
        self.assertEqual(result.value['fields']['evidence'][0]['excerpt'],source['body'])
        self.assertEqual(result.next_stage,'validate')

    async def test_review_revisions_are_bounded_and_uncertainty_stays_internal(self):
        from commulingo_pipeline.stages import Review
        from commulingo_pipeline.engine import Usage
        job={'id':9,'kind':'person','action':'create','target':'fixture','topic':'basics'}
        artifacts=[{'stage':'research','value':{'baseline':''}},
                   {'stage':'draft','value':{'fields':{},'sources':[]}}]
        def handlers(reads,proposal,fetched,box):
            async def decide(**value):
                box.update(value)
            return {'commulingo_review_decision':decide}
        for decision,count,expected in [('revise',0,('research','ready')),
                                        ('revise',1,('research','ready')),
                                        ('revise',2,('research','ready')),
                                        ('escalate',0,('complete','escalated')),
                                        ('reject',0,('complete','complete')),
                                        ('approve',0,('submit','ready'))]:
            async def model(**kwargs):
                await kwargs['handler']({'decision':decision,'reason':'Verified feedback','checks':[],'resolved_risks':[]})
            previous=[{'stage':'review','value':{'decision':'revise'}} for _ in range(count)]
            with patch('commulingo_pipeline.stages.service.call',return_value=None) as rpc, \
                 patch('scripts.commulingo_person_reviewer.make_handlers',side_effect=handlers), \
                 patch('commulingo_pipeline.stages.model_call',side_effect=model):
                result=await Review()(job,artifacts+previous,Usage(),.2)
            self.assertEqual((result.next_stage,result.status),expected)
            self.assertEqual([c.args[0]['command'] for c in rpc.call_args_list],['read'])
        # Old lifetime counters do not reject a corrected draft.
        job['payload']={'review_revisions':20}
        decision='revise'
        async def model(**kwargs):
            await kwargs['handler']({'decision':'revise','reason':'Correct classification from existing sources',
                'checks':[],'resolved_risks':[],'needs_research':False})
        with patch('commulingo_pipeline.stages.service.call',return_value=None), \
             patch('scripts.commulingo_person_reviewer.make_handlers',side_effect=handlers), \
             patch('commulingo_pipeline.stages.model_call',side_effect=model):
            first=await Review()(job,artifacts,Usage(),.2)
            self.assertEqual(first.next_stage,'draft')
            history=artifacts+[{'stage':'review','value':first.value}]
            second=await Review()(job,history,Usage(),.2)
            self.assertEqual(second.next_stage,'draft')
            history.append({'stage':'review','value':second.value})
            stalled=await Review()(job,history,Usage(),.2)
            self.assertEqual(stalled.status,'escalated')
            # Fetching more evidence cannot disguise an unchanged patch.
            history.append({'stage':'draft','value':{'fields':{'evidence':[{}]},'sources':[]}})
            self.assertEqual((await Review()(job,history,Usage(),.2)).status,'escalated')
            history.append({'stage':'draft','value':{'fields':{'fate':{'kind':'','label':{'ko':'미확정','en':'Unconfirmed'}}},'sources':[]}})
            corrected=await Review()(job,history,Usage(),.2)
            self.assertEqual(corrected.next_stage,'draft')
            self.assertNotEqual(corrected.value['reviewed_content_hash'],first.value['reviewed_content_hash'])

    async def test_rereview_receives_earlier_verdicts_of_the_current_bundle(self):
        from commulingo_pipeline.stages import Review
        from commulingo_pipeline.engine import Usage
        job={'id':9,'kind':'person','action':'create','target':'fixture','topic':'basics'}
        stale={'stage':'review','value':{'decision':'revise','reason':'Earlier section verdict','checks':[]}}
        boundary={'stage':'submit','value':{'remaining_topics':['sections']}}
        artifacts=[stale,boundary,{'stage':'research','value':{'baseline':''}},
                   {'stage':'draft','value':{'fields':{},'sources':[]}},
                   {'stage':'review','value':{'decision':'revise','reason':'Fix the patronymic','needs_research':False,
                       'checks':[{'citation':'c','source':'s','quote':'q'*20,'finding':'부칭 표기 오류'}]}},
                   {'stage':'draft','value':{'fields':{'bio':{'ko':'수정','en':'fixed'}},'sources':[]}}]
        def handlers(reads,proposal,fetched,box):
            async def decide(**value):
                box.update(value)
            return {'commulingo_review_decision':decide}
        prompts=[]
        async def model(**kwargs):
            prompts.append(kwargs['prompt'])
            await kwargs['handler']({'decision':'approve','reason':'Corrections were made','checks':[],'resolved_risks':[]})
        with patch('commulingo_pipeline.stages.service.call',return_value=None), \
             patch('scripts.commulingo_person_reviewer.make_handlers',side_effect=handlers), \
             patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result=await Review()(job,artifacts,Usage(),.2)
            self.assertEqual(result.next_stage,'submit')
            first=await Review()(job,artifacts[2:4],Usage(),.2)
            self.assertEqual(first.next_stage,'submit')
        self.assertIn('This is review #2 of the same job',prompts[0])
        self.assertIn('Fix the patronymic',prompts[0])
        self.assertIn('부칭 표기 오류',prompts[0])
        self.assertNotIn('Earlier section verdict',prompts[0])  # previous bundle stays out
        self.assertNotIn('"q'+'q'*19,prompts[0])  # quotes are not replayed, only findings
        self.assertNotIn('This is review #',prompts[1])

    async def test_manually_resolved_original_is_not_replaced(self):
        from commulingo_pipeline.stages import submit
        from commulingo_pipeline.engine import Usage
        job={'id':9,'kind':'person','action':'create','target':'fixture',
             'payload':{'replaces_suggestion_id':7}}
        artifacts=[{'stage':'draft','value':{'fields':{},'sources':[]}},
                   {'stage':'review','value':{'decision':'approve','checks':[],'reason':'Verified'}}]
        for status in ('approved','rejected'):
            with patch('commulingo_pipeline.config.load',return_value={'phase':'live'}), \
                 patch('runtime_tools.commulingo_review_queue.suggestion',return_value={'status':status,'review_note':'Manual decision'}), \
                 patch('commulingo_pipeline.stages.service.call') as rpc:
                result=await submit(job,artifacts,Usage(),.2)
            rpc.assert_not_called()
            self.assertEqual(result.status,'complete')

    async def test_correction_replaces_only_after_approval_with_replayable_keys(self):
        from commulingo_pipeline.stages import submit
        from commulingo_pipeline.engine import Usage
        job={'id':9,'kind':'person','action':'create','target':'fixture','topic':'review-repair:7',
             'payload':{'replaces_suggestion_id':7}}
        artifacts=[{'stage':'draft','value':{'fields':{},'sources':[]}},
                   {'stage':'review','value':{'decision':'approve','checks':[],'reason':'Verified'}}]
        original={'status':'pending','target_type':'person','review_note':''}
        requests=[]
        def rpc(request):
            requests.append(request)
            if request.get('suggestionId')==7:
                original.update(status='rejected',review_note=request['note'])
                return {'status':'rejected','suggestionId':7}
            return {'status':'pending','suggestionId':8} if request['command']=='submit' else {'status':'approved','suggestionId':8}
        with patch('commulingo_pipeline.config.load',return_value={'phase':'live'}), \
             patch('runtime_tools.commulingo_review_queue.suggestion',side_effect=lambda _:dict(original)), \
             patch('commulingo_pipeline.stages.service.call',side_effect=rpc):
            first=await submit(job,artifacts,Usage(),.2)
            second=await submit(job,artifacts,Usage(),.2)
        self.assertEqual(first.value['status'],'approved')
        self.assertEqual(second.value['status'],'approved')
        self.assertEqual(requests[:3],requests[3:])
        self.assertEqual([r['command'] for r in requests[:3]],['review','submit','review'])
        self.assertFalse(requests[0]['approve'])
        self.assertTrue(requests[2]['approve'])
        with patch('commulingo_pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo_pipeline.stages.service.call') as write:
            with self.assertRaisesRegex(ValueError,'approved independent review'):
                await submit(job,artifacts[:-1]+[{'stage':'review','value':{'decision':'revise'}}],Usage(),.2)
        write.assert_not_called()

    async def test_research_rejects_topic_names_and_accepts_body_evidence(self):
        from commulingo_pipeline.stages import Research
        from commulingo_pipeline.engine import Usage
        from jsonschema import Draft202012Validator
        source=snapshot('https://example.org/history','A retrieved historical source with adequate context for the body.')
        store=Mock()
        store.job_sources.return_value={source['id']:source}
        async def model(**kwargs):
            value={'status':'ready','reason':'Verified history for the commissioned term.',
                   'claims':[{'field':'history','claim':'Verified history','source_id':source['id'],'chunk':0}]}
            schema=kwargs['tool']['input_schema']
            self.assertTrue(list(Draft202012Validator(schema).iter_errors(value)))
            value['claims'].append({**value['claims'][0],'field':'endYear'})
            with self.assertRaisesRegex(ValueError,'not a commissioned topic'):
                await kwargs['handler'](value)
            value['claims'].pop()
            value['claims'][0]['field']='body'
            self.assertFalse(list(Draft202012Validator(schema).iter_errors(value)))
            with self.assertRaisesRegex(ValueError,'missing evidence for endYear'):
                await kwargs['handler'](value)
            value['claims'].append({**value['claims'][0],'field':'endYear'})
            await kwargs['handler'](value)
        job={'id':30,'kind':'term','action':'update','target':'fixture','topic':'history'}
        with patch('commulingo_pipeline.stages.service.call',return_value={'revision':'original'}), \
             patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result=await Research(store)(job,[{'stage':'validate','value':{'error':'400: evidence required for endYear'}}],Usage(),.2)
        self.assertEqual(result.next_stage,'draft')
        self.assertEqual(result.value['claims'][0]['field'],'body')

    async def test_person_draft_gets_real_group_ids_and_descriptions(self):
        from commulingo_pipeline.stages import Draft
        from commulingo_pipeline.engine import Usage
        store=Mock()
        store.sources.return_value={}
        groups=[{'id':'foreign-statesmen','title_ko':'정치가들','blurb_ko':'분류의 실제 기준'}]
        async def model(**kwargs):
            props=kwargs['tool']['input_schema']['properties']['fields']['properties']
            self.assertEqual(props['groupId']['enum'],['foreign-statesmen'])
            self.assertEqual(props['role']['properties']['category']['enum'],['foreign-statesman'])
            from jsonschema import Draft202012Validator
            validator=Draft202012Validator(kwargs['tool']['input_schema'])
            for collection,edits in (('aliases','aliasEdits'),('career','careerEdits'),('scenes','sceneEdits')):
                errors=list(validator.iter_errors({'fields':{collection:[],edits:[]}}))
                self.assertTrue(any(e.validator=='not' for e in errors))
            self.assertIn('분류의 실제 기준',kwargs['prompt'])
            self.assertIn('draft_target',kwargs['prompt'])
            self.assertIn('720',kwargs['prompt'])
            with self.assertRaises(ValueError):
                await kwargs['handler']({'fields':{'groupId':'people'}})
            await kwargs['handler']({'fields':{'groupId':'foreign-statesmen'}})
        with patch('runtime_tools.commulingo_people._list_groups',return_value=groups), \
             patch('runtime_tools.commulingo_people._list_categories',return_value=[{'id':'foreign-statesman'}]), \
             patch('commulingo_pipeline.stages.model_call',side_effect=model), patch('commulingo_pipeline.stages.service.call',return_value={}):
            result=await Draft(store)({'id':921,'kind':'person','action':'update','topic':'basics','target':'fixture'},
                [{'stage':'research','value':{'baseline':'original','claims':[]}}],Usage(),.2)
        self.assertEqual(result.value['fields']['groupId'],'foreign-statesmen')
        self.assertEqual(result.value['fields']['expectedRevision'],'original')

    async def test_enrichment_draft_cannot_reclassify_an_existing_person(self):
        from commulingo_pipeline.stages import Draft
        from commulingo_pipeline.engine import Usage
        store=Mock()
        store.sources.return_value={}
        groups=[{'id':'bolshevik','title_ko':'볼셰비키','blurb_ko':'설명'}]
        seen={}
        async def model(**kwargs):
            seen['props']=set(kwargs['tool']['input_schema']['properties']['fields']['properties'])
            await kwargs['handler']({'fields':{'epithet':{'ko':'수정','en':'fixed'}}})
        current={'revision':'v1','group':'bolshevik','groupId':'bolshevik',
                 'role':{'officeId':'party-leadership','category':''}}
        with patch('runtime_tools.commulingo_people._list_groups',return_value=groups), \
             patch('runtime_tools.commulingo_people._list_categories',return_value=[{'id':'socialist-bloc-leader'}]), \
             patch('commulingo_pipeline.stages.model_call',side_effect=model), patch('commulingo_pipeline.stages.service.call',return_value={}):
            await Draft(store)({'id':1,'kind':'person','action':'update','topic':'basics','target':'stalin'},
                [{'stage':'research','value':{'baseline':'v1','claims':[],'current':current}}],Usage(),.2)
            self.assertFalse({'role','group','groupId'} & seen['props'])
            # A person without any classification still receives one.
            await Draft(store)({'id':2,'kind':'person','action':'update','topic':'basics','target':'new'},
                [{'stage':'research','value':{'baseline':'v1','claims':[],'current':{'revision':'v1','role':{}}}}],Usage(),.2)
            self.assertTrue({'role','groupId'} <= seen['props'])

    async def test_submission_replay_uses_identical_receipt_keys(self):
        from commulingo_pipeline.stages import submit
        from commulingo_pipeline.engine import Usage
        job = {'id':9,'kind':'term','action':'update','target':'test'}
        artifacts = [{'stage':'draft','value':{'fields':{'definition':{'ko':'정의','en':'Definition'}},'sources':['source']}},
                     {'stage':'review','value':{'decision':'approve','reason':'Independent check','checks':[]}}]
        requests = []
        def rpc(request):
            requests.append(request)
            return {'suggestionId':12,'status':'approved' if request['command']=='review' else 'pending'}
        with patch('commulingo_pipeline.config.load',return_value={'phase':'live'}),patch('commulingo_pipeline.stages.service.call',side_effect=rpc):
            await submit(job,artifacts,Usage(),.2)
            await submit(job,artifacts,Usage(),.2)
        self.assertEqual(requests[:2],requests[2:])

    async def test_source_stage_resumes_from_committed_artifact(self):
        store = Mock()
        job = {'id':1,'stage':'draft','kind':'person','attempts':1,'lease_token':'lease'}
        store.claim.return_value = job
        store.detail.return_value = {'artifacts':[{'stage':'research','value':{'claims':['saved']}}]}
        async def draft(job, artifacts, usage, budget):
            self.assertEqual(artifacts[0]['value']['claims'],['saved'])
            return Result({'draft':'new'},'validate')
        result = await Engine(store,{'draft':draft}).run_one()
        self.assertEqual(result['stage'],'validate')
        store.finish_stage.assert_called_once()
        store.reserve.assert_not_called()

    async def test_lost_lease_cancels_inflight_stage_and_keeps_unknown_cost(self):
        store = Mock()
        store.claim.return_value = {'id':1,'stage':'research','kind':'term','attempts':1}
        store.detail.return_value = {'artifacts':[]}
        store.heartbeat.side_effect = LostLease('1')
        cancelled = asyncio.Event()
        async def research(*args):
            args[2].started = True
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()
        research.uses_llm = True
        result = await Engine(store,{'research':research},heartbeat_seconds=.01).run_one()
        self.assertEqual(result['status'],'lease_lost')
        self.assertTrue(cancelled.is_set())
        store.finish_stage.assert_not_called()
        store.settle.assert_not_called()

    async def test_draft_only_never_calls_review_or_submit(self):
        store = Mock()
        store.claim.return_value = {'id':1,'stage':'submit'}
        result = await Engine(store,{}).run_one()
        self.assertEqual(result['status'],'draft_ready')
        store.reserve.assert_not_called()
        self.assertFalse(store.defer.call_args.kwargs['failed'])

    async def test_evaluation_reviews_but_never_publishes(self):
        store = Mock()
        store.claim.return_value = {'id':1,'stage':'review','attempts':1}
        store.detail.return_value = {'artifacts':[]}
        async def review(*args):
            return Result({'decision':'approve'},'submit')
        result = await Engine(store,{'review':review}).run_one(allow_review=True)
        self.assertEqual(result['stage'],'submit')
        store.claim.return_value = {'id':1,'stage':'submit'}
        result = await Engine(store,{}).run_one(allow_review=True)
        self.assertEqual(result['status'],'draft_ready')

    async def test_missing_field_evidence_returns_to_research(self):
        from commulingo_pipeline.stages import validate
        from commulingo_pipeline.engine import Usage
        job={'kind':'person','action':'create','target':'fixture'}
        artifacts=[{'stage':'draft','value':{'fields':{'citizenship':'lao'},'sources':[]}}]
        with patch('commulingo_pipeline.stages.service.call',side_effect=ValueError(
                '400: evidence must identify the claim and page/section supporting citizenship')):
            result=await validate(job,artifacts,Usage(),.2)
        self.assertEqual(result.next_stage,'research')
        self.assertTrue(result.value['needs_research'])

    async def test_term_missing_year_evidence_returns_to_research_with_bounded_retries(self):
        from commulingo_pipeline.stages import validate
        from commulingo_pipeline.engine import Usage
        job = {'kind':'term','action':'update','target':'fixture'}
        draft = {'stage':'draft','value':{'fields':{'startYear':1990},'sources':['source']}}
        error = '400: evidence required for startYear'
        with patch('commulingo_pipeline.stages.service.call',side_effect=ValueError(error)):
            for previous in range(3):
                artifacts = [draft, *[{'stage':'validate','value':{'error':error}} for _ in range(previous)]]
                result = await validate(job,artifacts,Usage(),.2)
                self.assertEqual(result.next_stage,'research')
                self.assertEqual(result.status,'escalated' if previous==2 else 'ready')
                self.assertTrue(result.value['needs_research'])

    async def test_empty_sources_return_to_research_instead_of_rewriting(self):
        from commulingo_pipeline.stages import validate
        from commulingo_pipeline.engine import Usage
        job={'kind':'person','action':'update','target':'fixture'}
        artifacts=[{'stage':'draft','value':{'fields':{'bio':{'en':'A biography'}},'sources':[]}}]
        with patch('commulingo_pipeline.stages.service.call',side_effect=ValueError(
                '400: sources must be a non-empty list of references for this edit')):
            result=await validate(job,artifacts,Usage(),.2)
        self.assertEqual(result.next_stage,'research')
        self.assertTrue(result.value['needs_research'])

    async def test_evidence_failures_do_not_exhaust_format_repairs(self):
        from commulingo_pipeline.stages import validate
        from commulingo_pipeline.engine import Usage
        job = {'kind':'term','action':'update','target':'fixture'}
        artifacts = [{'stage':'draft','value':{'fields':{'definition':{'en':'Definition'}},'sources':[]}},
                     *[{'stage':'validate','value':{'error':'400: evidence required for startYear'}} for _ in range(3)],
                     {'stage':'validate','value':{'error':'revision_conflict'}}]
        with patch('commulingo_pipeline.stages.service.call',side_effect=ValueError('definition too long')):
            result = await validate(job,artifacts,Usage(),.2)
        self.assertEqual((result.next_stage,result.status),('draft','ready'))


@unittest.skipUnless(os.getenv('COMMULINGO_PIPELINE_TEST_PORT'), 'isolated PostgreSQL opt-in')
class PostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg2
        @contextmanager
        def connect():
            conn = psycopg2.connect(host='127.0.0.1',port=int(os.environ['COMMULINGO_PIPELINE_TEST_PORT']),
                                    user='postgres',dbname='commulingo_integrity_test')
            try:
                with conn:
                    yield conn
            finally:
                conn.close()
        cls.store = Store(connect)
        with cls.store.transaction() as cur:
            cur.execute(Path('commulingo_pipeline/schema.sql').read_text())

    def test_review_repair_survives_replay_and_is_not_consolidated(self):
        row={'id':77,'target_type':'person_section','target_id':'fixture','action':'create',
             'patch_json':{'slug':'history'},'source_refs':['https://example.org']}
        decision={'decision':'revise','reason':'Correct the date'}
        with ThreadPoolExecutor(2) as pool:
            ids=list(pool.map(lambda _:self.store.enqueue_review_repair(row,decision),range(2)))
        self.assertEqual(ids[0],ids[1])
        with self.store.transaction() as cur:
            cur.execute('SELECT * FROM commulingo_pipeline_jobs WHERE id=%s',(ids[0],))
            job=cur.fetchone()
        self.assertEqual(job['action'],'update')
        self.assertEqual(job['payload']['topics'],['sections'])
        self.store.consolidate(apply=True)
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_pipeline_jobs SET status='complete' WHERE id=%s",(ids[0],))
        self.assertEqual(self.store.enqueue_review_repair(row,decision),ids[0])

    def setUp(self):
        with self.store.transaction() as cur:
            cur.execute('TRUNCATE commulingo_pipeline_attempts,commulingo_pipeline_jobs,commulingo_pipeline_budget,commulingo_pipeline_artifacts,commulingo_pipeline_job_sources,commulingo_pipeline_publications RESTART IDENTITY')

    def add(self,target='test'):
        return self.store.enqueue(kind='person',action='update',target=target,topic='bio',reason='test')

    def test_consolidate_preserves_progress_and_resumes_section_in_same_job(self):
        self.add('bundle')
        self.store.enqueue(kind='person',action='update',target='bundle',topic='sections',reason='More context')
        progressing = self.add('progressing')
        self.store.enqueue(kind='person',action='update',target='progressing',topic='basics',reason='Facts')
        with self.store.transaction() as cur:
            cur.execute("INSERT INTO commulingo_pipeline_artifacts(job_id,stage,value) VALUES (%s,'research','{}')",(progressing,))
        preview = self.store.consolidate()
        self.assertEqual((preview['jobs_before'],preview['bundles'],preview['merged_jobs']),(2,1,1))
        self.assertEqual(len([j for j in self.store.list_jobs() if j['status']=='ready']),4)
        self.store.consolidate(apply=True)
        self.assertEqual(self.store.consolidate(apply=True)['jobs_before'],0)
        jobs = self.store.list_jobs()
        parent = next(j for j in jobs if j['target']=='bundle' and j['status']=='ready')
        child = next(j for j in jobs if j['target']=='bundle' and j['status']=='cancelled')
        self.assertEqual(child['payload']['bundled_into'],parent['id'])
        self.assertEqual(parent['payload']['topics'],['bio','sections'])
        self.assertEqual(len([j for j in jobs if j['target']=='progressing' and j['status']=='ready']),2)
        job = self.store.claim(job_id=parent['id'])
        self.store.finish_stage(job,{'remaining_topics':['sections']},next_stage='research')
        resumed = self.store.claim(job_id=parent['id'])
        self.assertEqual(resumed['payload']['remaining_topics'],['sections'])
        self.assertEqual(resumed['payload']['topics'],['bio','sections'])

    def test_consolidate_keeps_all_explicit_gap_links(self):
        from commulingo_pipeline.bundles import gap_ids
        for topic,gap in [('bio',11),('basics',12)]:
            self.store.enqueue(kind='person',action='update',target='bundle',topic=topic,
                               reason='Gap',payload={'gap_id':gap})
        self.store.consolidate(apply=True)
        parent = next(j for j in self.store.list_jobs() if j['status']=='ready')
        self.assertEqual(set(gap_ids(parent['payload'])),{11,12})

    def test_live_releases_only_canary_publication_waits(self):
        for target,reason in [('canary','canary publication slots exhausted'),('budget','daily budget unavailable')]:
            job_id = self.add(target)
            with self.store.transaction() as cur:
                cur.execute("UPDATE commulingo_pipeline_jobs SET status='deferred',stage='submit',last_error=%s WHERE id=%s",(reason,job_id))
        self.assertEqual(self.store.release_publication_waits(),1)
        self.assertEqual(self.store.release_publication_waits(),0)
        states = {j['target']:j['status'] for j in self.store.list_jobs()}
        self.assertEqual(states,{'canary':'ready','budget':'deferred'})

    def test_handoff_approval_resumes_bundle_and_ignores_old_reviews(self):
        from commulingo_pipeline.stages import latest
        job_id = self.store.enqueue(kind='person',action='update',target='handoff',topic='enrichment',
            reason='test',payload={'topics':['bio','sections']})
        with self.store.transaction() as cur:
            cur.execute("INSERT INTO commulingo_agent_suggestions(id,target_id,target_type,status,action) VALUES (98761,'handoff','person','approved','update'),(98762,'handoff','person_section','pending','update')")
            cur.execute("INSERT INTO commulingo_pipeline_artifacts(job_id,stage,value) VALUES (%s,'review',%s)",
                        (job_id,'{"suggestionId":98761}'))
            cur.execute("UPDATE commulingo_pipeline_jobs SET status='escalated' WHERE id=%s",(job_id,))
        self.assertEqual(self.store.reconcile_reviews(),1)
        detail = self.store.detail(job_id)
        self.assertEqual(detail['job']['status'],'ready')
        self.assertEqual(detail['job']['payload']['remaining_topics'],['sections'])
        self.assertEqual(latest(detail['artifacts'],'review'),{})
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_pipeline_jobs SET status='escalated' WHERE id=%s",(job_id,))
        self.assertEqual(self.store.reconcile_reviews(),0)
        with self.store.transaction() as cur:
            cur.execute("INSERT INTO commulingo_pipeline_artifacts(job_id,stage,value) VALUES (%s,'review',%s)",
                        (job_id,'{"suggestionId":98762}'))
            cur.execute("UPDATE commulingo_pipeline_jobs SET status='escalated' WHERE id=%s",(job_id,))
        self.assertEqual(self.store.reconcile_reviews(),0)
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_agent_suggestions SET status='approved' WHERE id=98762")
        self.assertEqual(self.store.reconcile_reviews(),1)
        self.assertEqual(self.store.detail(job_id)['job']['status'],'complete')
        with self.store.transaction() as cur:
            cur.execute('DELETE FROM commulingo_agent_suggestions WHERE id IN (98761,98762)')

    def test_concurrent_claim_and_expired_owner_cannot_commit(self):
        self.add()
        self.assertIsNone(self.add())
        with ThreadPoolExecutor(2) as pool:
            jobs = list(pool.map(lambda _:self.store.claim(),range(2)))
        old = next(j for j in jobs if j)
        self.assertEqual(sum(j is not None for j in jobs),1)
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_pipeline_jobs SET lease_until=now()-interval '1 second'")
        new = self.store.claim()
        with self.assertRaises(LostLease):
            self.store.finish_stage(old,{'invalid':True},next_stage='draft')
        self.store.finish_stage(new,{'valid':True},next_stage='draft')
        detail = self.store.detail(new['id'])
        self.assertEqual(len(detail['artifacts']),1)
        self.assertEqual(detail['job']['stage'],'draft')

    def test_atomic_daily_budget_and_idempotent_settlement(self):
        def reserve(_):
            try:
                return self.store.reserve('2.00',lane='person')
            except BudgetUnavailable:
                return None
        with ThreadPoolExecutor(2) as pool:
            reservations = list(pool.map(reserve,range(2)))
        token = next(t for t in reservations if t)
        self.assertEqual(sum(t is not None for t in reservations),1)
        self.store.reserve('1.00',lane='review')
        self.store.settle(token,'0.50')
        self.store.settle(token,'0.50')
        with self.assertRaises(ValueError):
            self.store.settle(token,'0.40')
        self.store.reserve('1.00',lane='term')

    def test_snapshot_expiry_preserves_metadata(self):
        source = snapshot('https://example.org/history','Original evidence.',datetime.now(timezone.utc)-timedelta(days=15))
        self.store.save_source(source)
        job = self.add()
        self.store.link_source(job,source['id'])
        self.store.expire_sources()
        retained = self.store.job_sources(job)[source['id']]
        self.assertIsNone(retained['body'])
        self.assertEqual(retained['content_hash'],source['content_hash'])

    def test_cache_keys_and_canary_limit(self):
        source = snapshot('https://example.org/cached','Verified source text')
        self.store.save_source(source)
        self.store.cache_source('fetch_url',{'url':source['url'],'max_length':100},source['id'])
        self.assertIsNone(self.store.cached_source('fetch_url',{'url':source['url'],'max_length':200}))
        self.assertEqual(self.store.cached_source('fetch_url',{'max_length':100,'url':source['url']})['id'],source['id'])
        one = {'id':self.add('one'),'kind':'person','action':'update'}
        two = {'id':self.add('two'),'kind':'person','action':'update'}
        self.store.publication_slot(one,1)
        self.store.publication_slot(one,1)
        with self.assertRaises(BudgetUnavailable):
            self.store.publication_slot(two,1)

    def test_discovery_commits_candidates_and_cursor_atomically(self):
        material = {'material_id':'report:discovery-fixture','content_hash':'fixed-hash','body':'A missing concept appears.'}
        job_id = self.store.enqueue(kind='term',action='create',target='material-fixture',
            topic='discovery',reason='test',stage='discover',payload=material)
        job = self.store.claim()
        self.assertEqual(job['id'],job_id)
        self.store.finish_stage(job,{'candidates':[{'kind':'term','target':'missing-concept',
            'reason':'Needed to understand this document','mention':'missing concept'}]},next_stage='complete',status='complete')
        created = [j for j in self.store.list_jobs() if j['target']=='missing-concept']
        self.assertEqual(len(created),1)
        self.assertEqual(created[0]['stage'],'research')
        with self.store.transaction() as cur:
            cur.execute('SELECT content_hash FROM commulingo_pipeline_materials WHERE material_id=%s',(material['material_id'],))
            self.assertEqual(cur.fetchone()['content_hash'],'fixed-hash')

    def test_planner_sql_and_material_hash_are_stable(self):
        from commulingo_pipeline.planner import Planner
        # No source data copied from production; only a minimal public-document fixture.
        with self.store.transaction() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS research_documents
                (slug text PRIMARY KEY,filename text,title text,markdown text,status text,content_sha256 text)''')
            cur.execute("DELETE FROM commulingo_pipeline_materials WHERE material_id='report:pipeline-fixture'")
            cur.execute("INSERT INTO research_documents(slug,filename,title,markdown,status,content_sha256) VALUES ('pipeline-fixture','pipeline-fixture.md','Fixture','A concept appears in this public document.','public','fixture-sha') ON CONFLICT DO NOTHING")
        planner = Planner(self.store)
        planner.plan()
        material = next(m for m in planner.materials(limit=10000) if m['material_id']=='report:pipeline-fixture')
        discovery_id = self.store.enqueue(kind='term',action='create',target='material-selection-fixture',
            topic='discovery',reason='test',stage='discover',payload=material)
        for status in ('ready','running','deferred','escalated'):
            with self.store.transaction() as cur:
                cur.execute('UPDATE commulingo_pipeline_jobs SET status=%s WHERE id=%s',(status,discovery_id))
            self.assertNotIn(material['material_id'],[m['material_id'] for m in planner.materials(limit=10000)])
        with self.store.transaction() as cur:
            cur.execute("UPDATE research_documents SET markdown=markdown || ' Changed.' WHERE slug='pipeline-fixture'")
        self.assertIn(material['material_id'],[m['material_id'] for m in planner.materials(limit=10000)])
        with self.store.transaction() as cur:
            cur.execute("UPDATE research_documents SET markdown='A concept appears in this public document.' WHERE slug='pipeline-fixture'")
            cur.execute("UPDATE commulingo_pipeline_jobs SET status='complete' WHERE id=%s",(discovery_id,))
        self.assertIn(material['material_id'],[m['material_id'] for m in planner.materials(limit=10000)])
        with self.store.transaction() as cur:
            cur.execute('''INSERT INTO commulingo_pipeline_materials(material_id,content_hash)
                VALUES (%s,%s) ON CONFLICT(material_id) DO UPDATE SET content_hash=EXCLUDED.content_hash''',
                (material['material_id'],material['content_hash']))
        self.assertNotIn(material['material_id'],[m['material_id'] for m in planner.materials(limit=10000)])
        with self.store.transaction() as cur:
            cur.execute("DELETE FROM research_documents WHERE slug='pipeline-fixture'")
            cur.execute("DELETE FROM commulingo_pipeline_materials WHERE material_id='report:pipeline-fixture'")

    def test_budget_release_and_drain_preserve_error_waits(self):
        author = self.add('budget-author')
        review = self.add('budget-review')
        failure = self.add('failed')
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_pipeline_jobs SET status='deferred',available_at=now()+interval '1 hour',last_error='daily budget reserved or spent' WHERE id=ANY(%s)",([author,review],))
            cur.execute("UPDATE commulingo_pipeline_jobs SET stage='review' WHERE id=%s",(review,))
            cur.execute("UPDATE commulingo_pipeline_jobs SET status='deferred',last_error='source failed',available_at=now()+interval '1 hour' WHERE id=%s",(failure,))
        reservation=self.store.reserve('.69',lane='person',cap='1',review_fraction='.3')
        self.store.settle(reservation,'.69')
        self.assertEqual(self.store.release_budget_waits(cap='1',amount='.2',review_fraction='.3'),1)
        self.assertEqual(self.store.claim(stages=['review','submit'])['id'],review)
        self.assertEqual(self.store.detail(author)['job']['status'],'deferred')
        self.assertEqual(self.store.release_budget_waits(cap='10',amount='.2',review_fraction='.3'),1)
        self.assertEqual(self.store.detail(author)['job']['status'],'ready')
        self.assertEqual(self.store.detail(failure)['job']['status'],'deferred')

    def test_attempt_metrics_do_not_double_count_cost_or_hide_unknown(self):
        job_id=self.add()
        job=self.store.claim(job_id=job_id)
        attempt=self.store.start_attempt(job)
        reservation=self.store.reserve('.2',lane='person',job_id=job_id)
        self.store.link_attempt_budget(attempt,reservation)
        self.store.finish_attempt(attempt,'error',None,'provider timeout',12,{'rounds_used':3})
        since=datetime.now(timezone.utc)-timedelta(hours=1)
        rows=self.store.efficiency(since)
        row=next(r for r in rows if r['kind']=='person' and r['action']=='update')
        self.assertEqual(row['attempts'],1)
        self.assertEqual(row['unsettled'],1)
        self.assertIsNone(row['actual'])
        self.store.settle(reservation,'.07')
        row=next(r for r in self.store.efficiency(since) if r['kind']=='person')
        self.assertAlmostEqual(row['actual'],.07)
        self.assertEqual(row['unsettled'],0)
        self.assertEqual(row['seconds'],12)
