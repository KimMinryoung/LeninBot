import asyncio
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from commulingo_pipeline.draft_repair import DraftRepair
from commulingo_pipeline.evidence import SourceHandles, snapshot, resolve_claim_chunks
from commulingo_pipeline.engine import Engine, Result, Usage
from commulingo_pipeline.stages import Draft, validate
from scripts.commulingo_write_session import draft_id


class EvidenceContracts(TestCase):
    def test_uncertain_fate_has_a_supported_schema_value(self):
        from runtime_tools.commulingo_people import COMMULINGO_PERSON_CREATE_TOOL
        from jsonschema import validate as schema_validate, ValidationError
        schema=COMMULINGO_PERSON_CREATE_TOOL['input_schema']['properties']['fields']['properties']['fate']
        schema_validate({'kind':'','label':{'ko':'사망 경위 미확정','en':'Circumstances unconfirmed'}},schema)
        with self.assertRaises(ValidationError):
            schema_validate({'kind':'unknown','label':{'ko':'미확정','en':'Unknown'}},schema)

    def test_short_and_legacy_ids_roundtrip_and_unknown_has_ranges(self):
        source=snapshot('https://example.org/archive','Documented fact. '*80)
        sources={source['id']:source}
        handles=SourceHandles(sources)
        for name in ('S1',source['id']):
            result=handles.resolve([{'source_id':name,'field':'body','claim':'fact','chunks':[0]}],sources)
            self.assertEqual(result[0]['source_id'],source['id'])
            self.assertEqual(resolve_claim_chunks(result,sources)[0]['start'],0)
        with self.assertRaisesRegex(ValueError,'S1: chunks 0..'):
            handles.resolve([{'source_id':'S99'}],sources)
        other=snapshot('https://example.org/other','Another page. '*10)
        self.assertEqual(handles.handle(other),'S2')
        self.assertEqual(handles.handle(source),'S1')
        # A later snapshot of the same URL keeps the handle and becomes its current target.
        later=snapshot('https://example.org/archive','Documented fact. '*80+'Appended page.')
        self.assertEqual(handles.handle(later),'S1')
        self.assertEqual(handles.ids['S1'],later['id'])

    def test_pages_of_one_url_merge_into_one_numbering(self):
        from commulingo_pipeline.evidence import SourcePages, SOURCE_CHUNK_CHARS
        pages=SourcePages()
        first,span1,created=pages.absorb('https://example.org/long','A'*600)
        self.assertTrue(created); self.assertEqual(span1,(0,600))
        second,span2,created=pages.absorb('https://example.org/long','B'*500)
        self.assertTrue(created)
        self.assertEqual(span2,(601,1101))
        self.assertEqual(second['body'],'A'*600+'\n'+'B'*500)
        self.assertEqual(span2[0]//SOURCE_CHUNK_CHARS,2,'page 2 starts in chunk 2, not at chunk 0')
        again,span_again,created=pages.absorb('https://example.org/long','A'*600)
        self.assertFalse(created); self.assertIs(again,second); self.assertEqual(span_again,span1)
        # Snapshots a job already holds for one URL are merged oldest first.
        from datetime import datetime,timezone,timedelta
        t=datetime(2026,9,19,tzinfo=timezone.utc)
        old={**snapshot('https://example.org/p','page one. '*30,now=t),}
        new={**snapshot('https://example.org/p','page two. '*30,now=t+timedelta(minutes=1))}
        single=snapshot('https://example.org/q','only page. '*30,now=t)
        seeded=SourcePages()
        merged=seeded.seed({old['id']:old,new['id']:new,single['id']:single})
        self.assertEqual([m['url'] for m in merged],['https://example.org/p'])
        self.assertTrue(merged[0]['body'].startswith('page one. ') and merged[0]['body'].endswith('page two. '))
        self.assertIs(seeded.current['https://example.org/q'],single)

    def test_expanded_ranges_preserve_all_evidence_above_old_limit(self):
        source=snapshot('https://example.org/archive','Documented fact. '*100)
        claim={'field':'body','claim':'fact','source_id':source['id'],'chunks':[0,2]}
        result=resolve_claim_chunks([claim]*32,{source['id']:source})
        self.assertEqual(len(result),64)
        self.assertEqual({(c['start'],c['end']) for c in result},{(0,240),(480,720)})

    def test_valid_draft_survives_downstream_error_and_repairs_follow_the_current_draft(self):
        repair=DraftRepair({'name':'draft','input_schema':{'type':'object','properties':{
            'fields':{'type':'object','properties':{'years':{'type':'string'}}}},'required':['fields']}})
        repair.prepare({'fields':{'years':'present'}})
        old=draft_id(repair.draft)
        self.assertIn(old,repair.feedback('years must be a range'))
        repair.prepare({'draft_id':old,'repairs':[{'op':'set','path':'/fields/years','value':'1900–'}]})
        self.assertNotEqual(draft_id(repair.draft),old)
        # An echoed earlier ID no longer costs a round: the call holds one draft.
        repair.prepare({'draft_id':old,'repairs':[{'op':'set','path':'/fields/years','value':'1900–1950'}]})
        self.assertEqual(repair.draft['args']['fields']['years'],'1900–1950')


class BatchContracts(IsolatedAsyncioTestCase):
    async def test_budget_block_drains_review_then_free_stages(self):
        engine=Engine(Mock(),{})
        engine.run_one=AsyncMock(side_effect=[
            {'status':'budget_deferred','job_id':1,'blocked_stage':'draft'},
            {'status':'budget_deferred','job_id':2,'blocked_stage':'review'},
            {'status':'complete','job_id':3,'stage':'complete'}, {'status':'idle'}])
        await engine.run_batch(draft_only=False)
        calls=engine.run_one.await_args_list
        self.assertEqual(calls[1].kwargs['claim_stages'],['validate','judge','submit','review'])
        self.assertEqual(calls[2].kwargs['claim_stages'],['validate','judge','submit'])
        self.assertEqual(calls[3].kwargs['claim_stages'],['validate','judge','submit'])

    async def test_explicit_job_budget_wait_stops_without_other_work(self):
        engine=Engine(Mock(),{})
        engine.run_one=AsyncMock(return_value={'status':'budget_deferred','job_id':1,'blocked_stage':'draft'})
        await engine.run_batch(job_id=1,draft_only=False)
        engine.run_one.assert_awaited_once()

    async def test_failed_stage_records_attempt_and_unknown_cost_is_not_zeroed(self):
        store=Mock()
        store.claim.return_value={'id':1,'stage':'research','kind':'person','attempts':1}
        store.detail.return_value={'artifacts':[]}
        async def fail(job,artifacts,usage,budget):
            usage.started=True
            usage.tracker['rounds_used']=2
            raise RuntimeError('provider failed')
        fail.uses_llm=True
        result=await Engine(store,{'research':fail}).run_one(draft_only=False)
        self.assertEqual(result['status'],'error')
        store.settle.assert_not_called()
        args=store.finish_attempt.call_args.args
        self.assertEqual(args[1],'error')
        self.assertEqual(args[3],'provider failed')
        self.assertEqual(args[5]['rounds_used'],2)


class DraftContracts(IsolatedAsyncioTestCase):
    def fixture(self):
        source=snapshot('https://example.org/archive','A documented definition supported by this archive.')
        store=Mock()
        store.sources.return_value={source['id']:source}
        claim={'field':'definition','claim':'Definition','source_id':source['id'],'start':0,'end':len(source['body'])}
        job={'id':5,'kind':'term','action':'update','target':'fixture','topic':'definition'}
        artifacts=[{'stage':'research','value':{'claims':[claim],'baseline':'original'}}]
        return store,job,artifacts

    async def test_storage_validation_is_repaired_in_same_call_and_keeps_revision(self):
        store,job,artifacts=self.fixture()
        rpc=Mock(side_effect=[ValueError('400: invalid period label'),{}])
        async def model(**kw):
            self.assertEqual(kw['read_tools']['commulingo_people']['input_schema']['properties']['action']['enum'],
                ['get_person','get_term','get_office','get_event','get_sections'])
            try:
                await kw['handler']({'fields':{'definition':{'ko':['정의'],'en':['Definition']}}})
            except ValueError as exc:
                import re
                current=re.search(r'draft_id=([a-f0-9]+)',str(exc))[1]
            else:
                self.fail('invalid storage draft accepted')
            await kw['handler']({'draft_id':current,'repairs':[
                {'op':'set','path':'/fields/definition/en/0','value':'Correct definition'}]})
        usage=Usage()
        with patch('commulingo_pipeline.stages.model_call',side_effect=model), patch('commulingo_pipeline.stages.service.call',rpc):
            result=await Draft(store)(job,artifacts,usage,.2)
        self.assertEqual(result.next_stage,'validate')
        self.assertEqual(result.value['fields']['expectedRevision'],'original')
        self.assertEqual(result.value['fields']['definition']['en'],'Correct definition')
        self.assertEqual(usage.tracker['preflight_failures'],1)
        self.assertTrue(usage.tracker['preflight_passed'])
        self.assertEqual(rpc.call_count,2)

    async def test_missing_evidence_preserves_draft_and_routes_to_research(self):
        store,job,artifacts=self.fixture()
        async def model(**kw):
            await kw['handler']({'fields':{'definition':{'ko':['정의'],'en':['Definition']}}})
        with patch('commulingo_pipeline.stages.model_call',side_effect=model), patch('commulingo_pipeline.stages.service.call',side_effect=ValueError('400: evidence required for body')):
            result=await Draft(store)(job,artifacts,Usage(),.2)
        self.assertIn('rejected_draft',result.value)
        self.assertNotIn('fields',result.value)
        artifacts.append({'stage':'draft','value':result.value})
        validated=await validate(job,artifacts,Usage(),.2)
        self.assertEqual(validated.next_stage,'research')
        self.assertTrue(validated.value['needs_research'])


class BudgetDrainRegression(IsolatedAsyncioTestCase):
    async def test_review_requesting_research_does_not_block_remaining_writes(self):
        engine=Engine(Mock(),{})
        engine.run_one=AsyncMock(side_effect=[
            {'status':'budget_deferred','job_id':1,'blocked_stage':'draft'},
            {'status':'ready','job_id':2,'stage':'research'},
            {'status':'complete','job_id':3,'stage':'complete'}, {'status':'idle'}])
        await engine.run_batch(draft_only=False)
        self.assertIsNone(engine.run_one.await_args_list[2].kwargs['job_id'])

    async def test_large_saved_research_reaches_draft_without_research_retry(self):
        store,job,artifacts=DraftContracts().fixture()
        artifacts[0]['value']['claims'] *= 63
        async def model(**kw):
            await kw['handler']({'fields':{'definition':{'ko':['정의'],'en':['Definition']}}})
        with patch('commulingo_pipeline.stages.model_call',side_effect=model) as call, patch('commulingo_pipeline.stages.service.call',return_value={}):
            result=await Draft(store)(job,artifacts,Usage(),.2)
        call.assert_called_once()
        self.assertEqual(result.next_stage,'validate')
        self.assertEqual(len(result.value['fields']['evidence']),63)


import os
import unittest

@unittest.skipUnless(os.getenv('COMMULINGO_FRONTEND_CONTAINER')=='commulingo-python-rpc'
    and os.getenv('COMMULINGO_PIPELINE_TEST_PORT')=='55439','isolated pipeline DB/RPC required')
class StorageDraftContracts(IsolatedAsyncioTestCase):
    async def test_real_rpc_repairs_fk_in_same_call_and_never_publishes(self):
        import psycopg2
        import uuid
        import re
        from commulingo_pipeline import service
        target='efficiency-'+uuid.uuid4().hex[:10]
        with psycopg2.connect(host='127.0.0.1',port=55439,user='postgres',dbname='commulingo_integrity_test') as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO commulingo_term_categories(id,label_ko,label_en) VALUES ('theory','이론','Theory') ON CONFLICT DO NOTHING")
        source=snapshot('https://example.org/efficiency','This archive documents the definition and its historical period.')
        claims=[{'field':field,'claim':'Documented '+field,'source_id':source['id'],'start':0,'end':len(source['body'])}
                for field in ('definition','period')]
        store=Mock()
        store.sources.return_value={source['id']:source}
        fields={'term':{'ko':'검증 용어 '+target,'en':'Test concept '+target},
            'definition':{'ko':['문헌에 근거한 개념이다.'],'en':['A concept supported by the document.']},
            'period':{'ko':'역사적 개념','en':'Historical concept'},'category':'theory',
            'aliases':{'ko':[],'en':[]},'people':['missing-person-'+target]}
        async def model(**kw):
            with self.assertRaises(ValueError) as rejected:
                await kw['handler']({'fields':fields})
            current=re.search(r'draft_id=([a-f0-9]+)',str(rejected.exception))[1]
            await kw['handler']({'draft_id':current,'repairs':[{'op':'remove','path':'/fields/people'}]})
        with patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result=await Draft(store)({'id':99,'target':target,'kind':'term','action':'create','topic':'basics'},
                [{'stage':'research','value':{'claims':claims,'baseline':''}}],Usage(),.2)
        self.assertEqual(result.next_stage,'validate')
        self.assertNotIn('people',result.value['fields'])
        self.assertIsNone(service.call({'command':'read','target':'term','id':target}))
        with psycopg2.connect(host='127.0.0.1',port=55439,user='postgres',dbname='commulingo_integrity_test') as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT count(*) FROM commulingo_agent_suggestions WHERE target_id=%s',(target,))
                self.assertEqual(cur.fetchone()[0],0)
