import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from commulingo_pipeline.evidence import snapshot, compile_evidence
from commulingo_pipeline.engine import Engine, Result
from commulingo_pipeline.store import Store, LostLease, BudgetUnavailable


class EvidenceTests(unittest.TestCase):
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


class EngineTests(unittest.IsolatedAsyncioTestCase):
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
            await kwargs['handler']({'fields':{'definition':{'ko':'검증한 정의','en':'Verified definition'}}})
        with patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result = await Draft(store)({'kind':'term','action':'update','topic':'definition'},
                [{'stage':'research','value':{'claims':[claim],'baseline':'original-revision'}}],Usage(),.2)
        self.assertEqual(result.value['fields']['expectedRevision'],'original-revision')
        self.assertEqual(result.value['fields']['evidence'][0]['excerpt'],source['body'])
        self.assertEqual(result.next_stage,'validate')

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

    def setUp(self):
        with self.store.transaction() as cur:
            cur.execute('TRUNCATE commulingo_pipeline_jobs,commulingo_pipeline_budget,commulingo_pipeline_artifacts,commulingo_pipeline_job_sources,commulingo_pipeline_publications RESTART IDENTITY')

    def add(self,target='test'):
        return self.store.enqueue(kind='person',action='update',target=target,topic='bio',reason='test')

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
                (slug text PRIMARY KEY,title text,markdown text,status text)''')
            cur.execute("INSERT INTO research_documents VALUES ('pipeline-fixture','Fixture','A concept appears in this public document.','public') ON CONFLICT DO NOTHING")
        planner = Planner(self.store)
        plan = planner.plan()
        material = next(m for m in plan['materials'] if m['material_id']=='report:pipeline-fixture')
        with self.store.transaction() as cur:
            cur.execute('''INSERT INTO commulingo_pipeline_materials(material_id,content_hash)
                VALUES (%s,%s) ON CONFLICT(material_id) DO UPDATE SET content_hash=EXCLUDED.content_hash''',
                (material['material_id'],material['content_hash']))
        self.assertNotIn(material['material_id'],[m['material_id'] for m in planner.materials()])
        with self.store.transaction() as cur:
            cur.execute("DELETE FROM research_documents WHERE slug='pipeline-fixture'")
            cur.execute("DELETE FROM commulingo_pipeline_materials WHERE material_id='report:pipeline-fixture'")
