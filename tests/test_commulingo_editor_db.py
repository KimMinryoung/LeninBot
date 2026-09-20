"""Opt-in end-to-end editor tests: isolated Postgres + private frontend RPC, no LLM/network."""
import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from commulingo_pipeline.engine import Engine
from commulingo_pipeline.planner import Planner
from commulingo_pipeline.store import Store, LostLease
from commulingo_pipeline.stages import stages

PORT = os.getenv('COMMULINGO_PIPELINE_TEST_PORT')
FRONTEND = os.getenv('COMMULINGO_EDITOR_FRONTEND')


@unittest.skipUnless(PORT and FRONTEND, 'isolated PostgreSQL and staged frontend opt-in')
class EditorDatabaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import psycopg2
        self.frontend = Path(FRONTEND).resolve()
        if not self.frontend.is_relative_to('/tmp'):
            raise RuntimeError('frontend must be an isolated /tmp worktree')
        @contextmanager
        def connect():
            conn = psycopg2.connect(host='127.0.0.1',port=int(PORT),user='postgres',dbname='commulingo_integrity_test')
            try:
                with conn:
                    yield conn
            finally:
                conn.close()
        self.store = Store(connect)
        self.target = 'editor-e2e-'+uuid.uuid4().hex[:12]
        self.job_id = None
        with self.store.transaction() as cur:
            cur.execute(Path('commulingo_pipeline/schema.sql').read_text())
            cur.execute("INSERT INTO commulingo_term_categories(id,label_ko,label_en) VALUES ('editor-test','테스트','Test') ON CONFLICT DO NOTHING")
            cur.execute("INSERT INTO commulingo_terms(id,term_ko,term_en,category,definition_ko,definition_en,period_ko,period_en) VALUES (%s,%s,%s,'editor-test','기존 정의','Existing definition','역사','History')",(self.target,'검증 '+self.target,'Test '+self.target))

    def tearDown(self):
        with self.store.transaction() as cur:
            if self.job_id:
                for table in ('attempts','artifacts','job_sources','publications','budget'):
                    cur.execute(f'DELETE FROM commulingo_pipeline_{table} WHERE job_id=%s',(self.job_id,))
                cur.execute('DELETE FROM commulingo_pipeline_jobs WHERE id=%s',(self.job_id,))
                cur.execute('DELETE FROM commulingo_editorial_receipts WHERE key LIKE %s',(f'pipeline:{self.job_id}:%',))
            cur.execute('DELETE FROM commulingo_terms WHERE id=%s',(self.target,))
            cur.execute('DELETE FROM commulingo_agent_suggestions WHERE target_id=%s',(self.target,))
            cur.execute('DELETE FROM commulingo_people_revisions WHERE entity_id=%s',(self.target,))

    def rpc(self, request):
        env = {**os.environ,'COMMULINGO_ISOLATED_TEST':'1','DB_HOST':'127.0.0.1','DB_PORT':PORT,
               'DB_NAME':'commulingo_integrity_test','DB_USER':'postgres','DB_PASSWORD':'','DB_SSL':'false'}
        process = subprocess.run(['node','scripts/commulingo-pipeline-service.js'],cwd=self.frontend,
                                 input=json.dumps(request),text=True,capture_output=True,env=env,timeout=30)
        value = json.loads(process.stdout)
        if not value.get('ok'):
            raise ValueError(f"{value.get('code')}: {value.get('error')}")
        return value['result']

    def test_concrete_planner_sql_omits_quotas(self):
        with patch('commulingo_pipeline.planner.report_mentions_by_term',return_value={}):
            candidates = Planner(self.store,overlap_allow=[],exclude=[],concrete=True).candidates()
        candidate = next(c for c in candidates if c['target']==self.target)
        self.assertEqual(set(candidate['payload']['topics']),{'definition','history'})

    def test_checkpoint_is_fenced_and_survives_worker_reclaim(self):
        self.job_id = self.store.enqueue(kind='term',action='update',target=self.target,topic='history',reason='Fixture')
        job = self.store.claim(job_id=self.job_id)
        self.store.save_editor_checkpoint(job,{'baseline':'r1','draft':{'fields':{'body':{'ko':'보존'}}}})
        self.store.defer(job,'provider interrupted',seconds=0,failed=False)
        reclaimed = self.store.claim(job_id=self.job_id)
        self.assertNotEqual(job['lease_token'],reclaimed['lease_token'])
        with self.assertRaises(LostLease):
            self.store.save_editor_checkpoint(job,{'draft':'stale result'})
        self.assertEqual(self.store.detail(self.job_id)['artifacts'][0]['value']['draft']['fields']['body']['ko'],'보존')

    async def test_author_review_atomic_publish_and_legacy_tick_resume(self):
        self.job_id = self.store.enqueue(kind='term',action='update',target=self.target,topic='history',
                                         reason='Commissioned glossary explanation: history')
        source = 'https://example.org/editor-fixture'
        text = 'The archive describes the documented historical context of this concept.'
        async def model(**kwargs):
            usage = kwargs['usage']; usage.started=True; usage.complete=True
            usage.tracker['total_cost']=.01
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{text}\n</external>'))(url=source)
            if kwargs['tool']['name']=='commulingo_pipeline_result':
                await kwargs['handler']({'status':'ready','reason':'The archive supports the missing historical context.',
                    'fields':{'body':{'ko':'원문으로 확인한 역사적 맥락이다.','en':'Historical context verified against the archive.'}},
                    'claims':[{'field':'body','claim':'The historical context is documented.','passages':['P1']}],
                    'issue_results':[{'id':'missing:body','status':'resolved','reason':'Added both languages with original evidence.'}]})
            else:
                await kwargs['handler']({'decision':'approve','reason':'Independent original text supports the changed historical explanation.',
                    'resolved_risks':[], 'checks':[{'citation_id':'S1','passages':['P1'],'finding':'The original confirms the explanation.'}],
                    'required_corrections':[], 'optional_suggestions':[]})
        from commulingo_pipeline.stages import READS
        reads = {name:AsyncMock(return_value=f'<external source="web">\n{text}\n</external>') for name in READS}
        with patch('commulingo_pipeline.service.call',side_effect=self.rpc), \
             patch('runtime_tools.registry.TOOL_HANDLERS',reads), \
             patch('commulingo_pipeline.stages.model_call',side_effect=model), \
             patch('commulingo_pipeline.config.load',return_value={'phase':'live'}):
            first = await Engine(self.store,stages(self.store,workflow='editor'),cap=1000).run_one(job_id=self.job_id,draft_only=False)
            self.assertEqual(first.get('stage'),'review',first)
            self.assertEqual(self.store.detail(self.job_id)['job']['payload']['workflow'],'editor')
            # A later timer still configured legacy must honor this job's pinned workflow.
            remaining = await Engine(self.store,stages(self.store,workflow='legacy'),cap=1000).run_batch(job_id=self.job_id,draft_only=False,limit=2)
        self.assertEqual([r['stage'] for r in remaining],['submit','complete'],remaining)
        self.assertEqual(self.store.detail(self.job_id)['job']['status'],'complete')
        current = self.rpc({'command':'read','target':'term','id':self.target})
        self.assertEqual(current['body']['ko'],'원문으로 확인한 역사적 맥락이다.')
        self.assertEqual(current['evidence'][0]['excerpt'],text)
        metrics = self.store.efficiency(datetime.now(timezone.utc)-timedelta(hours=1))
        term = next(m for m in metrics if m['kind']=='term' and m['action']=='update')
        self.assertGreaterEqual(term['resolved_issues'],1)
