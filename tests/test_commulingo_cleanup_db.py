"""Opt-in cleanup SQL/concurrency checks on a disposable PostgreSQL database."""
from contextlib import contextmanager
from pathlib import Path
import os
import unittest
import uuid

from psycopg2.errors import LockNotAvailable
from commulingo_pipeline.cleanup import retire
from commulingo_pipeline.store import Store


@unittest.skipUnless(os.getenv('COMMULINGO_CLEANUP_TEST_PORT'), 'disposable cleanup PostgreSQL required')
class CleanupDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg2
        cls.schema = 'cleanup_' + uuid.uuid4().hex
        @contextmanager
        def connect():
            conn = psycopg2.connect(host='127.0.0.1', port=int(os.environ['COMMULINGO_CLEANUP_TEST_PORT']),
                                    user='postgres', dbname='commulingo_integrity_test')
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute('SET search_path TO ' + cls.schema)
                    yield conn
            finally:
                conn.close()
        cls.store = Store(connect)
        with cls.store.transaction() as cur:
            cur.execute('CREATE SCHEMA ' + cls.schema)
            cur.execute(Path('commulingo_pipeline/schema.sql').read_text())
            cur.execute('''CREATE TABLE commulingo_people(id text PRIMARY KEY,years_label text,
                epithet_ko text,epithet_en text,bio_ko text,bio_en text,moment_ko text,moment_en text,
                citizenship_label_ko text,citizenship_label_en text,origin_label_ko text,origin_label_en text);
                CREATE TABLE commulingo_person_roles(person_id text);
                CREATE TABLE commulingo_person_career_entries(person_id text);
                CREATE TABLE commulingo_terms(id text PRIMARY KEY,definition_ko text,definition_en text,body_ko text,body_en text)''')

    @classmethod
    def tearDownClass(cls):
        with cls.store.transaction() as cur:
            cur.execute('DROP SCHEMA ' + cls.schema + ' CASCADE')

    def setUp(self):
        with self.store.transaction() as cur:
            cur.execute('''TRUNCATE commulingo_pipeline_jobs,commulingo_pipeline_attempts,
                commulingo_pipeline_artifacts,commulingo_pipeline_budget,commulingo_pipeline_job_sources,
                commulingo_pipeline_publications,commulingo_pipeline_sources,commulingo_pipeline_fetch_cache,
                commulingo_terms,commulingo_people,commulingo_person_roles,commulingo_person_career_entries CASCADE''')

    def job(self, target, *, body='Body', payload=None):
        with self.store.transaction() as cur:
            cur.execute('INSERT INTO commulingo_terms VALUES (%s,\'정의\',\'Definition\',\'본문\',%s)',(target,body))
        return self.store.enqueue(kind='term',action='update',target=target,topic='enrichment',
            reason='Bundled enrichment: definition, history',
            payload=payload if payload is not None else {'topics':['definition','history'],'gap_ids':[]})

    def statuses(self):
        with self.store.transaction() as cur:
            cur.execute('SELECT id,status FROM commulingo_pipeline_jobs')
            return {r['id']:r['status'] for r in cur.fetchall()}

    def test_preview_apply_and_every_history_exclusion(self):
        empty = self.job('empty')
        protected = [self.job('missing',body=''), self.job('gap',payload={'gap_ids':[7]}),
                     self.job('explicit',payload={'commissions':[{'reason':'Write a requested section'}]})]
        for kind in ('attempt','budget','source','artifact','running'):
            job = self.job(kind)
            protected.append(job)
            with self.store.transaction() as cur:
                if kind=='attempt':
                    cur.execute("INSERT INTO commulingo_pipeline_attempts(id,job_id,stage) VALUES (%s,%s,'research')",(str(uuid.uuid4()),job))
                elif kind=='budget':
                    cur.execute("INSERT INTO commulingo_pipeline_budget(id,job_id,lane,reserved) VALUES (%s,%s,'term',0)",(str(uuid.uuid4()),job))
                elif kind=='artifact':
                    cur.execute("INSERT INTO commulingo_pipeline_artifacts(job_id,stage,value) VALUES (%s,'editor_checkpoint','{}')",(job,))
                elif kind=='source':
                    cur.execute("INSERT INTO commulingo_pipeline_sources VALUES ('s','https://example.org','h',now(),now(),'body')")
                    cur.execute("INSERT INTO commulingo_pipeline_job_sources VALUES (%s,'s')",(job,))
                else:
                    cur.execute("UPDATE commulingo_pipeline_jobs SET status='running' WHERE id=%s",(job,))
        before = self.statuses()
        self.assertEqual([r['id'] for r in retire(self.store)['candidates']], [empty])
        self.assertEqual(before, self.statuses())
        self.assertEqual(retire(self.store,apply=True)['retired'], 1)
        after = self.statuses()
        self.assertEqual(after[empty], 'cancelled')
        for job in protected:
            self.assertEqual(after[job], before[job])
        self.assertEqual(retire(self.store,apply=True)['retired'], 0)

    def test_changes_after_preview_and_concurrent_writes_are_preserved(self):
        job = self.job('changed')
        self.assertEqual(len(retire(self.store)['candidates']), 1)
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_terms SET body_en='' WHERE id='changed'")
            with self.assertRaises(LockNotAvailable):
                retire(self.store,apply=True)
        self.assertEqual(retire(self.store,apply=True)['retired'], 0)
        self.assertEqual(self.statuses()[job], 'ready')

    def test_scan_rotates_and_checks_both_person_languages(self):
        needed = self.job('needs-work',body='')
        obsolete = self.job('obsolete')
        self.assertEqual(retire(self.store,apply=True,limit=1)['retired'],0)
        self.assertEqual(retire(self.store,apply=True,limit=1)['retired'],1)
        self.assertEqual(self.statuses()[needed],'ready')
        self.assertEqual(self.statuses()[obsolete],'cancelled')
        with self.store.transaction() as cur:
            cur.execute("INSERT INTO commulingo_people(id,bio_ko,bio_en) VALUES ('person','본문','')")
        person = self.store.enqueue(kind='person',action='update',target='person',topic='bio',reason='Commissioned bio')
        retire(self.store,apply=True)
        self.assertEqual(self.statuses()[person],'ready')
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_people SET bio_en='Body'")
        retire(self.store,apply=True)
        self.assertEqual(self.statuses()[person],'cancelled')

    def test_reconcile_only_explicit_complete_costs_and_keep_unknown_reservations(self):
        reservations = {}
        for target, metrics in (
                ('confirmed', {'cost_complete':True, 'actual_cost_usd':.013,
                    'input_tokens':1234, 'output_tokens':56, 'cache_read_tokens':1000,
                    'review_context_original_chars':2000, 'review_context_chars':1000}),
                ('partial', {'observed_llm_cost_usd':.01, 'input_tokens':500}),
                ('old', {})):
            job_id = self.job(target)
            job = {'id':job_id,'stage':'research'}
            attempt = self.store.start_attempt(job)
            reservation = self.store.reserve('.2',lane='term',job_id=job_id,cap='2')
            reservations[target] = str(reservation)
            self.store.link_attempt_budget(attempt,reservation)
            self.store.finish_attempt(attempt,'error',None,'fixture',1,metrics)
        preview = self.store.reconcile_costs()
        self.assertEqual(len(preview['candidates']),1)
        self.assertEqual(len(preview['unresolved_jobs']),2)
        self.store.reconcile_costs(apply=True)
        with self.store.transaction() as cur:
            cur.execute('SELECT id,actual FROM commulingo_pipeline_budget')
            actual = {str(r['id']):r['actual'] for r in cur.fetchall()}
        self.assertEqual(float(actual[reservations['confirmed']]),.013)
        self.assertIsNone(actual[reservations['partial']])
        self.assertIsNone(actual[reservations['old']])
        self.assertEqual(self.store.reconcile_costs(apply=True)['candidates'],[])
        from datetime import datetime, timezone
        row = self.store.efficiency(datetime(2000,1,1,tzinfo=timezone.utc))[0]
        self.assertEqual(row['input_tokens'],1734)
        self.assertEqual(row['token_measured'],2)
        self.assertEqual(row['review_original_chars'],2000)
        self.assertEqual(row['review_chars'],1000)

    def test_failure_checkpoint_is_fenced_and_cannot_overwrite_editor_checkpoint(self):
        from commulingo_pipeline.store import LostLease
        job_id = self.job('checkpoint')
        job = self.store.claim(job_id=job_id)
        self.store.save_editor_checkpoint(job,{'draft':{'fields':{'body':'retained'}}})
        self.store.save_fetch_failures(job,{'failures':{'url-hash':{'reason':'http_forbidden','until':9999999999}}})
        detail = self.store.detail(job_id)
        self.assertEqual([a['stage'] for a in detail['artifacts']],['editor_checkpoint','fetch_failures'])
        with self.store.transaction() as cur:
            cur.execute("UPDATE commulingo_pipeline_jobs SET lease_until=now()-interval '1 second' WHERE id=%s",(job_id,))
        with self.assertRaises(LostLease):
            self.store.save_fetch_failures(job,{'failures':{}})
        self.assertEqual(len(self.store.detail(job_id)['artifacts']),2)
