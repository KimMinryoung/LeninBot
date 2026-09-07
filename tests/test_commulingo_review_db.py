"""Opt-in: real queue leases + worker decisions through the isolated JS service."""
import asyncio
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch,AsyncMock
import uuid

if os.getenv('COMMULINGO_FRONTEND_CONTAINER')!='commulingo-python-rpc' or os.getenv('COMMULINGO_REVIEW_TEST_PORT')!='55439':
    raise unittest.SkipTest('isolated review DB/RPC required')
sys.path.insert(0,'/home/grass/leninbot')
ROOT=Path(os.environ.get('COMMULINGO_REVIEW_SOURCE',Path(__file__).resolve().parents[1]))
import runtime_tools
runtime_tools.__path__.insert(0,str(ROOT/'runtime_tools'))
from runtime_tools import commulingo_review_queue as queue
from runtime_tools.commulingo_person_service import call_person_service as rpc
import psycopg2
spec=importlib.util.spec_from_file_location('review_db_worker',ROOT/'scripts/commulingo_person_reviewer.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

@contextmanager
def connect():
    conn=psycopg2.connect(host='127.0.0.1',port=55439,dbname='commulingo_integrity_test',user='postgres',password='isolated-review-test')
    try:
        with conn:yield conn
    finally:conn.close()

class ReviewDatabase(unittest.TestCase):
    def setUp(self):
        self.patch=patch.object(queue,'get_conn',connect);self.patch.start();self.addCleanup(self.patch.stop)
        queue.query("INSERT INTO commulingo_people_groups(id,title_ko,title_en) VALUES ('review-test','검증','Review') ON CONFLICT DO NOTHING")
        suffix=uuid.uuid4().hex[:8];self.id='review-'+suffix;self.source='https://archive.example/'+suffix
        rpc({'command':'submit','target':'person','action':'create','id':self.id,'sources':[self.source],
            'fields':{'name':{'ko':'검증 인물 '+suffix,'en':'Review Person '+suffix},'groupId':'review-test','role':{'icon':'book-open'}}})
    def propose(self):
        current=rpc({'command':'read','id':self.id})
        result=rpc({'command':'submit','target':'person','action':'update','id':self.id,'sources':[self.source],
            'fields':{'expectedRevision':current['revision'],'epithet':{'ko':'검토한 수정'},'reviewFlags':['identity_uncertain']}})
        self.assertEqual(result['status'],'pending');queue.synchronize();return result['suggestionId']
    def decide(self,kind='approve'):
        quote='An independent record establishes the identity and the stated role of this person.'
        return {'decision':kind,'reason':'독립 기록에서 해당 인물의 신원과 직책을 대조하여 확인했습니다.',
            'resolved_risks':['identity_uncertain'], 'checks':[] if kind=='escalate' else [{'citation':self.source,'source':self.source,'quote':quote,'finding':'신원 확인'}]}, {self.source:quote}
    def test_concurrent_claim_has_one_owner_and_approval_is_atomic(self):
        sid=self.propose()
        with ThreadPoolExecutor(2) as pool:claimed=list(pool.map(lambda _:queue.claim(),range(2)))
        jobs=[j for j in claimed if j and str(j['suggestion_id'])==str(sid)];self.assertEqual(len(jobs),1)
        with patch.object(worker,'research',new=AsyncMock(return_value=self.decide())):
            asyncio.run(worker.process(jobs[0],{}))
        self.assertEqual(queue.suggestion(sid)['status'],'approved')
        self.assertEqual(queue.detail(sid)['review_job']['status'],'approved')
        self.assertEqual(rpc({'command':'read','id':self.id})['epithet']['ko'],'검토한 수정')
        asyncio.run(worker.process(jobs[0],{}))
        self.assertEqual(queue.suggestion(sid)['status'],'approved')
    def test_escalation_notification_delivery_retry_and_owner_retry(self):
        sid=self.propose();job=queue.claim()
        with patch.object(worker,'research',new=AsyncMock(return_value=self.decide('escalate'))):asyncio.run(worker.process(job,{}))
        self.assertEqual(queue.suggestion(sid)['status'],'pending')
        with patch.object(worker,'notify_owner',return_value=False):worker.deliver_notifications()
        self.assertIsNone(queue.detail(sid)['review_job']['notified_at'])
        queue.query("UPDATE commulingo_person_review_jobs SET notification_after=NOW() WHERE suggestion_id=%s",(sid,))
        with patch.object(worker,'notify_owner',return_value=True) as send:
            worker.deliver_notifications();worker.deliver_notifications();self.assertEqual(send.call_count,1)
        self.assertIsNotNone(queue.detail(sid)['review_job']['notified_at'])
        self.assertTrue(queue.retry(sid))
        job=queue.claim();queue.finish(job,'escalated','test finished')
    def test_stale_proposal_rejected_without_llm_and_manual_race_preserved(self):
        sid=self.propose();job=queue.claim();current=rpc({'command':'read','id':self.id})
        rpc({'command':'submit','target':'person','action':'update','id':self.id,'sources':[self.source],
             'fields':{'expectedRevision':current['revision'],'epithet':{'ko':'더 최신 수정'}}})
        with patch.object(worker,'research',new=AsyncMock()) as llm:
            asyncio.run(worker.process(job,{}));llm.assert_not_called()
        self.assertEqual(queue.suggestion(sid)['status'],'rejected')
        self.assertEqual(rpc({'command':'read','id':self.id})['epithet']['ko'],'더 최신 수정')
    def test_operator_rejection_during_research_wins_without_overwrite(self):
        sid=self.propose();job=queue.claim()
        async def research(*args):
            rpc({'command':'review','suggestionId':sid,'approve':False,'note':'소유자가 반려함','changedBy':'test-owner'})
            return self.decide()
        with patch.object(worker,'research',new=research):asyncio.run(worker.process(job,{}))
        self.assertEqual(queue.suggestion(sid)['status'],'rejected')
        self.assertEqual(queue.detail(sid)['review_job']['status'],'rejected')
        self.assertNotEqual(rpc({'command':'read','id':self.id})['epithet']['ko'],'검토한 수정')
    def test_third_runtime_failure_escalates_and_notifies(self):
        sid=self.propose()
        queue.query('UPDATE commulingo_person_review_jobs SET attempts=2 WHERE suggestion_id=%s',(sid,))
        with patch.object(worker,'research',new=AsyncMock(side_effect=RuntimeError('test research outage'))),patch.object(worker,'notify_owner',return_value=True):
            result=asyncio.run(worker.run(skip_budget=True))
        self.assertEqual(result['status'],'escalated')
        self.assertIsNotNone(queue.detail(sid)['review_job']['notified_at'])
    def test_expired_lease_recovered_and_exhaustion_escalates(self):
        sid=self.propose();old=queue.claim()
        queue.query("UPDATE commulingo_person_review_jobs SET lease_until=NOW()-INTERVAL '1 minute' WHERE suggestion_id=%s",(sid,))
        queue.synchronize();new=queue.claim();self.assertNotEqual(old['lease_token'],new['lease_token'])
        self.assertFalse(queue.owned(old))
        self.assertIsNone(queue.save_decision(old,*self.decide()))
        queue.query("UPDATE commulingo_person_review_jobs SET attempts=3,lease_until=NOW()-INTERVAL '1 minute' WHERE suggestion_id=%s",(sid,))
        queue.synchronize();self.assertEqual(queue.detail(sid)['review_job']['status'],'escalated')

if __name__=='__main__':unittest.main()
