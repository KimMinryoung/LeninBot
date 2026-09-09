import asyncio
from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from runtime_tools.commulingo_evidence import resolve_evidence_sources
from runtime_tools.commulingo_review_policy import review_source, resolve_review_checks, validate_decision, DECISION_TOOL
from scripts.commulingo_research_memory import ResearchMemory
from scripts.commulingo_run import RunBudget, RunFailure, submitted_edit
from scripts.commulingo_write_session import draft_id, prepare_write, repair_schema
from tool_gateway.validation import validate_tool_arguments, ToolArgumentValidationError
from tool_gateway.results import ToolRejection


class EfficiencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Unit tests exercise the local run ledger, independently of deployed flags.
        reservation = patch('commulingo_pipeline.config.legacy_reserve',return_value=None)
        reservation.start()
        self.addCleanup(reservation.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'research.sqlite3'

    def test_source_ids_are_exact_and_never_guess(self):
        evidence = [{'field': 'bio', 'source_id': 'S2', 'claim': 'fact', 'locator': 'p. 2'}]
        result = resolve_evidence_sources(evidence, ['First', 'Second — full annotation'])
        self.assertEqual(result[0]['source'], 'Second — full annotation')
        self.assertIn('source_id', evidence[0])
        for item in ({'source_id':'S0'}, {'source_id':'S3'}, {'source_id':'S2','source':'First'}):
            with self.assertRaises(ValueError): resolve_evidence_sources([item], ['First','Second'])

    async def test_person_boundary_resolves_ids_after_prose_normalization(self):
        from runtime_tools import commulingo_people as people
        citation = r'Archive\nreference — full description'
        async def inline(call,*args,**kwargs): return call(*args,**kwargs)
        with patch.object(people,'_run_edit',return_value='OK — approved: Logged as edit #12.') as write, \
             patch('asyncio.to_thread',side_effect=inline):
            result = await people._exec_commulingo_person_update('p',{
                'expectedRevision':'v1', 'bio':{'ko':'소개','en':'Biography'},
                'evidence':[{'field':'bio','source_id':'S1','claim':'Biography','locator':'p. 2'}]},[citation])
        self.assertTrue(result.startswith('OK'))
        submitted = write.call_args.args[3]
        self.assertEqual(submitted['evidence'][0]['source'],citation)
        self.assertNotIn('source_id',submitted['evidence'][0])

    def test_review_ranges_only_select_original_text(self):
        body = 'The original archive records the birth and the subsequent appointment.\nAnother paragraph.'
        url = 'https://archive.example/person'
        snapshots = {}
        source_id, display = review_source(url, body, snapshots)
        proposal = {'source_refs':[url+' — biography'], 'risks':[]}
        decision = {'decision':'approve','reason':'Original evidence substantiates the proposed facts.',
            'resolved_risks':[], 'checks':[{'citation_id':'S1','source_id':source_id,
                'line_start':1,'line_end':1,'finding':'The appointment is documented.'}]}
        validate_tool_arguments('commulingo_review_decision', decision,
                                schema=DECISION_TOOL['input_schema'], risk_class='state')
        resolved = resolve_review_checks(decision, proposal, snapshots)
        self.assertEqual(resolved['checks'][0]['quote'], body.splitlines(keepends=True)[0])
        self.assertEqual(validate_decision(resolved, proposal, {url:body}), resolved)
        self.assertIn('1: The original', display)
        with self.assertRaises(ValueError): resolve_review_checks(decision, proposal, {})
        with self.assertRaises(ValueError): validate_decision(resolved, proposal, {})
        invalid = deepcopy(decision); invalid['checks'][0]['line_end']=100
        with self.assertRaises(ValueError): resolve_review_checks(invalid, proposal, snapshots)
        # Numbering and chunk boundaries do not become source evidence.
        _, _ = review_source(url, 'x'*500, snapshots)

    def test_repair_revalidates_full_payload_and_binds_revision(self):
        schema = {'type':'object','additionalProperties':False,'properties':{
            'person_id':{'type':'string'}, 'fields':{'type':'object','additionalProperties':False,
                'properties':{'bio':{'type':'string','maxLength':5},'expectedRevision':{'type':'string'}},
                'required':['bio','expectedRevision']}},'required':['person_id','fields']}
        name = 'commulingo_person_update'
        draft = {'tool':name,'args':{'person_id':'p','fields':{'bio':'too long','expectedRevision':'v1'}}}
        args = {'draft_id':draft_id(draft),'repairs':[{'op':'set','path':'/fields/bio','value':'short'}]}
        validate_tool_arguments(name,args,schema=repair_schema(schema),risk_class='write')
        self.assertEqual(prepare_write(name,args,draft,schema)['fields']['bio'],'short')
        self.assertEqual(draft['args']['fields']['bio'],'too long')
        for path,value in [('/fields/expectedRevision','v2'),('/person_id','other'),('/fields',{'bio':'short','expectedRevision':'v2'}),('/fields/unknown',1)]:
            with self.assertRaises(ToolRejection): prepare_write(name,
                {'draft_id':draft_id(draft),'repairs':[{'op':'set','path':path,'value':value}]},draft,schema)
        baseline = {'id':'p','revision':'v1'}
        ordinary = {'person_id':'p','fields':{'bio':'short'}}
        validate_tool_arguments(name,ordinary,schema=repair_schema(schema,baseline),risk_class='write')
        self.assertEqual(prepare_write(name,ordinary,None,schema,baseline)['fields']['expectedRevision'],'v1')

    async def test_raw_pages_shared_but_drafts_and_searches_isolated(self):
        first = ResearchMemory('person:p:bio:v1',path=self.path)
        second = ResearchMemory('event:e:v2',path=self.path)
        provider = AsyncMock(return_value='Original source text')
        await first.wrap({'fetch_url':provider})['fetch_url'](url='https://archive.example/source')
        await second.wrap({'fetch_url':provider})['fetch_url'](url='https://archive.example/source')
        self.assertEqual(provider.await_count,1)
        first.rejected('commulingo_person_update', {'fields':{'bio':'bad'}}, 'length')
        self.assertIsNone(second._draft())
        self.assertIn('Original source text', second.context())

    async def test_gateway_repair_executes_only_revalidated_full_write(self):
        from tool_gateway.dispatcher import execute_tool
        memory = ResearchMemory('repair:p',path=self.path)
        name = 'commulingo_person_update'
        schema = {'type':'object','additionalProperties':False,'properties':{
            'person_id':{'type':'string'},'fields':{'type':'object','additionalProperties':False,
                'properties':{'bio':{'type':'string','maxLength':5}},'required':['bio']}},
            'required':['person_id','fields']}
        writer = AsyncMock(return_value='OK — approved: Logged as edit #12.')
        async def chat(messages, **kwargs):
            visible = kwargs['tools'][0]['input_schema']
            handlers = kwargs['tool_handlers']
            _, failed = await execute_tool(name,{'person_id':'p','fields':{'bio':'too long'}},handlers,tool_schema=visible)
            self.assertTrue(failed)
            writer.assert_not_awaited()
            draft = memory._draft()
            result, failed = await execute_tool(name,{'draft_id':draft_id(draft),
                'repairs':[{'op':'set','path':'/fields/bio','value':'short'}]},handlers,tool_schema=visible)
            self.assertFalse(failed,result)
            return result
        decision = SimpleNamespace(denied=False,risk_class='write')
        with patch('tool_gateway.security.get_caller',return_value=SimpleNamespace()), \
             patch('tool_gateway.security.authorize',return_value=decision), patch('tool_gateway.security.audit'):
            await memory.chat(chat,[],tools=[{'name':name,'input_schema':schema}],tool_handlers={name:writer})
        writer.assert_awaited_once_with(person_id='p',fields={'bio':'short'})
        self.assertIsNone(memory._draft())
        self.assertEqual(memory.metrics['write_rejections'],1)

    def test_budget_counts_failed_attempts_and_durable_outcomes(self):
        policy = SimpleNamespace(max_rounds=10,budget_usd=.1)
        run = RunBudget(policy,self.path,'enrich','p')
        run.account({'rounds_used':6,'total_cost':.06})
        _, rounds, cost = run.remaining()
        self.assertEqual(rounds,4); self.assertAlmostEqual(cost,.04)
        run.account({'rounds_used':4,'total_cost':.04})
        with self.assertRaises(RunFailure):run.remaining()
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT status FROM runs').fetchone()[0],'exhausted')

    def test_health_links_recorded_review_cost_to_submission(self):
        from scripts.commulingo_lane_health import execution_metrics
        policy=SimpleNamespace(max_rounds=10,budget_usd=.1)
        author=RunBudget(policy,self.path,'enrich','p')
        author.account({'total_cost':.03,'rounds_used':4})
        author.record('pending_review',writes=[{'result':'OK — pending: Logged as edit #12.'}],
                      metrics={'write_calls':1,'write_rejections':0})
        reviewer=RunBudget(policy,self.path,'review','12')
        reviewer.account({'total_cost':.02,'rounds_used':3})
        reviewer.record('approved')
        report='\n'.join(execution_metrics('-24h',self.path))
        self.assertIn('$0.0500/edit',report)
        self.assertIn('first-write 1/1',report)

    def test_submission_receipt_not_latest_lane_write(self):
        calls=[]
        def query(sql,params):
            calls.append(params)
            return {'id':12,'target_id':'p','status':'approved'}
        self.assertEqual(submitted_edit([{'result':'OK — approved: Logged as edit #12.','target':'p'}],query)['id'],12)
        self.assertEqual(calls,[{'id':12}])
        with self.assertRaises(RuntimeError): submitted_edit([{'result':'OK — approved: Logged as edit #12.','target':'other'}],query)

    async def test_stage_exception_keeps_previous_attempt_cost(self):
        from scripts import commulingo_people_maintainer as maintainer
        calls=[]
        async def chat(messages, **kwargs):
            calls.append(kwargs)
            kwargs['budget_tracker'].update(total_cost=.06 if len(calls)==1 else .01,
                                             rounds_used=6 if len(calls)==1 else 1)
            if len(calls)==1: return 'unfinished'
            self.assertAlmostEqual(kwargs['budget_usd'],.04)
            self.assertLessEqual(kwargs['max_rounds']+kwargs['max_length_continuations'],4)
            raise RuntimeError('provider disconnected')
        policy=SimpleNamespace(max_rounds=10,budget_usd=.1,max_output_continuations=1,
            max_output_tokens=1000,max_input_tokens=10000)
        binding=SimpleNamespace(chat=chat,client=None,model='fake',render_provider='test',reasoning={})
        with patch.object(maintainer,'resolve_agent_tool_loop',return_value=binding), \
             patch('scripts.commulingo_research_memory.STORE_PATH',self.path):
            with self.assertRaises(RunFailure) as failed:
                await maintainer._call_curator_stage(task='test',spec=SimpleNamespace(name='test',render_prompt=lambda **kw:'test'),
                    tools=[],handlers={},policy=policy,stage='test',expect_edit=True,before_count=0,
                    finalization_tools=[],terminal_tools=[])
        self.assertAlmostEqual(failed.exception.summary['cost_usd'],.07)
        self.assertEqual(failed.exception.summary['rounds_used'],7)
        self.assertEqual(failed.exception.summary['status'],'error')

    async def test_common_llm_loop_keeps_cost_on_cancellation(self):
        from test_claude_loop_rounds import FakeClient, _response, _tool_use_block
        from llm.claude_loop import chat_with_tools
        async def cancelled(*args, **kwargs): raise asyncio.CancelledError()
        tracker={}
        with patch('llm.claude_loop.execute_tools_batch',side_effect=cancelled), patch('llm.agent_loop.record_llm_call'):
            with self.assertRaises(asyncio.CancelledError):
                await chat_with_tools([{'role':'user','content':'test'}],
                    client=FakeClient([_response([_tool_use_block('t1','echo')],stop_reason='tool_use')]),
                    model='deepseek-v4-pro', tools=[{'name':'echo','description':'echo','input_schema':{'type':'object','properties':{}}}],
                    tool_handlers={},system_prompt='Test',budget_tracker=tracker,budget_usd=.1,max_rounds=2)
        self.assertGreater(tracker['total_cost'],0)
        self.assertEqual(tracker['rounds_used'],1)


if __name__ == '__main__': unittest.main()
