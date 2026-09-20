import asyncio
from copy import deepcopy
import json
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

from commulingo_test_support import HermeticAsyncCase
from commulingo_pipeline.engine import Engine, Result, Usage
from commulingo_pipeline.fetch_backoff import FetchBackoff
from commulingo_pipeline.review_context import context, context_tool
from commulingo_pipeline.store import BudgetUnavailable
from commulingo_pipeline import workflow
from tool_gateway.results import ToolFailure

JOB = {'id': 42, 'kind':'term', 'action':'update', 'topic':'history',
       'target':'fixture', 'reason':'Commissioned glossary explanation: history',
       'payload':{'workflow':'editor'}, 'stage':'research', 'attempts':1}
CURRENT = {'id':'fixture', 'revision':'r1','body':{'ko':'본문','en':'Body'}}


class ContextTests(HermeticAsyncCase):
    def test_context_tool_is_owner_reviewer_only(self):
        from security_gateway import CallerContext, authorize, policy
        with patch.object(policy,'enforce_mode',return_value='enforce'):
            for agent,owner,allowed in [('commulingo_reviewer',True,True),
                                        ('commulingo_curator',True,False),
                                        ('commulingo_reviewer',False,False)]:
                decision = authorize(CallerContext(interface='autonomous',agent_name=agent,is_owner=owner),
                    'commulingo_pipeline_review_context',{'fields':['body']},consume_rate_limit=False)
                self.assertEqual(decision.allowed,allowed)

    async def test_patch_and_evidence_are_preserved_without_repeating_new_text(self):
        old, new = 'Old paragraph. '*200, 'New paragraph. '*200
        current = {**CURRENT, 'body':{'ko':old,'en':'Old'}, 'aliases':['unchanged alias'],
                   'notes':'Author warning', 'sections':[{'slug':'old-section','body':old}]}
        proposal = {'target_type':'term','patch_json':{'body':{'ko':new,'en':'New'},
                    'expectedRevision':'r1','evidence':[{'field':'body','excerpt':'evidence text'}]},
                    'source_refs':[{'id':'S1','url':'https://example.org'}]}
        before = deepcopy(proposal)
        compact = context(proposal,current)
        self.assertEqual(proposal,before)
        self.assertEqual(compact['suggestion'],before)
        encoded = json.dumps(compact)
        self.assertEqual(encoded.count(new),1)
        self.assertNotIn('unchanged alias',encoded)
        self.assertIn('Author warning',encoded)
        _,read,_ = context_tool(current)
        self.assertIn('unchanged alias',await read(fields=['aliases']))
        with self.assertRaises(ValueError):
            await read(fields=['invented'])
        self.assertEqual(current['body']['ko'],old)

    async def test_section_and_rereview_use_the_matching_old_values(self):
        current = {'sections':[{'slug':'target','body':{'en':'old section'}},
                               {'slug':'other','body':{'en':'other text'}}]}
        proposal = {'target_type':'person_section','patch_json':{'slug':'target','body':{'en':'new section'}}}
        compact = context(proposal,current,{'slug':'target','body':{'en':'prior proposal'}})
        self.assertEqual(compact['changes'],[{'path':'/body/en','before':'old section'}])
        self.assertEqual(compact['changes_since_previous_patch'],[{'path':'/body/en','before':'prior proposal'}])


class PreflightTests(HermeticAsyncCase):
    async def test_provider_fallback_cost_and_jev_are_counted_once(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from commulingo_pipeline.stages import model_call, result_tool
        from commulingo_pipeline.prompts import spec
        tool = result_tool({'type':'object','properties':{'reason':{'type':'string'}},'required':['reason']})
        async def refuse(*args,**kwargs):
            kwargs['budget_tracker']['observed_llm_cost_usd'] = .01
            kwargs['budget_tracker']['total_cost'] = .01
            raise RuntimeError('Content Exists Risk')
        async def accept(*args,**kwargs):
            tracker = kwargs['budget_tracker']
            tracker['observed_llm_cost_usd'] += .02
            tracker['total_cost'] = .02
            tracker['jev_cost_usd'] = .001
            await kwargs['tool_handlers'][tool['name']](reason='Source verified')
        def resolve(agent,policy):
            return SimpleNamespace(chat=refuse if agent.provider=='deepseek' else accept,
                client=None,model='fixture',render_provider=agent.provider,reasoning={})
        usage = Usage()
        with patch('bot_config.resolve_agent_tool_loop',side_effect=resolve):
            await model_call(spec=replace(spec('research'),provider='deepseek'),prompt='Fixture',tool=tool,
                handler=AsyncMock(return_value='OK'),reads=set(),usage=usage,budget=.2)
        self.assertAlmostEqual(usage.tracker['total_cost'],.031)
        self.assertTrue(usage.complete)

    def store(self, job=None):
        store = Mock()
        store.claim.return_value = deepcopy(job or JOB)
        store.detail.return_value = {'artifacts':[]}
        store.reserve.side_effect = BudgetUnavailable('daily budget reserved or spent')
        return store

    async def test_routed_no_edit_completes_even_when_budget_exhausted(self):
        store = self.store()
        legacy = AsyncMock()
        legacy.uses_llm = True
        stages = workflow.routed_stages(store, {'research':legacy}, 'editor')
        with patch('commulingo_pipeline.service.call',return_value=CURRENT) as read:
            result = await Engine(store, stages).run_one()
        self.assertEqual(result['stage'],'judge')
        read.assert_called_once()
        store.reserve.assert_not_called()
        legacy.assert_not_awaited()
        self.assertTrue(store.finish_attempt.call_args.args[-1]['preflight_no_model'])

    async def test_explicit_request_still_needs_budget(self):
        store = self.store({**JOB, 'payload':{'workflow':'editor','gap_id':7}})
        legacy = AsyncMock()
        legacy.uses_llm = True
        stages = workflow.routed_stages(store, {'research':legacy}, 'editor')
        with patch('commulingo_pipeline.service.call',return_value=CURRENT):
            result = await Engine(store, stages).run_one()
        self.assertEqual(result['status'],'budget_deferred')
        store.reserve.assert_called_once()
        store.settle.assert_not_called()
        store.finish_stage.assert_not_called()

    async def test_prepare_keeps_lease_and_timeout_protection(self):
        store = self.store()
        called = AsyncMock()
        async def prepare(*args):
            await asyncio.sleep(1)
        called.prepare = prepare
        called.uses_llm = True
        result = await Engine(store,{'research':called},timeout=.01).run_one()
        self.assertEqual(result['status'],'error')
        store.reserve.assert_not_called()
        called.assert_not_awaited()

    async def test_complete_cost_receipt_survives_settlement_failure(self):
        store = self.store()
        store.reserve.side_effect = None
        store.reserve.return_value = 'reservation'
        store.settle.side_effect = RuntimeError('DB settlement failed')
        async def stage(job,artifacts,usage,budget):
            usage.started = usage.complete = True
            usage.tracker['total_cost'] = .017
            return Result({},'review')
        stage.uses_llm = True
        with self.assertRaisesRegex(RuntimeError,'settlement failed'):
            await Engine(store,{'research':stage}).run_one()
        metrics = store.finish_attempt.call_args.args[-1]
        self.assertTrue(metrics['cost_complete'])
        self.assertEqual(metrics['actual_cost_usd'],.017)


class FetchTests(HermeticAsyncCase):
    async def test_failures_survive_retry_and_offsets_cannot_bypass_backoff(self):
        store, usage = Mock(), Usage()
        call = AsyncMock(return_value=ToolFailure('Fetch diagnosis: http_forbidden - HTTP 403'))
        with patch('commulingo_pipeline.fetch_backoff.time.time',return_value=1000):
            guard = FetchBackoff(store,JOB,usage,[])
            fetch = guard.wrap('fetch_url',call)
            first, second = await asyncio.gather(fetch(url='https://example.org'),fetch(url='https://example.org',offset=10))
            self.assertIsInstance(first,ToolFailure)
            self.assertIn('not evidence',second)
            call.assert_awaited_once()
            saved = deepcopy(store.save_fetch_failures.call_args.args[1])
            retry = FetchBackoff(store,JOB,usage,[{'stage':'fetch_failures','value':saved}])
            result = await retry.wrap('fetch_url',call)(url='https://example.org',use_cache=False)
            self.assertIn('deferred',result)
            call.assert_awaited_once()
            await fetch(url='https://another.example.org')
            self.assertEqual(call.await_count,2)
        with patch('commulingo_pipeline.fetch_backoff.time.time',return_value=3000):
            call.return_value = '<external source="web">Original text.</external>'
            result = await retry.wrap('fetch_url',call)(url='https://example.org')
            self.assertIn('Original text',result)
            self.assertEqual(call.await_count,3)

    async def test_source_text_and_offset_errors_are_not_negative_cached(self):
        store = Mock()
        guard = FetchBackoff(store,JOB,Usage(),[])
        call = AsyncMock(return_value='<external source="web">Fetch diagnosis: http_forbidden</external>')
        fetch = guard.wrap('fetch_url',call)
        await fetch(url='https://example.org')
        call.return_value = 'Error: offset is beyond the fetched content.'
        await fetch(url='https://example.org',offset=999)
        store.save_fetch_failures.assert_not_called()
        self.assertEqual(guard.failures,{})


class TokenTests(TestCase):
    def test_tokens_accumulate_across_loops_and_cache_semantics(self):
        from llm.agent_loop import LoopState
        tracker = {}
        with patch('llm.agent_loop.record_llm_call'):
            first = LoopState(.2,budget_tracker=tracker)
            first.add_cost(.01,tokens_in=100,tokens_out=10,cache_read=60)
            # A provider fallback/new loop uses the same tracker.
            second = LoopState(.2,budget_tracker=tracker)
            second.add_cost(.02,tokens_in=20,tokens_out=15,cache_read=50,cache_create=10,token_semantics='anthropic')
        self.assertEqual(tracker['input_tokens'],180)
        self.assertEqual(tracker['output_tokens'],25)
        self.assertEqual(tracker['cache_read_tokens'],110)
        self.assertEqual(tracker['llm_responses'],2)
        self.assertAlmostEqual(tracker['observed_llm_cost_usd'],.03)
        self.assertNotIn('cost_complete',tracker)
