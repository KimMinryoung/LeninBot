import asyncio
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

ROOT=Path(os.environ.get('COMMULINGO_REVIEW_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,'/home/grass/leninbot')
import runtime_tools, telegram
runtime_tools.__path__.insert(0,str(ROOT/'runtime_tools'))
telegram.__path__.insert(0,str(ROOT/'telegram'))
from runtime_tools.commulingo_review_policy import validate_decision, resolve_review_checks
from telegram.commulingo_review import cmd_commulingo_review
spec=importlib.util.spec_from_file_location('reviewer_test_module',ROOT/'scripts/commulingo_person_reviewer.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
SOURCE='https://archive.example/entry'
QUOTE='The archived register identifies two distinct people with different birth dates.'
PROPOSAL={'source_refs':[SOURCE],'risks':['identity_uncertain']}
DECISION={'decision':'approve','reason':'원본 기록의 생년과 직책을 대조하여 동명이인임을 확인했습니다.',
    'resolved_risks':['identity_uncertain'],'checks':[{'citation':SOURCE,'source':SOURCE,'quote':QUOTE,'finding':'서로 다른 인물임을 확인'}]}
from runtime_tools.commulingo_review_policy import review_source as _review_source
LABEL=f"{_review_source(SOURCE,QUOTE,{})[0]}@0"   # the label the wrapper shows for QUOTE fetched at offset 0
SUBMITTED={**DECISION,'checks':[{'citation':SOURCE,'passages':[LABEL],'finding':'서로 다른 인물임을 확인'}]}

class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reservation = patch('commulingo_pipeline.config.legacy_reserve',return_value=None)
        reservation.start()
        self.addCleanup(reservation.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        ledger = patch('scripts.commulingo_research_memory.STORE_PATH', Path(directory.name) / 'review.sqlite3')
        ledger.start()
        self.addCleanup(ledger.stop)
        # All sync dependencies in this suite are mocks. Avoid creating an
        # executor solely for them (sandbox loop shutdown can lose its wakeup).
        async def inline(call, *args, **kwargs):
            return call(*args, **kwargs)
        threaded = patch('asyncio.to_thread', side_effect=inline)
        threaded.start()
        self.addCleanup(threaded.stop)

    def test_only_retrieved_passages_resolving_risks_can_approve(self):
        self.assertEqual(validate_decision(DECISION,PROPOSAL,{SOURCE:QUOTE}),DECISION)
        # A check cites labels of paragraphs shown in this review; with nothing retrieved no check resolves.
        with self.assertRaisesRegex(ValueError,'no check could be verified'):resolve_review_checks(SUBMITTED,PROPOSAL,{})
        snapshots={}; _review_source(SOURCE,QUOTE,snapshots)
        self.assertEqual(resolve_review_checks(SUBMITTED,PROPOSAL,snapshots)['checks'],DECISION['checks'])
        for change in ({'resolved_risks':[]},{'checks':[]}):
            with self.assertRaises(ValueError):validate_decision({**DECISION,**change},PROPOSAL,{SOURCE:QUOTE})
        # Relaxed 2026-09-17: approval no longer needs a check per cited reference
        # or a source outside Wikipedia, only verified quotes and resolved risks.
        self.assertEqual(validate_decision(DECISION,{**PROPOSAL,'source_refs':['another citation',SOURCE]},{SOURCE:QUOTE}),DECISION)
        wiki='https://en.wikipedia.org/wiki/Entry'
        wiki_only={**DECISION,'checks':[{**DECISION['checks'][0],'source':wiki}]}
        self.assertEqual(validate_decision(wiki_only,PROPOSAL,{wiki:QUOTE}),wiki_only)
    def test_research_routing_hint_requires_boolean(self):
        for needed in (True,False):
            value={**DECISION,'decision':'revise','needs_research':needed}
            self.assertEqual(validate_decision(value,PROPOSAL,{SOURCE:QUOTE}),value)
        with self.assertRaisesRegex(ValueError,'boolean'):
            validate_decision({**DECISION,'needs_research':'false'},PROPOSAL,{SOURCE:QUOTE})

    def test_many_checks_are_valid_and_an_unshown_label_drops_only_its_check(self):
        snapshots={}; _review_source(SOURCE,QUOTE,snapshots)
        checks=[dict(SUBMITTED['checks'][0]) for _ in range(63)]
        value={**DECISION,'checks':checks}
        resolved=resolve_review_checks(value,PROPOSAL,snapshots)
        self.assertEqual(len(resolved['checks']),63)
        self.assertEqual(validate_decision(resolved,PROPOSAL,{SOURCE:QUOTE}),resolved)
        checks[-1]['passages']=['R0123456789abcdef@0']
        resolved=resolve_review_checks(value,PROPOSAL,snapshots)
        self.assertEqual((len(resolved['checks']),resolved['dropped_checks'][0]['check']),(62,63))

    def test_failed_coverage_reports_exact_missing_identifiers(self):
        with self.assertRaisesRegex(ValueError, 'identity_uncertain'):
            validate_decision({**DECISION, 'resolved_risks': ['identity_uncertain: explanation']}, PROPOSAL, {SOURCE: QUOTE})

    def test_uncertainty_can_escalate_without_inventing_evidence(self):
        value={**DECISION,'decision':'escalate','checks':[]}
        validate_decision(value,PROPOSAL,{})
    def test_passage_errors_identify_the_check(self):
        snapshots={}; sid,_=_review_source(SOURCE,QUOTE+'\nShort title',snapshots)
        for label, reason in [(f'{sid}@{len(QUOTE)+1}', 'cited passage is shorter than 20 characters'),
                              (f'{sid}@7', f'passage label not shown in this review: {sid}@7')]:
            value = copy.deepcopy(SUBMITTED)
            value['checks'].append({**value['checks'][0], 'passages':[label]})
            value['checks'][0]['passages']=[f'{sid}@0']
            resolved=resolve_review_checks(value,PROPOSAL,snapshots)
            self.assertEqual(resolved['dropped_checks'],[{'check':2,'labels':[label],'reason':reason}])
            self.assertEqual(resolved['checks'][0]['quote'],QUOTE)
    def test_revision_requires_independently_retrieved_evidence(self):
        value={**DECISION,'decision':'revise'}
        self.assertEqual(validate_decision(value,PROPOSAL,{SOURCE:QUOTE}),value)
        with self.assertRaisesRegex(ValueError,'no check could be verified'):
            resolve_review_checks({**SUBMITTED,'decision':'revise'},PROPOSAL,{})
        with self.assertRaises(ValueError):
            validate_decision({**value,'checks':[]},PROPOSAL,{SOURCE:QUOTE})

    async def test_notifications_disabled_even_for_old_pending_requests(self):
        with patch.object(worker.queue,'notifications') as pending, patch.object(worker.queue,'synchronize'):
            self.assertEqual(worker.deliver_notifications(),0)
            result=await worker.run(notify_only=True)
        pending.assert_not_called()
        self.assertEqual(result['status'],'notifications_disabled')

    async def test_legacy_revision_persists_correction_without_approving_original(self):
        row={'id':7,'status':'pending','target_type':'person','target_id':'fixture',
             'action':'create','patch_json':{},'source_refs':[SOURCE]}
        decision={**DECISION,'decision':'revise'}
        job={'suggestion_id':7,'lease_token':'token'}
        with patch.object(worker.queue,'suggestion',return_value=row), \
             patch.object(worker.queue,'save_decision',return_value=True), \
             patch.object(worker.queue,'owned',return_value=True), \
             patch.object(worker.queue,'finish') as finish, \
             patch.object(worker,'research',new=AsyncMock(return_value=(decision,{SOURCE:QUOTE}))), \
             patch.object(worker,'call_person_service',return_value=None) as rpc, \
             patch('commulingo_pipeline.store.Store.enqueue_review_repair',return_value=123) as enqueue:
            await worker.process(job,{})
        enqueue.assert_called_once_with(row,decision)
        self.assertEqual(rpc.call_count,1)
        self.assertEqual(rpc.call_args.args[0]['command'],'read')
        self.assertIn('123',finish.call_args.args[2])

    async def test_fetch_wrapper_counts_body_not_failed_diagnostics_or_search(self):
        box,fetched={},{}
        handlers=worker.make_handlers({'fetch_url':AsyncMock(return_value='[fetch_url]\nError: '+QUOTE)},PROPOSAL,fetched,box)
        await handlers['fetch_url'](url=SOURCE)
        self.assertEqual(fetched,{})
        handlers=worker.make_handlers({'fetch_url':AsyncMock(return_value=f'<external source="url:{SOURCE}">\n{QUOTE}\n</external>')},PROPOSAL,fetched,box)
        shown=await handlers['fetch_url'](url=SOURCE)
        self.assertIn(f'<external source="url:{SOURCE}">\n[{LABEL}] {QUOTE}\n</external>',shown)
        await handlers['commulingo_review_decision'](**SUBMITTED)
        self.assertEqual(box['decision'],'approve')
        self.assertEqual((box['checks'][0]['source'],box['checks'][0]['quote']),(SOURCE,QUOTE))
    async def test_decision_gate_annotates_or_bounces_before_boxing(self):
        from tool_gateway.results import ToolRejection
        async def annotate(value):
            return {**value,'checks':[{**c,'citation_check':{'support':'supports'}} for c in value['checks']]}
        box,fetched={},{}
        handlers=worker.make_handlers({'fetch_url':AsyncMock(return_value=f'<external source="url:{SOURCE}">\n{QUOTE}\n</external>')},PROPOSAL,fetched,box,gate=annotate)
        await handlers['fetch_url'](url=SOURCE)
        await handlers['commulingo_review_decision'](**SUBMITTED)
        self.assertEqual(box['checks'][0]['citation_check'],{'support':'supports'})
        async def bounce(value):
            raise ValueError('citation check failed for one check')
        box,fetched={},{}
        handlers=worker.make_handlers({'fetch_url':AsyncMock(return_value=f'<external source="url:{SOURCE}">\n{QUOTE}\n</external>')},PROPOSAL,fetched,box,gate=bounce)
        await handlers['fetch_url'](url=SOURCE)
        with self.assertRaisesRegex(ToolRejection,'citation check failed'):
            await handlers['commulingo_review_decision'](**SUBMITTED)
        self.assertEqual(box,{})
    async def test_real_runner_context_and_typed_terminal_without_network(self):
        import db
        with patch.object(db,'query',return_value=[]),patch.object(db,'query_one',return_value=None):
            import agents
            agents.__path__.insert(0,str(ROOT/'agents'))
            import bot_config
            import runtime_tools.registry as registry
        from tool_gateway.security import get_caller
        from tool_gateway.dispatcher import execute_tool
        async def chat(*args,**kwargs):
            self.assertEqual(get_caller().agent_name,'commulingo_reviewer')
            self.assertTrue(kwargs['continue_on_length'])
            self.assertEqual(kwargs['max_length_continuations'], 2)
            self.assertEqual(set(kwargs['tool_handlers']),{'wiki_search','wiki_get','web_search','fetch_url','commulingo_people','commulingo_review_decision'})
            self.assertFalse(any(k.startswith('commulingo_person_') for k in kwargs['tool_handlers']))
            await kwargs['tool_handlers']['fetch_url'](url=SOURCE)
            with patch('tool_gateway.security.audit'):
                result, failed = await execute_tool('commulingo_review_decision', DECISION, kwargs['tool_handlers'], tool_schema=worker.DECISION_TOOL)
            self.assertFalse(failed, result)
            return 'An irrelevant model summary'
        names=['wiki_search','wiki_get','web_search','fetch_url','commulingo_people']
        reads={name:AsyncMock(return_value=f'<external source="url:{SOURCE}">\n{QUOTE}\n</external>') for name in names}
        binding=SimpleNamespace(chat=chat,client=None,model='fixture',render_provider='deepseek',reasoning={})
        with patch.object(bot_config,'resolve_agent_tool_loop',return_value=binding),patch.object(registry,'TOOLS',[{'name':name} for name in names]),patch.object(registry,'TOOL_HANDLERS',reads):
            decision,fetched=await worker.research({**PROPOSAL,'id':1},{}, {})
        self.assertEqual(decision['decision'],'approve')
        self.assertIn(QUOTE,fetched[SOURCE])
    def test_review_decision_is_reviewer_only_and_owner_only(self):
        from security_gateway import CallerContext, authorize
        from security_gateway import policy
        with patch.object(policy, 'enforce_mode', return_value='enforce'):
            for agent, owner, expected in [('commulingo_reviewer', True, True), ('commulingo_curator', True, False), ('commulingo_reviewer', False, False)]:
                ctx = CallerContext(interface='autonomous', agent_name=agent, is_owner=owner)
                result = authorize(ctx, 'commulingo_review_decision', {}, consume_rate_limit=False)
                self.assertEqual(result.allowed, expected, result)

    def test_health_digest_treats_empty_review_queue_as_normal(self):
        health_spec=importlib.util.spec_from_file_location('review_health',ROOT/'scripts/commulingo_lane_health.py')
        health=importlib.util.module_from_spec(health_spec);health_spec.loader.exec_module(health)
        with patch.object(health,'journal',return_value='  "status": "idle",\n  "cost_usd": 0'):
            stats=health.tally('leninbot-commulingo-review.service','today')
        self.assertEqual(stats['idle'],1)
        self.assertEqual(health.problems('review',stats),[])
        self.assertIn('review',health.LANES)
    def test_stale_and_legacy_versions_never_refresh(self):
        row={'target_type':'person','action':'update','patch_json':{'expectedRevision':'old'}}
        self.assertTrue(worker.invalidated(row,{'revision':'new'}))
        self.assertIsNone(worker.invalidated(row,{'revision':'old'}))
        self.assertTrue(worker.invalidated({**row,'patch_json':{}},{'revision':'new'}))
    async def test_owner_private_chat_and_explicit_note_are_required(self):
        ctx={'is_allowed':lambda user:user==1}
        for uid,kind in [(2,'private'),(1,'group')]:
            message=SimpleNamespace(from_user=SimpleNamespace(id=uid),chat=SimpleNamespace(type=kind),text='/commulingo_review approve 5 승인',answer=AsyncMock())
            with patch('telegram.commulingo_review.queue.detail') as read:
                await cmd_commulingo_review(message,ctx);read.assert_not_called();message.answer.assert_not_called()
        message=SimpleNamespace(from_user=SimpleNamespace(id=1),chat=SimpleNamespace(type='private'),text='/commulingo_review approve 5',answer=AsyncMock())
        with patch('telegram.commulingo_review.queue.detail',return_value={'status':'pending'}),patch('telegram.commulingo_review.call_person_service') as rpc:
            await cmd_commulingo_review(message,ctx);rpc.assert_not_called()
    async def test_owner_review_uses_common_service_with_identity(self):
        message=SimpleNamespace(from_user=SimpleNamespace(id=1),chat=SimpleNamespace(type='private'),text='/commulingo_review approve 5 원문과 생년을 대조함',answer=AsyncMock())
        with patch('telegram.commulingo_review.queue.detail',return_value={'status':'pending','target_type':'person'}),patch('telegram.commulingo_review.queue.synchronize'),patch('telegram.commulingo_review.call_person_service',return_value={'status':'approved'}) as rpc:
            await cmd_commulingo_review(message,{'is_allowed':lambda uid:uid==1})
            self.assertEqual(rpc.call_args.args[0]['changedBy'],'telegram-owner:1')
            self.assertTrue(rpc.call_args.args[0]['approve'])

if __name__=='__main__':unittest.main()
