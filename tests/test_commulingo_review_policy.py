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
from runtime_tools.commulingo_review_policy import validate_decision
from telegram.commulingo_review import cmd_commulingo_review
spec=importlib.util.spec_from_file_location('reviewer_test_module',ROOT/'scripts/commulingo_person_reviewer.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
SOURCE='https://archive.example/entry'
QUOTE='The archived register identifies two distinct people with different birth dates.'
PROPOSAL={'source_refs':[SOURCE],'risks':['identity_uncertain']}
DECISION={'decision':'approve','reason':'원본 기록의 생년과 직책을 대조하여 동명이인임을 확인했습니다.',
    'resolved_risks':['identity_uncertain'],'checks':[{'citation':SOURCE,'source':SOURCE,'quote':QUOTE,'finding':'서로 다른 인물임을 확인'}]}

class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
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

    def test_only_retrieved_quotes_covering_sources_and_risks_can_approve(self):
        self.assertEqual(validate_decision(DECISION,PROPOSAL,{SOURCE:QUOTE}),DECISION)
        for fetched in ({},{SOURCE:'search result snippet'}):
            with self.assertRaises(ValueError):validate_decision(DECISION,PROPOSAL,fetched)
        for change in ({'resolved_risks':[]},{'checks':[]}):
            with self.assertRaises(ValueError):validate_decision({**DECISION,**change},PROPOSAL,{SOURCE:QUOTE})
        with self.assertRaises(ValueError):validate_decision(DECISION,{**PROPOSAL,'source_refs':['another citation']},{SOURCE:QUOTE})
    def test_failed_coverage_reports_exact_missing_identifiers(self):
        with self.assertRaisesRegex(ValueError, 'original citation with annotation'):
            validate_decision(DECISION, {**PROPOSAL, 'source_refs': ['original citation with annotation']}, {SOURCE: QUOTE})
        with self.assertRaisesRegex(ValueError, 'identity_uncertain'):
            validate_decision({**DECISION, 'resolved_risks': ['identity_uncertain: explanation']}, PROPOSAL, {SOURCE: QUOTE})

    def test_uncertainty_can_escalate_without_inventing_evidence(self):
        value={**DECISION,'decision':'escalate','checks':[]}
        validate_decision(value,PROPOSAL,{})
    async def test_fetch_wrapper_counts_body_not_failed_diagnostics_or_search(self):
        box,fetched={},{}
        handlers=worker.make_handlers({'fetch_url':AsyncMock(return_value='[fetch_url]\nError: '+QUOTE)},PROPOSAL,fetched,box)
        await handlers['fetch_url'](url=SOURCE)
        self.assertEqual(fetched,{})
        handlers=worker.make_handlers({'fetch_url':AsyncMock(return_value=f'<external source="url:{SOURCE}">\n{QUOTE}\n</external>')},PROPOSAL,fetched,box)
        await handlers['fetch_url'](url=SOURCE)
        await handlers['commulingo_review_decision'](**DECISION)
        self.assertEqual(box['decision'],'approve')
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
        with patch('telegram.commulingo_review.queue.detail',return_value={'status':'pending'}),patch('telegram.commulingo_review.queue.synchronize'),patch('telegram.commulingo_review.call_person_service',return_value={'status':'approved'}) as rpc:
            await cmd_commulingo_review(message,{'is_allowed':lambda uid:uid==1})
            self.assertEqual(rpc.call_args.args[0]['changedBy'],'telegram-owner:1')
            self.assertTrue(rpc.call_args.args[0]['approve'])

if __name__=='__main__':unittest.main()
