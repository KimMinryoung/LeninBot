from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
from commulingo_test_support import HermeticAsyncCase
from commulingo.pipeline.decisions import Decisions
from commulingo.pipeline.editor_context import RepairReads, prose_budgets
from commulingo.pipeline.engine import Usage
from tool_gateway.results import ToolRejection


class DecisionTests(HermeticAsyncCase):
    async def test_jev_classification_is_memoized_and_has_no_generation_fallback(self):
        usage = Usage()
        helper = Decisions({'kind':'term','action':'create'},None,None,usage)
        verdict = {'category':'economy','low_confidence':False}
        with patch('commulingo.classify.classify_term',return_value=verdict) as classify:
            a,_ = await helper.classify({'term':{'en':'Planning'}},[],{})
            b,_ = await helper.classify({'term':{'en':'Planning'}},[],{})
        self.assertEqual(a['category'],'economy'); self.assertEqual(a,b)
        classify.assert_called_once()
        helper.cache.clear()
        with patch('commulingo.classify.classify_term',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'no LLM fallback'):
                await helper.classify({'term':{'en':'Planning'}},[],{})

    async def test_existing_classification_is_preserved_during_code_assignment(self):
        helper = Decisions({'kind':'person','action':'update'},
            {'groupId':'existing','role':{'category':'scholar'}},None,Usage())
        with patch('commulingo.classify.classify_person_codes',return_value={'citizenship':{'code':'france'}}), \
             patch('commulingo.classify.classify_person_card') as card:
            result,_ = await helper.classify({'citizenship':{'label':{'ko':'프랑스','en':'France'}}},[],{})
        card.assert_not_called()
        self.assertNotIn('groupId',result); self.assertNotIn('role',result)
        self.assertEqual(result['citizenship']['code'],'france')

    def test_jev_cost_is_added_to_stage_ledger(self):
        usage = Usage(); usage.tracker['total_cost']=.01
        helper = Decisions({},None,None,usage)
        with patch('llm.call_registry.decide_detailed',return_value=SimpleNamespace(decision=SimpleNamespace(cost_usd=.0001))):
            helper.detailed('feature',{}, {})
        self.assertAlmostEqual(usage.tracker['jev_cost_usd'],.0001)
        self.assertAlmostEqual(usage.tracker['total_cost'],.01)
        self.assertEqual(usage.tracker['jev_calls'],1)

    async def test_format_repair_blocks_search_but_explicit_research_can_resume(self):
        sources = Mock(); call = AsyncMock(return_value='source')
        sources.wrap.return_value = call
        reads = RepairReads(sources,True)
        fetch = reads.wrap('fetch_url',call)
        with self.assertRaises(ToolRejection): await fetch(url='https://example.org')
        call.assert_not_called()
        _,reopen,_ = reads.tool(['body'],Usage())
        await reopen(fields=['body'],reason='The date conflicts with the source.')
        self.assertEqual(await fetch(url='https://example.org'),'source')

    def test_prose_budget_is_not_a_minimum(self):
        budgets = prose_budgets({'properties':{'bio':{'properties':{'ko':{'maxLength':500}}}}})
        self.assertEqual(budgets['bio']['ko'],{'draft_target':400,'hard_limit':500})
