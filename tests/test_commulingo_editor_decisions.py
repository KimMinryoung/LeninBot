from unittest.mock import patch
from types import SimpleNamespace
from commulingo_test_support import HermeticAsyncCase
from commulingo.pipeline.decisions import Decisions
from commulingo.pipeline.editor_context import prose_budgets
from commulingo.pipeline.engine import Usage


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
            {'groupId':'existing','activities':[{'functionId':'scholarship','primary':True}]},None,Usage())
        with patch('commulingo.classify.classify_person_codes',return_value={'citizenship':{'code':'france'}}), \
             patch('commulingo.classify.classify_person_card') as card:
            result,_ = await helper.classify({'citizenship':{'label':{'ko':'프랑스','en':'France'}}},[],{})
        card.assert_not_called()
        self.assertNotIn('groupId',result); self.assertNotIn('activities',result)
        self.assertEqual(result['citizenship']['code'],'france')

    async def test_update_appends_a_documented_party_activity_only_when_none_exists(self):
        body = 'He joined the Communist Party in 1932.'
        claims = [{'source_id':'s1','field':'career','claim':'joined the party','start':0,'end':len(body)}]
        sources = {'s1':{'body':body,'url':'https://example.org/p'}}
        primary = {'functionId':'military','affiliationId':'state-soviet','primary':True}
        party = {'functionId':'military','affiliationId':'soviet-party','relation':'membership','primary':False}
        helper = Decisions({'kind':'person','action':'update'},{'groupId':'g','activities':[primary]},None,Usage())
        with patch('commulingo.classify.classify_party_activity',return_value=party) as stage:
            result,_ = await helper.classify({'bio':{'en':'x'}},claims,sources)
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[2][0]['excerpt'], body)
        self.assertEqual(result['activities'],[primary,party])
        member = Decisions({'kind':'person','action':'update'},{'groupId':'g','activities':[primary,party]},None,Usage())
        with patch('commulingo.classify.classify_party_activity') as stage:
            result,_ = await member.classify({'bio':{'en':'x'}},claims,sources)
        stage.assert_not_called()
        self.assertNotIn('activities',result)

    def test_jev_cost_is_added_to_stage_ledger(self):
        usage = Usage(); usage.tracker['total_cost']=.01
        helper = Decisions({},None,None,usage)
        with patch('llm.call_registry.decide_detailed',return_value=SimpleNamespace(decision=SimpleNamespace(cost_usd=.0001))):
            helper.detailed('feature',{}, {})
        self.assertAlmostEqual(usage.tracker['jev_cost_usd'],.0001)
        self.assertAlmostEqual(usage.tracker['total_cost'],.01)
        self.assertEqual(usage.tracker['jev_calls'],1)

    def test_prose_budget_is_not_a_minimum(self):
        budgets = prose_budgets({'properties':{'bio':{'properties':{'ko':{'maxLength':500}}}}})
        self.assertEqual(budgets['bio']['ko'],{'draft_target':400,'hard_limit':500})
