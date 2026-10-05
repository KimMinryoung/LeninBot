import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from commulingo.evidence import resolve_evidence_sources
from commulingo.review_policy import review_source, resolve_review_checks, validate_decision, DECISION_TOOL
from scripts.commulingo_write_session import draft_id, prepare_write, repair_schema
from tool_gateway.validation import validate_tool_arguments
from tool_gateway.results import ToolRejection


class EfficiencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
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


    def test_review_checks_cite_displayed_passage_labels(self):
        body = 'The original archive records the birth — and the “subsequent” appointment.\nAnother paragraph.'
        url = 'https://archive.example/person'
        from commulingo.pipeline.evidence import Passages
        snapshots, passages = {}, Passages()
        source_id, shown = review_source(url, body, snapshots, passages, base=500)
        # Short labels bind the immutable slice, independently of its page offset.
        self.assertEqual(shown, '[P1] The original archive records the birth — and the “subsequent” appointment.\n'
                                '[P2] Another paragraph.')
        proposal = {'source_refs':[url+' — biography'], 'risks':[]}
        decision = {'decision':'approve','reason':'Original evidence substantiates the proposed facts.',
            'resolved_risks':[], 'checks':[{'citation_id':'S1','passages':['P1'],'finding':'The appointment is documented.'}]}
        validate_tool_arguments('commulingo_review_decision', decision,
                                schema=DECISION_TOOL['input_schema'], risk_class='state')
        resolved = resolve_review_checks(decision, proposal, snapshots, passages)
        self.assertEqual(resolved['checks'][0]['quote'], 'The original archive records the birth — and the “subsequent” appointment.')
        self.assertEqual(resolved['checks'][0]['source'], url)
        self.assertEqual(resolved['checks'][0]['citation'], url+' — biography')
        self.assertEqual(validate_decision(resolved, proposal), resolved)
        both = deepcopy(decision); both['checks'][0]['passages'] = ['P2', 'P1']
        self.assertEqual(resolve_review_checks(both, proposal, snapshots, passages)['checks'][0]['quote'], body)
        partial = deepcopy(decision); partial['checks'].append({'citation_id':'S1','passages':['P999'],'finding':'x'})
        with self.assertRaisesRegex(ValueError, 'check 2: passage labels not displayed: P999'):
            resolve_review_checks(partial, proposal, snapshots, passages)
        # Labels of two retrieved snapshots become one check per snapshot.
        review_source('https://other.example/page', 'Different page with enough text to cite.', snapshots, passages)
        two = deepcopy(decision); two['checks'][0]['passages'] = ['P1', 'P3']
        self.assertEqual([c['source'] for c in resolve_review_checks(two, proposal, snapshots, passages)['checks']], [url, 'https://other.example/page'])
        absent = deepcopy(decision); absent['checks'][0]['passages'] = ['P999']
        with self.assertRaisesRegex(ValueError, 'check 1: passage labels not displayed: P999'):
            resolve_review_checks(absent, proposal, snapshots, passages)

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


class DraftLengthGuidanceTests(unittest.TestCase):
    def test_overlength_rejection_lists_paragraph_sizes_and_counts_repeats(self):
        from commulingo.pipeline.draft_repair import DraftRepair
        tool = {'name':'commulingo_pipeline_result','input_schema':{'type':'object','additionalProperties':False,
            'properties':{'fields':{'type':'object','properties':{'body':{'type':'object',
                'properties':{'en':{'type':'string','maxLength':20}}}}}},'required':['fields']}}
        repairs = DraftRepair(tool)
        messages = []
        for text in ('first para here\n\nsecond paragraph is long', 'first para here\n\nsecond paragraph', 'first para here\n\nsecond para'):
            with self.assertRaises(ValueError) as rejected:
                repairs.prepare({'fields':{'body':{'en':text}}})
            messages.append(str(rejected.exception))
        self.assertIn('fields.body.en: remove at least 21 characters in ONE repair. Paragraph sizes: P1 15, P2 24', messages[0])
        self.assertNotIn('rejection #', messages[0])
        self.assertIn('over-length rejection #3', messages[2])
        # A schema error without a length excess adds no length guidance.
        with self.assertRaises(ValueError) as rejected:
            repairs.prepare({'fields':{'body':{'en':5}}})
        self.assertNotIn('Paragraph sizes', str(rejected.exception))
