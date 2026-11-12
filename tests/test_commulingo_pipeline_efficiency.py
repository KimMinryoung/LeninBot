from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from commulingo.pipeline.draft_repair import DraftRepair
from commulingo.pipeline.editor import Editor
from commulingo.pipeline.evidence import snapshot, Passages, resolve_passages, compile_evidence
from commulingo.pipeline.engine import Usage
from scripts.commulingo_write_session import draft_id
from commulingo_test_support import EditorCase


class EvidenceContracts(TestCase):
    def test_uncertain_fate_has_a_supported_schema_value(self):
        from commulingo.people import COMMULINGO_PERSON_CREATE_TOOL, _COMMULINGO_FIELD_SCHEMA
        from jsonschema import validate as schema_validate, ValidationError
        schema=_COMMULINGO_FIELD_SCHEMA['properties']['fate']
        schema_validate({'kind':'','label':{'ko':'사망 경위 미확정','en':'Circumstances unconfirmed'}},schema)
        with self.assertRaises(ValidationError):
            schema_validate({'kind':'unknown','label':{'ko':'미확정','en':'Unknown'}},schema)
        # The writer's tool carries the label only; the runner assigns kind.
        tool_fate=COMMULINGO_PERSON_CREATE_TOOL['input_schema']['properties']['fields']['properties']['fate']
        self.assertNotIn('kind', tool_fate['properties'])
        schema_validate({'label':{'ko':'사망 경위 미확정','en':'Circumstances unconfirmed'}},tool_fate)

    def test_empty_arguments_are_explained_as_unparsed_json(self):
        from tool_gateway.validation import validate_tool_arguments, ToolArgumentValidationError
        schema={'type':'object','properties':{'fields':{'type':'object'}},'required':['fields']}
        with self.assertRaisesRegex(ToolArgumentValidationError,'empty object.*not parseable.*fields'):
            validate_tool_arguments('commulingo_pipeline_result',{},schema=schema,risk_class='state')
        with self.assertRaisesRegex(ToolArgumentValidationError,"^'fields' is a required property"):
            validate_tool_arguments('commulingo_pipeline_result',{'notes':'x'},schema=schema,risk_class='state')
        merge={'type':'object','properties':{'changes':{'type':'object'}},'minProperties':1}
        with self.assertRaisesRegex(ToolArgumentValidationError,'empty object.*not parseable'):
            validate_tool_arguments('merge_tool',{},schema=merge,risk_class='state')
        # The editor submit tool accepts parts, so an empty call asks for smaller ones.
        with self.assertRaisesRegex(ToolArgumentValidationError,'Do not resend the same call.*in parts'):
            validate_tool_arguments('commulingo_pipeline_submit_draft',{},schema=merge,risk_class='state')

    def test_many_passages_are_all_kept(self):
        source=snapshot('https://example.org/archive','Documented fact number one. '*100)
        sources={source['id']:source}
        passages=Passages(); passages.show(source['id'],source['body'])
        claim={'field':'body','claim':'fact','passages':['P1']}
        result=resolve_passages([claim]*40,passages,sources)
        self.assertEqual(len(result),40)

    def test_valid_draft_survives_downstream_error_and_repairs_follow_the_current_draft(self):
        repair=DraftRepair({'name':'draft','input_schema':{'type':'object','properties':{
            'fields':{'type':'object','properties':{'years':{'type':'string'}}}},'required':['fields']}})
        repair.prepare({'fields':{'years':'present'}})
        old=draft_id(repair.draft)
        self.assertIn(old,repair.feedback('years must be a range'))
        repair.prepare({'draft_id':old,'repairs':[{'op':'set','path':'/fields/years','value':'1900–'}]})
        self.assertNotEqual(draft_id(repair.draft),old)
        # An echoed earlier ID no longer costs a round: the call holds one draft.
        repair.prepare({'draft_id':old,'repairs':[{'op':'set','path':'/fields/years','value':'1900–1950'}]})
        self.assertEqual(repair.draft['args']['fields']['years'],'1900–1950')


class EditorStorageContracts(EditorCase):
    async def test_storage_validation_is_repaired_in_same_call_and_keeps_revision(self):
        from test_commulingo_editor import BODY, CURRENT, JOB, URL, candidate, repair_call, store_mock, submission
        rpc=Mock(side_effect=[CURRENT,ValueError('400: invalid period label'),{}])
        async def model(**kw):
            await kw['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            with self.assertRaisesRegex(ValueError,'invalid period label'):
                await kw['handler'](submission(candidate()))
            value=candidate()
            value['fields']['body']['en']='A corrected documented historical context.'
            await repair_call(kw,value)
        usage=Usage()
        with patch('commulingo.pipeline.stages.model_call',side_effect=model) as call, \
             patch('commulingo.pipeline.service.call',rpc):
            result=await Editor(store_mock())(JOB,[],usage,.2)
        call.assert_called_once()
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(result.value['draft']['fields']['expectedRevision'],CURRENT['revision'])
        self.assertEqual(result.value['draft']['fields']['body']['en'],'A corrected documented historical context.')
        self.assertEqual(usage.tracker['preflight_failures'],1)
        self.assertTrue(usage.tracker['preflight_passed'])
        self.assertEqual([c.args[0]['command'] for c in rpc.call_args_list],['read','validate','validate'])


class BudgetDrainRegression(IsolatedAsyncioTestCase):

    def test_large_claim_sets_compile_without_a_cap(self):
        source=snapshot('https://example.org/archive','A documented definition supported by this archive.')
        claim={'field':'definition','claim':'Definition','source_id':source['id'],'start':0,'end':len(source['body'])}
        evidence=compile_evidence([claim]*63,{source['id']:source},{'definition'})
        self.assertEqual(len(evidence),63)
        self.assertEqual({e['excerpt'] for e in evidence},{source['body']})

