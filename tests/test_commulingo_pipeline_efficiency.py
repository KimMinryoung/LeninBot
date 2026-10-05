from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from commulingo.pipeline.draft_repair import DraftRepair
from commulingo.pipeline.editor import Editor
from commulingo.pipeline.evidence import SourceHandles, snapshot, Passages, resolve_passages, compile_evidence
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

    def test_handles_are_one_per_url_and_follow_the_latest_snapshot(self):
        source=snapshot('https://example.org/archive','Documented fact. '*80)
        sources={source['id']:source}
        handles=SourceHandles(sources)
        passages=Passages(); passages.show(source['id'],source['body'])
        self.assertEqual(resolve_passages([{'field':'body','claim':'fact','passages':['P1']}],passages,sources)[0]['start'],0)
        other=snapshot('https://example.org/other','Another page. '*10)
        self.assertEqual(handles.handle(other),'S2')
        self.assertEqual(handles.handle(source),'S1')
        # A later snapshot of the same URL keeps the handle and becomes its current target.
        later=snapshot('https://example.org/archive','Documented fact. '*80+'Appended page.')
        self.assertEqual(handles.handle(later),'S1')
        self.assertEqual(handles.ids['S1'],later['id'])

    def test_pages_of_one_url_merge_into_one_snapshot(self):
        from commulingo.pipeline.evidence import SourcePages
        pages=SourcePages()
        first,span1,created=pages.absorb('https://example.org/long','A'*600)
        self.assertTrue(created); self.assertEqual(span1,(0,600))
        second,span2,created=pages.absorb('https://example.org/long','B'*500)
        self.assertTrue(created)
        self.assertEqual(span2,(601,1101))
        self.assertEqual(second['body'],'A'*600+'\n'+'B'*500)
        again,span_again,created=pages.absorb('https://example.org/long','A'*600)
        self.assertFalse(created); self.assertIs(again,second); self.assertEqual(span_again,span1)
        # Snapshots a job already holds for one URL are merged oldest first.
        from datetime import datetime,timezone,timedelta
        t=datetime.now(timezone.utc)-timedelta(days=1)
        old={**snapshot('https://example.org/p','page one. '*30,now=t),}
        new={**snapshot('https://example.org/p','page two. '*30,now=t+timedelta(minutes=1))}
        single=snapshot('https://example.org/q','only page. '*30,now=t)
        seeded=SourcePages()
        merged=seeded.seed({old['id']:old,new['id']:new,single['id']:single})
        self.assertEqual([m['url'] for m in merged],['https://example.org/p'])
        self.assertTrue(merged[0]['body'].startswith('page one. ') and merged[0]['body'].endswith('page two. '))
        self.assertIs(seeded.current['https://example.org/q'],single)

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
        sources={source['id']:source}; handles=SourceHandles(sources)
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

