from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import unittest

from jsonschema import Draft202012Validator

from commulingo.pipeline.author_draft import AuthorDraft, obj
from commulingo.pipeline.draft_repair import RepairProtocolError
from commulingo_test_support import EditorCase
from tool_gateway.validation import validate_tool_arguments, ToolArgumentValidationError


def session():
    fields = obj({'bio': obj({'ko': {'type':'string', 'maxLength':20},
                              'en': {'type':'string', 'maxLength':40}}, ['ko', 'en']),
                  'years': {'type':'string'}})
    draft = AuthorDraft(fields, [{'id':'missing:bio'}], factual_fields={'bio', 'years'})
    return draft


def edit():
    return {'fields': {'bio': {'ko':'검증한 인물이다.', 'en':'A documented person.'}, 'years':'1900–1980'},
            'evidence': {'bio': [{'claim':'A documented person.', 'passages':['P1']}],
                         'years': [{'claim':'The life dates.', 'passages':['P2']}]},
            'issues': {'missing:bio': {'status':'resolved', 'reason':'Added the supported biography.'}},
            'reason':'The archive establishes the commissioned facts.'}


class AuthorDraftTests(unittest.TestCase):
    def validate(self, value, draft):
        return validate_tool_arguments(draft.submit_tool['name'], value,
                                       schema=draft.submit_tool['input_schema'], risk_class='write')

    def test_complete_edit_is_one_call_with_flat_values(self):
        draft = session()
        value = self.validate(edit(), draft)
        result = draft.submission(value)
        self.assertEqual(draft.missing(result), [])
        draft.prepare(result)
        self.assertEqual(draft.view(), {**edit(), 'notes':''})
        self.assertEqual(result['fields'], edit()['fields'])
        self.assertEqual([c['field'] for c in result['claims']], ['bio', 'years'])

    def test_evidence_can_arrive_separately_and_survives_restart(self):
        draft = session()
        value = edit(); del value['evidence']['bio']
        part = draft.submission(value)
        self.assertEqual(draft.missing(part), ['evidence.bio'])
        resumed = session(); resumed.draft = {'tool':draft.name, 'args':deepcopy(part)}
        whole = resumed.submission(self.validate({'evidence': {'bio':edit()['evidence']['bio']}}, resumed))
        self.assertEqual(resumed.missing(whole), [])
        self.assertEqual(whole['fields'], part['fields'])
        resumed.draft['args'] = whole
        self.assertEqual(resumed.missing(resumed.submission({'evidence':{'bio':[]}})), ['evidence.bio'])
        with self.assertRaisesRegex(RepairProtocolError, 'needs a saved or supplied'):
            session().submission({'evidence': {'bio':edit()['evidence']['bio']}})

    def test_field_and_evidence_updates_preserve_other_fields(self):
        draft = session(); draft.prepare(draft.submission(edit()))
        result = draft.prepare(draft.submission({'fields': {'bio':{'en':'A revised biography.'}},
            'evidence': {'bio':[{'claim':'A revised claim.', 'passages':['P3']}]}}))
        self.assertEqual(result['fields']['years'], '1900–1980')
        self.assertEqual(result['fields']['bio']['ko'], edit()['fields']['bio']['ko'])
        self.assertEqual([(c['field'], c['passages']) for c in result['claims']], [('years',['P2']), ('bio',['P3'])])
        self.assertEqual(result['issue_results'][0]['id'], 'missing:bio')

    def test_bilingual_parts_and_metadata_accumulate(self):
        draft = session()
        part = draft.submission({'fields':{'bio':{'ko':'검증한 인물이다.'}}})
        self.assertEqual(draft.missing(part), ['fields.bio.en', 'evidence.bio', 'issues.missing:bio', 'reason'])
        draft.draft = {'tool':draft.name, 'args':deepcopy(part)}
        with self.assertRaises(ValueError):
            draft.prepare({**part, 'reason':edit()['reason']})
        second = draft.submission({'fields':{'bio':{'en':'A documented person.'}},
            'evidence': {'bio':edit()['evidence']['bio']}, 'issues':edit()['issues'], 'reason':edit()['reason']})
        self.assertEqual(draft.missing(second), [])
        self.assertEqual(second['fields']['bio'], edit()['fields']['bio'])

    def test_invalid_input_never_corrupts_saved_draft(self):
        draft = session(); draft.prepare(draft.submission(edit())); saved = deepcopy(draft.draft)
        for value in ({'fields':{'bio':'wrong type'}}, {'evidence':{'invented':[]}},
                      {'repairs':[{'path':'/fields/bio','op':'remove'}]}, {}, {'fields':{}}, {'issues':{}},
                      {'fields':{'bio':{'value':edit()['fields']['bio']}}}):
            with self.subTest(value=value), self.assertRaises(RepairProtocolError):
                draft.submission(value)
            self.assertEqual(draft.draft, saved)

    def test_overlength_text_is_saved_and_repaired_without_evidence_resend(self):
        draft = session(); value = edit(); value['fields']['bio']['en'] = 'x' * 41
        self.validate(value, draft)
        with self.assertRaisesRegex(ValueError, 'fields.bio.en'):
            draft.prepare(draft.submission(value))
        self.assertEqual(draft.view()['fields']['bio']['en'], 'x' * 41)
        result = draft.prepare(draft.submission({'fields':{'bio':{'en':'A documented person.'}}}))
        self.assertEqual(result['fields'], edit()['fields'])
        self.assertEqual(draft.view()['evidence'], edit()['evidence'])

    def test_withdrawal_removes_only_optional_draft_fields_and_evidence(self):
        draft = AuthorDraft(obj({'bio':{'type':'string'},'years':{'type':'string'}}, ['bio']),
                            [{'id':'missing:bio'}])
        self.assertEqual(draft.submit_tool['input_schema']['properties']['remove_fields']['items']['enum'], ['years'])
        draft = session(); draft.prepare(draft.submission(edit()))
        result = draft.prepare(draft.submission({'remove_fields':['years']}))
        self.assertNotIn('years', result['fields'])
        self.assertEqual([c['field'] for c in result['claims']], ['bio'])
        for extra in ({'fields':{'bio':edit()['fields']['bio']}}, {'evidence':{'bio':[]}}):
            with self.assertRaisesRegex(RepairProtocolError, 'same call'):
                draft.submission({**extra, 'remove_fields':['bio']})
        with self.assertRaisesRegex(RepairProtocolError, 'No saved draft'):
            session().submission({'remove_fields':['years']})

    def test_explicit_issue_decisions_cannot_be_invented_or_inferred(self):
        draft = session(); value = edit(); value['issues'] = {}
        self.assertEqual(draft.missing(draft.submission(value)), ['issues.missing:bio'])
        value['issues'] = {'invented': {'status':'resolved', 'reason':'Not a commissioned issue.'}}
        with self.assertRaises(RepairProtocolError): draft.submission(value)
        draft.prepare(draft.submission(edit()))
        result = draft.submission({'issues':{'missing:bio':{'status':'deferred','reason':'A conflict needs research.'}}})
        self.assertEqual(result['issue_results'][0]['status'], 'deferred')
        self.assertEqual(result['fields'], edit()['fields'])

    def test_legacy_checkpoint_keeps_values_labels_and_claim_error_positions(self):
        draft = session()
        draft.draft = {'tool':draft.name, 'args':{'status':'ready', 'reason':edit()['reason'],
            'fields':edit()['fields'], 'claims':[
                {'field':'years','claim':'Life dates','passages':['P18']},
                {'field':'bio','claim':'Previous claim','passages':['P27']}],
            'issue_results':[{'id':'missing:bio', **edit()['issues']['missing:bio']}]}}
        self.assertEqual(draft.view()['evidence']['bio'][0]['passages'], ['P27'])
        result = draft.prepare(draft.submission({'evidence':{'bio':edit()['evidence']['bio']}}))
        self.assertEqual(result['claims'][0]['passages'], ['P18'])
        self.assertEqual(draft.author_error('/claims/1/passages'), '/evidence/bio/0/passages')

    def test_malformed_legacy_containers_are_retained_until_replacement(self):
        draft = session()
        draft.draft = {'tool':draft.name,'args':{'fields':['bad'],'claims':'bad','issue_results':None}}
        saved = deepcopy(draft.draft)
        self.assertTrue(draft.view()['needs_full_submission'])
        fresh = draft.submission(edit())
        self.assertEqual(draft.missing(fresh), [])
        self.assertEqual(draft.draft, saved)
        draft.prepare(fresh)
        self.assertEqual(draft.view()['fields'], edit()['fields'])

    def test_validation_and_saved_state_do_not_alias_caller_data(self):
        draft = session()
        payload = draft.submission(edit())
        result = draft.prepare(payload)
        payload['fields']['bio']['ko'] = 'changed by caller'
        result['claims'][0]['passages'].append('P99')
        self.assertEqual(draft.view(), {**edit(), 'notes': ''})
        saved = deepcopy(draft.draft)
        with self.assertRaises(RepairProtocolError):
            draft.prepare({})
        self.assertEqual(draft.draft, saved)
        resumed = AuthorDraft(draft.canonical_fields, [{'id':'missing:bio'}], draft=saved)
        resumed.draft['args']['fields']['bio']['ko'] = 'changed after resume'
        self.assertEqual(saved, draft.draft)

    def test_only_provider_arguments_envelope_is_unwrapped(self):
        draft = session()
        for wrapped in (edit(), json.dumps(edit())):
            self.assertEqual(self.validate({'arguments':wrapped}, draft), edit())
        with self.assertRaises(ToolArgumentValidationError):
            self.validate({'arguments':edit(), 'fields':edit()['fields']}, draft)
        # Retired nesting is rejected rather than silently moving editorial data.
        with self.assertRaises(ToolArgumentValidationError):
            self.validate({'changes':{'bio':{'value':edit()['fields']['bio']}}}, draft)

    def test_no_edit_requires_a_reason_and_every_issue(self):
        draft = session()
        good = {'status':'sources_unavailable','reason':'The original could not be retrieved.',
                'issues':{'missing:bio':{'status':'deferred','reason':'No accessible supporting original.'}}}
        draft.validate_call(good, draft.no_edit_tool)
        for extra in ('fields', 'evidence', 'claims', 'repairs'):
            with self.assertRaises(RepairProtocolError):
                draft.validate_call({**good, extra:{}}, draft.no_edit_tool)
        for tool in (draft.submit_tool, draft.no_edit_tool):
            self.assertFalse(set(tool['input_schema']) & {'not','oneOf','anyOf','allOf','enum','const'})
            Draft202012Validator.check_schema(tool['input_schema'])


class AuthorWorkflowTests(EditorCase):
    async def test_typed_edit_reaches_independent_review_and_bound_publication(self):
        from test_commulingo_editor import JOB, CURRENT, URL, BODY, candidate, submission, store_mock
        from commulingo.pipeline.editor import Editor
        from commulingo.pipeline.engine import Usage
        from commulingo.pipeline import workflow
        from commulingo.pipeline.stages import latest, write_request
        from commulingo.pipeline.patches import patch_hash
        async def author(**kwargs):
            schema = kwargs['tool']['input_schema']
            self.assertEqual(set(schema['properties']['fields']['properties']), {'body'})
            self.assertNotIn('draft_contract', kwargs['prompt'])
            await kwargs['read_wrap']('fetch_url', AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            await kwargs['handler'](submission(candidate()))
        async def reviewer(**kwargs):
            await kwargs['read_wrap']('fetch_url', None)(url=URL)
            await kwargs['handler']({'decision':'approve','reason':'Independent original verifies the changed facts.',
                'resolved_risks':[], 'checks':[{'citation_id':'S1','passages':['P1'], 'finding':'The explanation matches the original.'}],
                'required_corrections':[], 'coverage':{'sufficient':True,'reason':'The commissioned topic is adequately covered.'}, 'optional_suggestions':[]})
        with patch('commulingo.pipeline.service.call', return_value=CURRENT), \
             patch('commulingo.pipeline.stages.model_call', side_effect=author):
            result = await Editor(store_mock())(JOB, [], Usage(), .2)
        artifacts = [{'stage':'research', 'value':result.value}]
        reads = {name:AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>')
                 for name in ('wiki_search','wiki_get','web_search','fetch_url','commulingo_people')}
        with patch('commulingo.pipeline.service.call', return_value=CURRENT), \
             patch('runtime_tools.registry.TOOL_HANDLERS', reads), \
             patch('commulingo.review_handlers.review_risks', return_value=[]), \
             patch('commulingo.pipeline.stages.model_call', side_effect=reviewer):
            review = await workflow.Review()(JOB, artifacts, Usage(), .2)
        artifacts.append({'stage':'review', 'value':review.value})
        with patch('commulingo.pipeline.config.load', return_value={'phase':'live'}), \
             patch('commulingo.pipeline.service.call', return_value={'status':'approved','suggestionId':123}) as rpc:
            await workflow.publish(JOB, artifacts, Usage(), .2)
        request = rpc.call_args.args[0]
        self.assertEqual(request['approvedPatchHash'], patch_hash(write_request(JOB, latest(artifacts, 'draft'))))
        self.assertEqual(request['fields']['expectedRevision'], CURRENT['revision'])
        self.assertTrue(request['fields']['evidence'])

    async def test_no_edit_is_a_terminal_through_the_real_model_dispatcher(self):
        from commulingo.pipeline.stages import model_call
        from commulingo.pipeline.prompts import spec
        from commulingo.pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        draft = session()
        handler = AsyncMock()
        no_edit = AsyncMock(return_value='OK: no-edit judgment')
        async def chat(messages, **kwargs):
            name = draft.no_edit_tool['name']
            self.assertIn(name, kwargs['terminal_tools'])
            self.assertIn(name, kwargs['finalization_tools'])
            value = {'status':'sources_unavailable', 'reason':'No original is available for this issue.',
                     'issues':{'missing:bio':{'status':'deferred', 'reason':'No accessible original source.'}}, 'notes':None}
            wire = next(t for t in kwargs['tools'] if t['name']==name)
            self.assertTrue(wire['strict'])
            with patch('tool_gateway.security.audit'):
                response, failed = await execute_tool(name, value, kwargs['tool_handlers'],
                                                       tool_schema=wire)
            self.assertFalse(failed, response)
            kwargs['budget_tracker']['total_cost'] = 0
        binding = SimpleNamespace(chat=chat, client=None, model='fixture', render_provider='openai', reasoning={})
        with patch('bot_config.resolve_agent_tool_loop', return_value=binding):
            await model_call(spec=spec('research'), prompt='Fixture commission.', tool=draft.submit_tool,
                handler=handler, reads=set(), usage=Usage(), budget=.2,
                local_tools=[(draft.no_edit_tool, no_edit, True)])
        no_edit.assert_awaited_once()
        self.assertNotIn('notes', no_edit.call_args.kwargs)
        handler.assert_not_awaited()

    async def test_saved_partial_submission_does_not_finish_terminal_stage(self):
        from commulingo.pipeline.stages import StageContinues, model_call
        from commulingo.pipeline.prompts import spec
        from commulingo.pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        draft = session()
        calls = 0
        async def save(value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise StageContinues('Saved incomplete draft; send fields.bio.en next')
            return 'OK: validated draft'
        async def chat(messages, **kwargs):
            name = draft.submit_tool['name']
            with patch('tool_gateway.security.audit'):
                progress, pending = await execute_tool(name, {'reason':'first'},
                    kwargs['tool_handlers'], tool_schema=draft.submit_tool)
                done, finished = await execute_tool(name, {'reason':'second'},
                    kwargs['tool_handlers'], tool_schema=draft.submit_tool)
            self.assertTrue(pending)
            self.assertIn('Saved incomplete draft', progress)
            self.assertFalse(finished)
            self.assertEqual(done, 'OK: validated draft')
            kwargs['budget_tracker']['total_cost'] = 0
        binding = SimpleNamespace(chat=chat, client=None, model='fixture', render_provider='deepseek', reasoning={})
        usage = Usage()
        with patch('bot_config.resolve_agent_tool_loop', return_value=binding):
            await model_call(spec=spec('research'), prompt='Fixture commission.', tool=draft.submit_tool,
                handler=save, reads=set(), usage=usage, budget=.2)
        self.assertEqual(calls,2)
        self.assertEqual(usage.tracker['partial_submissions'],1)

    async def test_unfinished_stage_reports_saved_partial_count(self):
        from commulingo.pipeline.stages import StageContinues, model_call
        from commulingo.pipeline.prompts import spec
        from commulingo.pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        draft = session()
        async def save(value):
            raise StageContinues('Saved incomplete draft')
        async def chat(messages, **kwargs):
            with patch('tool_gateway.security.audit'):
                _, pending = await execute_tool(draft.submit_tool['name'], {'reason':'first'},
                    kwargs['tool_handlers'], tool_schema=draft.submit_tool)
            self.assertTrue(pending)
            kwargs['budget_tracker']['total_cost'] = 0
        binding = SimpleNamespace(chat=chat, client=None, model='fixture', render_provider='deepseek', reasoning={})
        with patch('bot_config.resolve_agent_tool_loop', return_value=binding):
            with self.assertRaisesRegex(RuntimeError, 'saved partial submissions: 1'):
                await model_call(spec=spec('research'), prompt='Fixture commission.', tool=draft.submit_tool,
                    handler=save, reads=set(), usage=Usage(), budget=.2)

    async def test_no_edit_tool_uses_the_existing_owner_and_caller_boundary(self):
        from security_gateway import CallerContext, authorize, policy
        with patch.object(policy, 'enforce_mode', return_value='enforce'):
            for owner, agent in ((False,'commulingo_curator'), (True,'roleplay'), (True,'commulingo_curator')):
                decision = authorize(CallerContext(interface='autonomous', agent_name=agent, is_owner=owner),
                                     'commulingo_pipeline_no_edit', {}, consume_rate_limit=False)
                self.assertEqual(decision.allowed, owner and agent == 'commulingo_curator')
