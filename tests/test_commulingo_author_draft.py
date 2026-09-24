from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import unittest

from jsonschema import Draft202012Validator

from commulingo_pipeline.author_draft import AuthorDraft, obj
from commulingo_pipeline.draft_repair import RepairProtocolError
from commulingo_test_support import EditorCase


def session():
    fields = obj({'bio': obj({'ko': {'type':'string', 'maxLength':20},
                              'en': {'type':'string', 'maxLength':40}}, ['ko', 'en']),
                  'years': {'type':'string'}})
    canonical = obj({'status': {'type':'string'}, 'reason': {'type':'string'},
                     'fields': fields, 'claims': {'type':'array'}, 'issue_results': {'type':'array'},
                     'notes': {'type':'string'}}, ['status', 'reason'])
    draft = AuthorDraft({'name':'commulingo_pipeline_result', 'input_schema':canonical}, capture_invalid=True)
    draft.configure(fields, [{'id':'missing:bio'}])
    return draft


def edit():
    return {'changes': {
                'bio': {'value': {'ko':'검증한 인물이다.', 'en':'A documented person.'},
                        'evidence':[{'claim':'A documented person.', 'passages':['P1']}]},
                'years': {'value':'1900–1980',
                          'evidence':[{'claim':'The life dates.', 'passages':['P2']}]}},
            'issues': {'missing:bio': {'status':'resolved', 'reason':'Added the supported biography.'}},
            'reason':'The archive establishes the commissioned facts.'}


class AuthorDraftTests(unittest.TestCase):
    def test_field_update_replaces_value_and_evidence_atomically(self):
        draft = session()
        draft.prepare(draft.submission(edit()))
        patch = {'changes': {'bio': {'value': {'ko':'자료로 확인한 인물이다.', 'en':'A revised biography.'},
                                     'evidence':[{'claim':'A revised claim.', 'passages':['P3']}]}}}
        result = draft.prepare(draft.submission(patch, update=True))
        self.assertEqual(result['fields']['years'], '1900–1980')
        self.assertEqual([(c['field'], c['passages']) for c in result['claims']], [('years',['P2']), ('bio',['P3'])])
        self.assertEqual(result['issue_results'][0]['id'], 'missing:bio')
        self.assertEqual(draft.view()['changes']['bio'], patch['changes']['bio'])

    def test_invalid_typed_update_never_corrupts_saved_draft(self):
        draft = session()
        draft.prepare(draft.submission(edit()))
        saved = deepcopy(draft.draft)
        for patch in ({'changes':{'bio':{'value':'wrong type', 'evidence':[]}}},
                      {'changes':{'bio':{'value':{'ko':'문장', 'en':'Text'}}}},
                      {'repairs':[{'path':'/fields/bio', 'op':'remove'}]},
                      {}, {'changes':{}}, {'issues':{}}):
            with self.subTest(patch=patch), self.assertRaises(RepairProtocolError):
                draft.submission(patch, update=True)
            self.assertEqual(draft.draft, saved)

    def test_overlength_value_is_saved_and_repaired_as_one_field(self):
        draft = session()
        value = edit()
        value['changes']['bio']['value']['en'] = 'x' * 41
        self.assertFalse(list(Draft202012Validator(draft.submit_tool['input_schema']).iter_errors(value)))
        with self.assertRaisesRegex(ValueError, 'changes.bio.value.en'):
            draft.prepare(draft.submission(value))
        self.assertEqual(draft.view()['changes']['bio']['value']['en'], 'x' * 41)
        result = draft.prepare(draft.submission({'changes':{'bio':edit()['changes']['bio']}}, update=True))
        self.assertEqual(result['fields']['bio']['en'], 'A documented person.')

    def test_withdrawal_removes_only_draft_field_and_its_evidence(self):
        draft = session()
        draft.prepare(draft.submission(edit()))
        result = draft.prepare(draft.submission({'remove_fields':['years']}, update=True))
        self.assertNotIn('years', result['fields'])
        self.assertEqual([c['field'] for c in result['claims']], ['bio'])
        with self.assertRaisesRegex(RepairProtocolError, 'same call'):
            draft.submission({'changes':{'bio':edit()['changes']['bio']}, 'remove_fields':['bio']}, update=True)

    def test_outcomes_are_explicit_and_ids_cannot_be_invented(self):
        draft = session()
        for outcomes in ({}, {'invented': {'status':'resolved','reason':'Claiming completion is insufficient.'}}):
            value = edit(); value['issues'] = outcomes
            with self.assertRaises(RepairProtocolError):
                draft.submission(value)
        draft.prepare(draft.submission(edit()))
        result = draft.prepare(draft.submission({'issues':{'missing:bio':{
            'status':'deferred','reason':'A conflicting source needs checking.'}}}, update=True))
        self.assertEqual(result['issue_results'][0]['status'], 'deferred')
        self.assertEqual(result['fields']['bio'], edit()['changes']['bio']['value'])

    def test_old_checkpoint_is_read_and_repaired_without_label_reassignment(self):
        draft = session()
        draft.draft = {'tool':'commulingo_pipeline_result', 'args':{
            'status':'ready', 'reason':'Previously saved canonical draft.',
            'fields':{'bio':{'ko':'이전 초안', 'en':'Prior draft'}, 'years':'1900–1980'},
            'claims':[{'field':'years','claim':'Life dates','passages':['P18']},
                      {'field':'bio','claim':'Previous claim','passages':['P27']}],
            'issue_results':[{'id':'missing:bio','status':'resolved','reason':'A previously supported explanation.'}]}}
        self.assertEqual(draft.view()['changes']['bio']['evidence'][0]['passages'], ['P27'])
        result = draft.prepare(draft.submission({'changes':{'bio':edit()['changes']['bio']}}, update=True))
        self.assertEqual(result['claims'][0]['passages'], ['P18'])
        self.assertEqual(draft.author_error('/claims/1/passages'), '/changes/bio/evidence/0/passages')

    def test_malformed_legacy_containers_are_retained_until_explicit_replacement(self):
        draft = session()
        draft.draft = {'tool':draft.name, 'args':{'fields':['malformed legacy value'],
            'claims':'bad container', 'issue_results':None, 'reason':'Saved before typed intake.'}}
        saved = deepcopy(draft.draft)
        self.assertTrue(draft.view()['needs_full_submission'])
        with self.assertRaisesRegex(RepairProtocolError, 'complete replacement'):
            draft.submission({'changes':{'bio':edit()['changes']['bio']}}, update=True)
        self.assertEqual(draft.draft, saved)
        draft.prepare(draft.submission(edit()))
        self.assertEqual(draft.view()['changes']['bio'], edit()['changes']['bio'])

    def test_no_edit_has_no_content_or_pointer_protocol(self):
        draft = session()
        good = {'status':'sources_unavailable','reason':'The original could not be retrieved.',
                'issues':{'missing:bio':{'status':'deferred','reason':'No accessible supporting original.'}}}
        draft.validate_call(good, draft.no_edit_tool)
        for extra in ('fields', 'changes', 'claims', 'repairs'):
            with self.assertRaises(RepairProtocolError):
                draft.validate_call({**good, extra:{}}, draft.no_edit_tool)
        for tool in (draft.submit_tool, draft.update_tool, draft.no_edit_tool):
            self.assertFalse(set(tool['input_schema']) & {'not','oneOf','anyOf','allOf','enum','const'})
            Draft202012Validator.check_schema(tool['input_schema'])


class AuthorWorkflowTests(EditorCase):
    async def test_typed_edit_reaches_independent_review_and_bound_publication(self):
        from test_commulingo_editor import JOB, CURRENT, URL, BODY, candidate, submission, store_mock
        from commulingo_pipeline.editor import Editor
        from commulingo_pipeline.engine import Usage
        from commulingo_pipeline import workflow
        from commulingo_pipeline.stages import latest, write_request
        from commulingo_pipeline.patches import patch_hash
        async def author(**kwargs):
            schema = kwargs['tool']['input_schema']
            self.assertEqual(set(schema['properties']['changes']['properties']), {'body'})
            self.assertNotIn('draft_contract', kwargs['prompt'])
            await kwargs['read_wrap']('fetch_url', AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            await kwargs['handler'](submission(candidate()))
        async def reviewer(**kwargs):
            await kwargs['read_wrap']('fetch_url', None)(url=URL)
            await kwargs['handler']({'decision':'approve','reason':'Independent original verifies the changed facts.',
                'resolved_risks':[], 'checks':[{'citation_id':'S1','passages':['P1'], 'finding':'The explanation matches the original.'}],
                'required_corrections':[], 'optional_suggestions':[]})
        with patch('commulingo_pipeline.service.call', return_value=CURRENT), \
             patch('commulingo_pipeline.stages.model_call', side_effect=author):
            result = await Editor(store_mock())(JOB, [], Usage(), .2)
        artifacts = [{'stage':'research', 'value':result.value}]
        reads = {name:AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>')
                 for name in ('wiki_search','wiki_get','web_search','fetch_url','commulingo_people')}
        with patch('commulingo_pipeline.service.call', return_value=CURRENT), \
             patch('runtime_tools.registry.TOOL_HANDLERS', reads), \
             patch('runtime_tools.commulingo_review_handlers.review_risks', return_value=[]), \
             patch('commulingo_pipeline.stages.model_call', side_effect=reviewer):
            review = await workflow.Review()(JOB, artifacts, Usage(), .2)
        artifacts.append({'stage':'review', 'value':review.value})
        with patch('commulingo_pipeline.config.load', return_value={'phase':'live'}), \
             patch('commulingo_pipeline.service.call', return_value={'status':'approved','suggestionId':123}) as rpc:
            await workflow.publish(JOB, artifacts, Usage(), .2)
        request = rpc.call_args.args[0]
        self.assertEqual(request['approvedPatchHash'], patch_hash(write_request(JOB, latest(artifacts, 'draft'))))
        self.assertEqual(request['fields']['expectedRevision'], CURRENT['revision'])
        self.assertTrue(request['fields']['evidence'])

    async def test_no_edit_is_a_terminal_through_the_real_model_dispatcher(self):
        from commulingo_pipeline.stages import model_call
        from commulingo_pipeline.prompts import spec
        from commulingo_pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        draft = session()
        handler = AsyncMock()
        no_edit = AsyncMock(return_value='OK: no-edit judgment')
        async def chat(messages, **kwargs):
            name = draft.no_edit_tool['name']
            self.assertIn(name, kwargs['terminal_tools'])
            self.assertIn(name, kwargs['finalization_tools'])
            value = {'status':'sources_unavailable', 'reason':'No original is available for this issue.',
                     'issues':{'missing:bio':{'status':'deferred', 'reason':'No accessible original source.'}}}
            with patch('tool_gateway.security.audit'):
                response, failed = await execute_tool(name, value, kwargs['tool_handlers'],
                                                       tool_schema=draft.no_edit_tool)
            self.assertFalse(failed, response)
            kwargs['budget_tracker']['total_cost'] = 0
        binding = SimpleNamespace(chat=chat, client=None, model='fixture', render_provider='deepseek', reasoning={})
        with patch('bot_config.resolve_agent_tool_loop', return_value=binding):
            await model_call(spec=spec('research'), prompt='Fixture commission.', tool=draft.submit_tool,
                handler=handler, reads=set(), usage=Usage(), budget=.2,
                local_tools=[(draft.no_edit_tool, no_edit, True)])
        no_edit.assert_awaited_once()
        handler.assert_not_awaited()

    async def test_no_edit_tool_uses_the_existing_owner_and_caller_boundary(self):
        from security_gateway import CallerContext, authorize, policy
        with patch.object(policy, 'enforce_mode', return_value='enforce'):
            for owner, agent in ((False,'commulingo_curator'), (True,'roleplay'), (True,'commulingo_curator')):
                decision = authorize(CallerContext(interface='autonomous', agent_name=agent, is_owner=owner),
                                     'commulingo_pipeline_no_edit', {}, consume_rate_limit=False)
                self.assertEqual(decision.allowed, owner and agent == 'commulingo_curator')
