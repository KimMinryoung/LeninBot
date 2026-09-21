from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, Mock, patch

from commulingo_test_support import EditorCase, citation_result

from commulingo_pipeline.editor import Editor
from commulingo_pipeline.engine import Usage
from commulingo_pipeline.evidence import snapshot
from commulingo_pipeline.issues import commission
from commulingo_pipeline.patches import changes, patch_hash
from commulingo_pipeline.source_session import Sources
from commulingo_pipeline import workflow
from commulingo_pipeline.stages import latest, write_request

URL = 'https://example.org/archive'
BODY = 'The original historical account documents the definition and its context.'
CURRENT = {'id':'fixture','revision':'v1-original','body':{'ko':'','en':''},'evidence':[]}
JOB = {'id':101,'kind':'term','action':'update','target':'fixture','topic':'history',
       'reason':'Commissioned glossary explanation: history','stage':'research','lease_token':'fixture'}


def candidate():
    return {'status':'ready','reason':'The retrieved original text supports the missing explanation.',
            'fields':{'body':{'ko':'근거로 확인한 역사적 맥락이다.','en':'A documented historical context.'}},
            'claims':[{'field':'body','claim':'Historical context is documented.','passages':['P1']}],
            'issue_results':[{'id':'missing:body','status':'resolved','reason':'Supplied both languages with original evidence.'}]}


def store_mock():
    store = Mock()
    store.job_sources.return_value = {}
    store.cached_source.return_value = None
    return store


class SourceAndIssueTests(EditorCase):
    async def test_unlabelled_cached_page_can_be_opened_and_restored(self):
        page = snapshot(URL, BODY)
        store = store_mock()
        store.job_sources.return_value = {page['id']:page}
        session = await Sources.load(store, JOB, Usage())
        checkpoint = {}
        async def save():
            checkpoint['passages'] = deepcopy(session.passages.shown)
        tool, read, _ = session.cached_tool(on_read=save)
        from jsonschema import validate, ValidationError
        validate({'source_id':page['id']}, tool['input_schema'])
        validate({'passages':['P1']}, tool['input_schema'])
        validate({}, tool['input_schema'])
        for args in ({'source_id':page['id'],'passages':['P1']},):
            with self.assertRaises(ValidationError):
                validate(args, tool['input_schema'])
        with self.assertRaisesRegex(ValueError, 'source_id.*Never guess P1'):
            await read(passages=['P1'])
        fetched_at, expires_at = page['fetched_at'], page['expires_at']
        self.assertIn('[P1]', await read(source_id=page['id']))
        restored = await Sources.load(store, JOB, Usage(), checkpoint)
        _, reread, _ = restored.cached_tool()
        self.assertIn(BODY, await reread(passages=['P1']))
        self.assertEqual(restored.passages.shown, session.passages.shown)
        self.assertEqual((page['fetched_at'],page['expires_at']), (fetched_at,expires_at))
        page['expires_at'] = datetime.now(timezone.utc)-timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, 'expired'):
            await reread(source_id=page['id'])

    async def test_cache_discovery_tracks_new_pages_and_excludes_unusable_sources(self):
        page = snapshot(URL, BODY)
        expired = snapshot(URL + '/expired', BODY + ' Old.')
        expired['expires_at'] = datetime.now(timezone.utc)-timedelta(seconds=1)
        session = Sources(store_mock(), JOB, Usage(), {expired['id']:expired})
        _, read, _ = session.cached_tool()
        self.assertEqual(json.loads(await read())['available_pages'], [])
        shown = session.display(page)
        self.assertIn(page['id'], shown)
        before = deepcopy(session.passages.shown)
        catalog = json.loads(await read())['available_pages']
        self.assertEqual([p['source_id'] for p in catalog], [page['id']])
        self.assertEqual(catalog[0]['passage_labels'], ['P1'])
        with self.assertRaises(ValueError) as error:
            await read(source_id='S1')
        self.assertIn(page['id'], str(error.exception))
        self.assertEqual(session.passages.shown, before)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            await read(source_id=page['id'], passages=['P1'])

    async def test_cached_passage_read_keeps_labels_and_rejects_expiry(self):
        page = snapshot(URL,BODY)
        session = Sources(store_mock(),JOB,Usage(),{page['id']:page})
        session.display(page)
        before = deepcopy(session.passages.shown)
        _, read, terminal = session.cached_tool()
        self.assertFalse(terminal)
        self.assertIn('[P1]',await read(passages=['P1']))
        self.assertIn(BODY,await read(passages=['P1']))
        self.assertEqual(before,session.passages.shown)
        with self.assertRaisesRegex(ValueError,'unknown'):
            await read(passages=['P99'])
        page['expires_at'] = datetime.now(timezone.utc)-timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError,'expired'):
            await read(passages=['P1'])

    def test_prose_diagnostics_locate_exact_field_and_keep_quoted_titles(self):
        from commulingo_pipeline.diagnostics import prose_errors
        errors = json.loads(prose_errors({'body':{'en':'A clause — another clause.',
            'ko':'「스페인의 교훈 — 마지막 경고」'}}))
        self.assertEqual([e['path'] for e in errors],['/fields/body/en'])
    def test_commissions_are_defects_not_prose_quotas(self):
        self.assertEqual([i['id'] for i in commission(JOB,CURRENT)], ['missing:body'])
        full = {**CURRENT,'body':{'ko':'본문','en':'Body'},'evidence':[{'field':'body'}]}
        self.assertEqual(commission(JOB,full), [])
        self.assertEqual(commission({**JOB,'topic':'examples'},full), [])
        person = {**JOB,'kind':'person','topic':'sections','reason':'Commissioned missing information or evidence: sections'}
        self.assertEqual(commission(person,{'sections':[]}), [])
        explicit = {**person,'reason':'Add the documented 1930 trial section.'}
        self.assertEqual(commission(explicit,{})[0]['id'],'requested')
        # Missing provenance on existing prose is not a commission (2026-09-20).
        self.assertEqual(commission(JOB,{**full,'evidence':[]}), [])
        complete_person = {'years':'1900–1980','epithet':{'ko':'역사가','en':'Historian'},
                           'role':{'category':'scholar'},'career':[{'y':'1930'}],'evidence':[{'field':'years'}]}
        self.assertEqual(commission({**person,'topic':'basics'},complete_person), [])

    async def test_same_url_pages_are_cached_individually_without_merging(self):
        store, usage = store_mock(), Usage()
        session = await Sources.load(store, JOB, usage)
        call = AsyncMock(side_effect=[f'<external source="web">\n{BODY}\n</external>',
                                      '<external source="web">\nA different independently stored page.\n</external>'])
        fetch = session.wrap('fetch_url',call)
        first = await fetch(url=URL,offset=0)
        second = await fetch(url=URL,offset=100)
        self.assertIn('[P1]',first); self.assertIn('[P2]',second)
        self.assertEqual(len(session.sources),2)
        self.assertEqual({s['body'] for s in session.sources.values()}, {BODY,'A different independently stored page.'})
        self.assertEqual(store.save_source.call_count,2)
        self.assertEqual(store.link_source.call_count,2)
        cached = next(iter(session.sources.values()))
        store.cached_source.return_value = cached
        await fetch(url=URL,offset=0)
        self.assertEqual(call.await_count,2)
        self.assertEqual(usage.tracker['pipeline_cache_hits'],1)
        self.assertEqual(store.save_source.call_count,2)

    async def test_cached_page_does_not_get_a_new_acquisition_time(self):
        store = store_mock()
        page = snapshot(URL,BODY,now=datetime.now(timezone.utc)-timedelta(days=10))
        store.cached_source.return_value = page
        session = await Sources.load(store,JOB,Usage())
        call = AsyncMock()
        shown = await session.wrap('fetch_url',call)(url=URL)
        self.assertIn(str(page['fetched_at']),shown)
        call.assert_not_awaited(); store.save_source.assert_not_called()

    def test_delta_and_hash_bind_actual_patch_and_revision(self):
        patch_data = {'target':'term','action':'update','id':'fixture','fields':{'body':{'ko':'변경','en':'Same'},'expectedRevision':'r1'},'sources':[URL]}
        self.assertEqual(changes({'body':{'ko':'이전','en':'Same'}},patch_data['fields']),
                         [{'path':'/body/ko','before':'이전','after':'변경'}])
        before = patch_hash(patch_data)
        patch_data['fields']['expectedRevision'] = 'r2'
        self.assertNotEqual(patch_hash(patch_data), before)


class EditorTests(EditorCase):
    async def test_format_state_survives_restart_and_research_reopen_is_saved(self):
        store = store_mock()
        async def first(**kwargs):
            self.assertIn('"draft_saved": false', kwargs['prompt'])
            await kwargs['read_wrap']('fetch_url', AsyncMock(
                return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate()
            value['fields']['body']['en'] = 'A clause — another clause.'
            with self.assertRaises(ValueError) as failure:
                await kwargs['handler'](value)
            state = json.loads(str(failure.exception).splitlines()[-1])['work_status']
            self.assertEqual(state['mode'], 'format_repair')
            self.assertTrue(state['draft_saved'])
            self.assertEqual(state['saved_fields'], ['body'])
            self.assertEqual(state['next_tool'], 'commulingo_pipeline_repair')
            raise RuntimeError('disconnect')
        with patch('commulingo_pipeline.service.call', return_value=CURRENT), \
             patch('commulingo_pipeline.stages.model_call', side_effect=first):
            with self.assertRaisesRegex(RuntimeError, 'disconnect'):
                await Editor(store)(JOB, [], Usage(), .2)
        checkpoint = deepcopy(store.save_editor_checkpoint.call_args.args[1])
        page = store.save_source.call_args.args[0]
        store.job_sources.return_value = {page['id']:page}
        async def resume(**kwargs):
            self.assertIn('"mode": "format_repair"', kwargs['prompt'])
            self.assertIn('"draft_saved": true', kwargs['prompt'])
            call = AsyncMock()
            from tool_gateway.results import ToolRejection
            with self.assertRaises(ToolRejection) as blocked:
                await kwargs['read_wrap']('web_search', call)(query='date')
            call.assert_not_awaited()
            self.assertIn('"mode": "format_repair"', str(blocked.exception))
            read = next(t[1] for t in kwargs['local_tools']
                        if t[0]['name']=='commulingo_pipeline_cached_passages')
            await read(source_id=page['id'])
            self.assertEqual(store.save_editor_checkpoint.call_args.args[1]['error'], checkpoint['error'])
            reopen = next(t[1] for t in kwargs['local_tools']
                          if t[0]['name']=='commulingo_pipeline_research')
            response = await reopen(fields=['body'], reason='Check a conflicting historical fact.')
            self.assertIn('"mode": "research_allowed"', response)
            self.assertFalse(store.save_editor_checkpoint.call_args.args[1]['repair_only'])
            await kwargs['handler']({'repairs':[{'op':'set','path':'/fields/body/en',
                                               'value':candidate()['fields']['body']['en']}]})
        with patch('commulingo_pipeline.service.call', return_value=CURRENT), \
             patch('commulingo_pipeline.stages.model_call', side_effect=resume):
            result = await Editor(store)(JOB, [{'stage':'editor_checkpoint','value':checkpoint}], Usage(), .2)
        self.assertEqual(result.next_stage, 'review')

    async def test_cached_source_labels_are_checkpointed_before_first_draft(self):
        store = store_mock()
        page = snapshot(URL, BODY)
        store.job_sources.return_value = {page['id']:page}
        async def model(**kwargs):
            read = next(handler for tool,handler,_ in kwargs['local_tools']
                        if tool['name']=='commulingo_pipeline_cached_passages')
            await read(source_id=page['id'])
            saved = store.save_editor_checkpoint.call_args.args[1]
            self.assertIsNone(saved['draft'])
            self.assertIn('P1', saved['passages'])
            await kwargs['handler'](candidate())
        with patch('commulingo_pipeline.service.call', return_value=CURRENT), \
             patch('commulingo_pipeline.stages.model_call', side_effect=model):
            result = await Editor(store)(JOB, [], Usage(), .2)
        self.assertEqual(result.next_stage, 'review')

    async def test_citation_rejection_preserves_draft_and_requires_supported_repair(self):
        store, usage = store_mock(), Usage()
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url', AsyncMock(
                return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            self.jev.side_effect = lambda *args: citation_result(*args, support='unrelated')
            value = candidate()
            value['claims'][0]['claim'] = 'An unrelated invented claim.'
            with self.assertRaisesRegex(ValueError, 'citation check failed') as failure:
                await kwargs['handler'](value)
            state = json.loads(str(failure.exception).splitlines()[-1])['work_status']
            self.assertEqual(state['error_kind'], 'citation')
            self.assertEqual(state['mode'], 'research_allowed')
            self.assertEqual(state['next_tool'], 'commulingo_pipeline_cached_passages')
            self.assertEqual(state['submission_tool'], 'commulingo_pipeline_repair')
            self.assertEqual(state['saved_claim_count'], 1)
            self.assertEqual(store.save_editor_checkpoint.call_args.args[1]['draft']['args']['fields'], value['fields'])
            self.jev.side_effect = citation_result
            await kwargs['handler']({'repairs': [{'op': 'set', 'path': '/claims/0/claim',
                                                  'value': candidate()['claims'][0]['claim']}]})
        with patch('commulingo_pipeline.service.call', return_value=CURRENT) as rpc, \
             patch('commulingo_pipeline.stages.model_call', side_effect=model):
            result = await Editor(store)(JOB, [], usage, .2)
        self.assertEqual(result.next_stage, 'review')
        self.assertEqual([call.args[0]['command'] for call in rpc.call_args_list], ['read', 'validate'])
        self.assertEqual(usage.tracker['citation_rejections'], 1)
        self.assertEqual(usage.tracker['jev_calls'], 2)
        self.assertEqual(result.value['research']['claims'][0]['citation_check']['support'], 'supports')

    async def test_classification_outage_retains_new_term_draft_without_publication(self):
        store = store_mock()
        job = {**JOB, 'action': 'create'}
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url', AsyncMock(
                return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            await kwargs['handler']({
                'status': 'ready', 'reason': 'The source supports the new term definition.',
                'fields': {'term': {'ko': '검증 용어', 'en': 'Fixture term'},
                           'definition': {'ko': '검증한 개념이다.', 'en': 'A documented concept.'},
                           'aliases': {'ko': [], 'en': []},
                           'period': {'ko': '개념', 'en': 'Concept'}},
                'claims': [{'field': field, 'claim': 'This field is documented.', 'passages': ['P1']}
                           for field in ('definition', 'period')],
                'issue_results': [{'id': 'register', 'status': 'resolved', 'reason': 'Supported bilingual entry.'}],
            })
        with patch('commulingo_pipeline.service.call', return_value=None) as rpc, \
             patch('runtime_tools.commulingo_classify.classify_term', return_value=None), \
             patch('commulingo_pipeline.stages.model_call', side_effect=model):
            with self.assertRaisesRegex(RuntimeError, 'classification unavailable'):
                await Editor(store)(job, [], Usage(), .2)
        self.assertEqual([call.args[0]['command'] for call in rpc.call_args_list], ['read'])
        saved = store.save_editor_checkpoint.call_args.args[1]
        self.assertEqual(saved['draft']['args']['fields']['term']['en'], 'Fixture term')
        self.assertNotIn('category', saved['draft']['args']['fields'])

    async def test_context_restores_notes_sections_and_original_proposal(self):
        current = {**CURRENT,'notes':'Earlier author unresolved question',
                   'definition':{'ko':'기존 정의','en':'Existing definition to preserve'},
                   'sections':[{'slug':'prior-theme'}]}
        job = {**JOB,'payload':{'original_proposal':{'patch_json':{'slug':'fixed-theme'}}}}
        async def model(**kwargs):
            for text in ('Earlier author unresolved question','prior-theme','fixed-theme','prose_budgets'):
                self.assertIn(text,kwargs['prompt'])
            self.assertIn('"surrounding_context": {"definition":', kwargs['prompt'])
            self.assertIn('Existing definition to preserve', kwargs['prompt'])
            context_tool, read_context, _ = next(t for t in kwargs['local_tools']
                                                if t[0]['name']=='commulingo_pipeline_context')
            from jsonschema import validate
            fields = context_tool['input_schema']['properties']['fields']['items']['enum']
            self.assertGreater(len(fields), 10)
            validate({'fields':fields}, context_tool['input_schema'])
            self.assertIn('Earlier author unresolved question', await read_context(fields))
            self.assertIn('Do not use an em dash',kwargs['spec'].render_prompt(provider='deepseek'))
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['fields']['notes']='Private deferred detail'
            await kwargs['handler'](value)
        with patch('commulingo_pipeline.service.call',return_value=current), patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())(job,[],Usage(),.2)
        self.assertNotIn('notes',result.value['draft']['fields'])
        self.assertEqual(result.value['draft']['notes'],'Private deferred detail')

    async def test_create_person_and_term_use_jev_classification(self):
        from runtime_tools.commulingo_people import _COMMULINGO_FIELD_SCHEMA
        group = [{'id':'fixture-group'}]
        catalogs = (group, [], [{'id':'fixture-role'}])
        person_fields = {'givenName':{'ko':'검증','en':'Fixture'},'groupId':'fixture-group',
            'role':{'category':'fixture-role'},'epithet':{'ko':'역사가','en':'Historian'},
            'bio':{'ko':'자료로 확인한 인물이다.','en':'A documented historical person.'},'career':[],
            'citizenship':{'code':'france','label':{'ko':'프랑스','en':'France'}},
            'nationalOrigin':{'code':'france','label':{'ko':'프랑스','en':'France'}}}
        term_fields = {'term':{'ko':'검증 용어','en':'Fixture term'},
            'definition':{'ko':'자료로 확인한 개념이다.','en':'A documented concept.'},
            'aliases':{'ko':[],'en':[]},'period':{'ko':'역사','en':'History'},
            'category':_COMMULINGO_FIELD_SCHEMA['properties']['category']['enum'][0]}
        for kind, fields, facts in [('person',person_fields,['bio','citizenship','nationalOrigin']),
                                    ('term',term_fields,['definition','period'])]:
            with self.subTest(kind=kind):
                job = {**JOB,'kind':kind,'action':'create'}
                from commulingo_pipeline.decisions import Decisions
                fields = deepcopy(fields)
                Decisions(job,None,catalogs,Usage()).strip_assigned(fields)
                async def model(**kwargs):
                    await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
                    await kwargs['handler']({'status':'ready','reason':'The original archive supports this new entry.',
                        'fields':fields,'claims':[{'field':f,'claim':'The source documents this field.','passages':['P1']} for f in facts],
                        'issue_results':[{'id':'register','status':'resolved','reason':'Registered a supported bilingual entry.'}]})
                with patch('commulingo_pipeline.service.call',return_value=None), \
                     patch('runtime_tools.commulingo_classify.load_catalogs',return_value=catalogs), \
                     patch('runtime_tools.commulingo_classify.classify_person_card',return_value={'codes':{'citizenship':{'code':'france'},'nationalOrigin':{'code':'france'}},'person':{'groupId':'fixture-group','role':{'category':'fixture-role'}}}) as classifier, \
                     patch('runtime_tools.commulingo_classify.classify_term',return_value={'category':'culture'}), \
                     patch('commulingo_pipeline.stages.model_call',side_effect=model):
                    result = await Editor(store_mock())(job,[],Usage(),.2)
                self.assertEqual(result.next_stage,'review')
                self.assertEqual(classifier.call_count,1 if kind=='person' else 0)
                self.assertNotIn('expectedRevision',result.value['draft']['fields'])

    async def test_one_session_reads_writes_and_repairs_without_research_restart(self):
        store, usage = store_mock(), Usage()
        fetch = AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>')
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url',fetch)(url=URL)
            value = candidate(); value['claims'] = []
            with self.assertRaisesRegex(ValueError, 'evidence required for body'):
                await kwargs['handler'](value)
            await kwargs['handler']({'repairs':[{'op':'set','path':'/claims','value':candidate()['claims']}]})
        with patch('commulingo_pipeline.service.call',return_value=CURRENT) as rpc, \
             patch('commulingo_pipeline.stages.model_call',side_effect=model) as model_call:
            result = await Editor(store)(JOB,[],usage,.2)
        self.assertEqual(model_call.call_count,1)
        fetch.assert_awaited_once()
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(result.value['draft']['fields']['expectedRevision'],'v1-original')
        self.assertEqual(result.value['draft']['fields']['evidence'][0]['excerpt'],BODY)
        self.assertEqual(result.value['draft']['patch_hash'],patch_hash(write_request(JOB,result.value['draft'])))
        self.assertTrue(store.save_editor_checkpoint.called)
        self.assertEqual([c.args[0]['command'] for c in rpc.call_args_list],['read','validate'])
        artifacts = [{'stage':'research','value':result.value}]
        self.assertEqual(latest(artifacts,'draft'),result.value['draft'])
        self.assertEqual(latest(artifacts,'research')['baseline'],'v1-original')

    async def test_dispatcher_rejection_is_checkpointed_and_repaired_in_editor(self):
        from tool_gateway.dispatcher import execute_tool
        from tool_gateway.security import caller_scope, new_run_context
        from tool_gateway.results import ToolRejection
        store = store_mock()
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            async def terminal(**value):
                try:
                    return await kwargs['handler'](value)
                except ValueError as exc:
                    raise ToolRejection(str(exc)) from exc
            ctx = new_run_context(interface='autonomous',agent_name='commulingo_curator',is_owner=True,
                scope_type='maintenance_job',scope_id='commulingo_pipeline:editor-dispatch-test')
            value = candidate(); value['fields']['sections'] = [{'invalid':'unsupported'}]
            with caller_scope(ctx), patch('tool_gateway.security.audit'):
                response, failed = await execute_tool(kwargs['tool']['name'],value,
                    {kwargs['tool']['name']:terminal},tool_schema=kwargs['tool'])
                self.assertTrue(failed)
                self.assertIn('commulingo_pipeline_repair',response)
                saved = store.save_editor_checkpoint.call_args.args[1]
                self.assertEqual(saved['draft']['args']['claims'],candidate()['claims'])
                self.assertIn('sections',saved['draft']['args']['fields'])
                edit_tool = kwargs['local_tools'][0][0]
                response, failed = await execute_tool(edit_tool['name'],
                    {'repairs':[{'op':'remove','path':'/fields/sections'}]},
                    {edit_tool['name']:terminal},tool_schema=edit_tool)
                self.assertFalse(failed,response)
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result = await Editor(store)(JOB,[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')

    async def test_missing_outcomes_and_evidence_reported_together(self):
        async def model(**kwargs):
            value = candidate(); del value['issue_results']; del value['claims']
            with self.assertRaises(ValueError) as failure:
                await kwargs['handler'](value)
            self.assertIn('issue_results',str(failure.exception))
            self.assertIn('evidence required for body',str(failure.exception))
            raise RuntimeError('end test after feedback')
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), patch('commulingo_pipeline.stages.model_call',side_effect=model):
            with self.assertRaisesRegex(RuntimeError,'end test'):
                await Editor(store_mock())(JOB,[],Usage(),.2)

    async def test_repeated_malformed_repairs_do_not_hold_editor(self):
        store = store_mock()
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['claims'] = []
            with self.assertRaises(ValueError):
                await kwargs['handler'](value)
            edit = kwargs['local_tools'][0][1]
            for _ in range(3):
                with self.assertRaises(ValueError):
                    await edit(repairs=[{'op':'set','path':'/claims','value':candidate()['claims']}],reason='unexpected')
                with self.assertRaises(ValueError):
                    await kwargs['handler']({})
                self.assertEqual(store.save_editor_checkpoint.call_args.args[1]['draft']['args'], value)
            await edit(repairs=[{'op':'set','path':'/claims','value':candidate()['claims']}])
        usage = Usage()
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result = await Editor(store)(JOB,[],usage,.2)
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(usage.tracker['repair_protocol_errors'],6)

    async def test_unchanged_year_and_its_claim_are_removed_together(self):
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['fields']['startYear'] = 1923
            value['claims'].append({'field':'startYear','claim':'The year is 1923.','passages':['P1']})
            await kwargs['handler'](value)
        with patch('commulingo_pipeline.service.call',return_value={**CURRENT,'startYear':1923}), patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())(JOB,[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')
        self.assertNotIn('startYear',result.value['draft']['fields'])
        self.assertEqual([c['field'] for c in result.value['research']['claims']],['body'])

    async def test_crash_after_rejection_resumes_saved_patch_and_passage_ids(self):
        store = store_mock()
        async def first(**kwargs):
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['claims']=[]
            with self.assertRaises(ValueError):
                await kwargs['handler'](value)
            raise RuntimeError('simulated provider disconnect')
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), patch('commulingo_pipeline.stages.model_call',side_effect=first):
            with self.assertRaisesRegex(RuntimeError,'disconnect'):
                await Editor(store)(JOB,[],Usage(),.2)
        checkpoint = deepcopy(store.save_editor_checkpoint.call_args.args[1])
        page = store.save_source.call_args.args[0]
        store.job_sources.return_value = {page['id']:page}
        async def resume(**kwargs):
            self.assertIn('saved_draft',kwargs['prompt'])
            self.assertIn('"missing_evidence_fields": ["body"]', kwargs['prompt'])
            self.assertIn('"error_kind": "missing_evidence"', kwargs['prompt'])
            await kwargs['handler']({'repairs':[{'op':'set','path':'/claims','value':candidate()['claims']}]})
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), patch('commulingo_pipeline.stages.model_call',side_effect=resume):
            result = await Editor(store)(JOB,[{'stage':'editor_checkpoint','value':checkpoint}],Usage(),.2)
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(result.value['draft']['fields']['evidence'][0]['excerpt'],BODY)

    async def test_identical_failure_stops_instead_of_spending_more_rounds(self):
        store = store_mock()
        async def model(**kwargs):
            value = candidate(); value['claims']=[]
            with self.assertRaises(ValueError): await kwargs['handler'](value)
            await kwargs['handler'](value)
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result = await Editor(store)(JOB,[],Usage(),.2)
        self.assertEqual(result.status,'escalated')
        self.assertIn('without progress',result.value['hold_reason'])

    async def test_no_concrete_defect_needs_no_model_call(self):
        current = {**CURRENT,'body':{'ko':'본문','en':'Body'},'evidence':[{'field':'body'}]}
        with patch('commulingo_pipeline.service.call',return_value=current), patch('commulingo_pipeline.stages.model_call') as model:
            result = await Editor(store_mock())(JOB,[],Usage(),.2)
        self.assertEqual(result.next_stage,'judge'); model.assert_not_called()


class ReviewAndPublishTests(EditorCase):
    async def test_workflow_switch_preserves_old_submit_receipts_and_pins_new_authoring(self):
        store = Mock()
        legacy = {name:AsyncMock(return_value='legacy') for name in ('research','submit')}
        editor = {name:AsyncMock(return_value='editor') for name in legacy}
        with patch('commulingo_pipeline.workflow.stages',return_value=editor):
            routed = workflow.routed_stages(store,legacy,'editor')
        job = {**JOB,'payload':{}}
        self.assertEqual(await routed['submit'](job,[],Usage(),.2),'legacy')
        store.pin_editor_workflow.assert_not_called()
        self.assertEqual(await routed['research'](job,[],Usage(),.2),'editor')
        store.pin_editor_workflow.assert_called_once()
        self.assertEqual(job['payload']['workflow'],'editor')
        self.assertEqual(await routed['submit'](job,[],Usage(),.2),'editor')

    def artifacts(self):
        fields = candidate()['fields']
        fields['expectedRevision'] = CURRENT['revision']
        draft = {'fields':fields,'sources':[URL], 'issue_results':candidate()['issue_results']}
        return [{'stage':'research','value':{'editor_version':2,'research':{'baseline':CURRENT['revision']},'draft':draft}}]

    async def test_review_requires_factual_correction_and_approves_exact_patch(self):
        artifacts = self.artifacts()
        value = {'decision':'approve','reason':'The original archive verifies the changed facts.',
                 'resolved_risks':[], 'checks':[{'citation_id':'S1','passages':['P1'],'finding':'The source verifies the explanation.'}],
                 'required_corrections':[], 'optional_suggestions':['Could add a later example.']}
        async def model(**kwargs):
            fetch = kwargs['read_wrap']('fetch_url',None)
            await fetch(url=URL)
            with self.assertRaisesRegex(ValueError,'specific factual corrections'):
                await kwargs['handler']({**value,'decision':'revise'})
            await kwargs['handler'](value)
        reads = {name:AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>')
                 for name in ('wiki_search','wiki_get','web_search','fetch_url','commulingo_people')}
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), \
             patch('runtime_tools.registry.TOOL_HANDLERS',reads), \
             patch('scripts.commulingo_person_reviewer.review_risks',return_value=[]), \
             patch('commulingo_pipeline.stages.model_call',side_effect=model):
            result = await workflow.Review()(JOB,artifacts,Usage(),.2)
        self.assertEqual(result.next_stage,'submit')
        self.assertEqual(result.value['approved_patch_hash'],patch_hash(write_request(JOB,latest(artifacts,'draft'))))
        self.assertEqual(result.value['optional_suggestions'],value['optional_suggestions'])

    async def test_unchanged_rejected_patch_is_held_before_paid_review(self):
        artifacts = self.artifacts()
        digest = patch_hash(write_request(JOB,latest(artifacts,'draft')))
        artifacts.append({'stage':'review','value':{'decision':'revise','reviewed_patch_hash':digest}})
        with patch('commulingo_pipeline.service.call',return_value=CURRENT), patch('commulingo_pipeline.stages.model_call') as model:
            result = await workflow.Review()(JOB,artifacts,Usage(),.2)
        self.assertEqual(result.status,'escalated'); model.assert_not_called()

    async def test_publish_is_one_rpc_and_tampering_requires_new_review(self):
        artifacts = self.artifacts()
        digest = patch_hash(write_request(JOB,latest(artifacts,'draft')))
        decision = {'decision':'approve','approved_patch_hash':digest,'reason':'Independently verified.',
                    'checks':[{'citation':URL,'source':URL,'quote':BODY,'finding':'Verified.'}]}
        artifacts.append({'stage':'review','value':decision})
        with patch('commulingo_pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo_pipeline.service.call',return_value={'status':'approved','suggestionId':123}) as rpc:
            result = await workflow.publish(JOB,artifacts,Usage(),.2)
            rpc.assert_called_once()
            request = rpc.call_args.args[0]
            self.assertEqual(request['command'],'publish')
            self.assertEqual(request['approvedPatchHash'],digest)
            self.assertEqual(result.value['resolved_issues'],['missing:body'])
            rpc.reset_mock()
            artifacts[0]['value']['draft']['fields']['body']['ko']='승인 후 변경'
            result = await workflow.publish(JOB,artifacts,Usage(),.2)
            rpc.assert_not_called()
            self.assertEqual(result.next_stage,'review')
