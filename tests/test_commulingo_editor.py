from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, Mock, patch

from commulingo_test_support import EditorCase, citation_result

from commulingo.pipeline.editor import Editor
from commulingo.pipeline.engine import Usage
from commulingo.pipeline.evidence import snapshot
from commulingo.pipeline.issues import commission
from commulingo.pipeline.patches import changes, patch_hash
from commulingo.pipeline.source_session import Sources
from commulingo.pipeline import workflow
from commulingo.pipeline.stages import latest, write_request

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



def submission(value):
    """Express existing canonical test fixtures through the public author API."""
    result = {'changes': {field: {'value': content, 'evidence': [
                    {key: val for key, val in claim.items() if key != 'field'}
                    for claim in value.get('claims', []) if claim['field'] == field]}
                for field, content in value.get('fields', {}).items() if field != 'notes'},
              'reason': value['reason']}
    if 'issue_results' in value:
        result['issues'] = {item['id']: {'status': item['status'], 'reason': item['reason']}
                            for item in value['issue_results']}
    if 'notes' in value or 'notes' in value.get('fields', {}):
        result['notes'] = value.get('notes', value.get('fields', {}).get('notes'))
    return result


async def repair_call(kwargs, value):
    # A later submission to the same tool sends only the changes.
    return await kwargs['handler']({'changes': submission(value)['changes']})


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
        from jsonschema import validate
        validate({'source_id':page['id']}, tool['input_schema'])
        validate({'passages':['P1']}, tool['input_schema'])
        validate({}, tool['input_schema'])
        validate({'passages':[]}, tool['input_schema'])
        self.assertFalse(set(tool['input_schema']) & {'not','oneOf','anyOf','allOf','enum','const'})
        with self.assertRaisesRegex(ValueError, 'source_id.*Never guess P1'):
            await read(passages=['P1'])
        # With both arguments an unknown label opens the page instead of failing.
        self.assertIn('[P1]', await read(source_id=page['id'], passages=['P1']))
        self.assertIn('[P1]', await read(source_id=page['id'], passages=[]))
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
        self.assertEqual(await read(passages=[]), await read())
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
        both = await read(source_id=page['id'], passages=['P1'])
        self.assertIn(BODY, both)
        self.assertIn('source_id was ignored', both)

    async def test_oversized_passage_request_shows_first_batch_and_names_the_rest(self):
        page = snapshot(URL, BODY)
        session = Sources(store_mock(), JOB, Usage(), {page['id']:page})
        session.display(page)
        tool, read, _ = session.cached_tool()
        from jsonschema import validate
        labels = ['P1'] * 3 + [f'P{i}' for i in range(2, 12)]
        validate({'passages':labels}, tool['input_schema'])
        with patch.object(session.passages, 'shown', {**session.passages.shown,
                          **{f'P{i}': session.passages.shown['P1'] for i in range(2, 12)}}):
            text = await read(passages=labels)
        self.assertIn('Showing 8 of 11 labels', text)
        self.assertIn('P9, P10, P11', text)

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
        from commulingo.pipeline.diagnostics import prose_errors
        errors = json.loads(prose_errors({'body':{'en':'A clause — another clause.',
            'ko':'「스페인의 교훈 — 마지막 경고」'}}))
        self.assertEqual([e['path'] for e in errors],['/fields/body/en'])
    def test_commissions_are_defects_not_prose_quotas(self):
        self.assertEqual([i['id'] for i in commission(JOB,CURRENT)], ['missing:body'])
        full = {**CURRENT,'body':{'ko':'본문','en':'Body'},'evidence':[{'field':'body'}]}
        self.assertEqual(commission(JOB,full), [])
        self.assertEqual(commission({**JOB,'topic':'examples'},full), [])
        person = {**JOB,'kind':'person','topic':'sections','reason':'Commissioned missing information or evidence: sections'}
        self.assertEqual([(i['id'],i['field']) for i in commission(person,{'sections':[]})],
                         [('missing:sections','body')])
        self.assertEqual(commission(person,{'sections':[{'slug':'existing-theme'}]}), [])
        explicit = {**person,'reason':'Add the documented 1930 trial section.'}
        self.assertEqual({i['id'] for i in commission(explicit,{})}, {'missing:sections','requested'})
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
    async def test_no_edit_retains_rejected_draft_without_removing_content(self):
        store = store_mock()
        async def model(**kwargs):
            value = candidate()
            value['claims'] = []
            with self.assertRaisesRegex(ValueError, 'evidence required'):
                await kwargs['handler'](submission(value))
            saved = deepcopy(store.save_editor_checkpoint.call_args.args[1]['draft'])
            tool, no_edit, _ = next(t for t in kwargs['local_tools']
                                    if t[0]['name'] == 'commulingo_pipeline_no_edit')
            response = await no_edit(status='sources_unavailable',
                reason='No available original establishes the commissioned facts.',
                issues={'missing:body': {'status':'deferred', 'reason':'No reliable original could be retrieved.'}})
            self.assertIn('no-edit judgment recorded', response)
            self.assertEqual(store.save_editor_checkpoint.call_args.args[1]['draft'], saved)
        with patch('commulingo.pipeline.service.call', return_value=CURRENT) as rpc, \
             patch('commulingo.pipeline.stages.model_call', side_effect=model):
            result = await Editor(store)(JOB, [], Usage(), .2)
        self.assertEqual(result.next_stage, 'judge')
        self.assertNotIn('draft', result.value)
        self.assertEqual(latest([{'stage':'research', 'value':result.value}], 'research')['status'], 'sources_unavailable')
        self.assertEqual([call.args[0]['command'] for call in rpc.call_args_list], ['read'])

    async def test_format_state_survives_restart_and_research_reopen_is_saved(self):
        store = store_mock()
        async def first(**kwargs):
            self.assertIn('"draft_saved": false', kwargs['prompt'])
            await kwargs['read_wrap']('fetch_url', AsyncMock(
                return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate()
            value['fields']['body']['en'] = 'A clause — another clause.'
            with self.assertRaises(ValueError) as failure:
                await kwargs['handler'](submission(value))
            state = json.loads(str(failure.exception).splitlines()[-1])['work_status']
            self.assertEqual(state['mode'], 'format_repair')
            self.assertTrue(state['draft_saved'])
            self.assertEqual(state['saved_fields'], ['body'])
            self.assertEqual(state['next_tool'], 'commulingo_pipeline_submit_draft')
            raise RuntimeError('disconnect')
        with patch('commulingo.pipeline.service.call', return_value=CURRENT), \
             patch('commulingo.pipeline.stages.model_call', side_effect=first):
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
            await repair_call(kwargs, candidate())
        with patch('commulingo.pipeline.service.call', return_value=CURRENT), \
             patch('commulingo.pipeline.stages.model_call', side_effect=resume):
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
            await kwargs['handler'](submission(candidate()))
        with patch('commulingo.pipeline.service.call', return_value=CURRENT), \
             patch('commulingo.pipeline.stages.model_call', side_effect=model):
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
                await kwargs['handler'](submission(value))
            state = json.loads(str(failure.exception).splitlines()[-1])['work_status']
            self.assertEqual(state['error_kind'], 'citation')
            self.assertEqual(state['mode'], 'research_allowed')
            self.assertEqual(state['next_tool'], 'commulingo_pipeline_cached_passages')
            self.assertEqual(state['submission_tool'], 'commulingo_pipeline_submit_draft')
            self.assertEqual(state['saved_claim_count'], 1)
            self.assertEqual(store.save_editor_checkpoint.call_args.args[1]['draft']['args']['fields'], value['fields'])
            self.jev.side_effect = citation_result
            await repair_call(kwargs, candidate())
        with patch('commulingo.pipeline.service.call', return_value=CURRENT) as rpc, \
             patch('commulingo.pipeline.stages.model_call', side_effect=model):
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
            await kwargs['handler'](submission({
                'status': 'ready', 'reason': 'The source supports the new term definition.',
                'fields': {'term': {'ko': '검증 용어', 'en': 'Fixture term'},
                           'definition': {'ko': '검증한 개념이다.', 'en': 'A documented concept.'},
                           'aliases': {'ko': [], 'en': []},
                           'period': {'ko': '개념', 'en': 'Concept'}},
                'claims': [{'field': field, 'claim': 'This field is documented.', 'passages': ['P1']}
                           for field in ('definition', 'period')],
                'issue_results': [{'id': 'register', 'status': 'resolved', 'reason': 'Supported bilingual entry.'}],
            }))
        with patch('commulingo.pipeline.service.call', return_value=None) as rpc, \
             patch('commulingo.classify.classify_term', return_value=None), \
             patch('commulingo.pipeline.stages.model_call', side_effect=model):
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
            policy = kwargs['spec'].render_prompt(provider='deepseek')
            self.assertIn('Do not use an em dash', policy)
            self.assertNotIn('Finish this stage with commulingo_pipeline_result', policy)
            self.assertNotIn('Discovery proposes', policy)
            self.assertEqual(kwargs['prompt'].count('"missing_evidence_fields":'), 1)
            self.assertEqual(kwargs['prompt'].count('"last_error":'), 1)
            self.assertNotIn('"scope":', kwargs['prompt'])
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['fields']['notes']='Private deferred detail'
            await kwargs['handler'](submission(value))
        with patch('commulingo.pipeline.service.call',return_value=current), patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())(job,[],Usage(),.2)
        self.assertNotIn('notes',result.value['draft']['fields'])
        self.assertEqual(result.value['draft']['notes'],'Private deferred detail')

    async def test_create_person_and_term_use_jev_classification(self):
        from commulingo.people import _COMMULINGO_FIELD_SCHEMA
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
                from commulingo.pipeline.decisions import Decisions
                fields = deepcopy(fields)
                Decisions(job,None,catalogs,Usage()).strip_assigned(fields)
                async def model(**kwargs):
                    await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
                    await kwargs['handler'](submission({'status':'ready','reason':'The original archive supports this new entry.',
                        'fields':fields,'claims':[{'field':f,'claim':'The source documents this field.','passages':['P1']} for f in facts],
                        'issue_results':[{'id':'register','status':'resolved','reason':'Registered a supported bilingual entry.'}]}))
                with patch('commulingo.pipeline.service.call',return_value=None), \
                     patch('commulingo.classify.load_catalogs',return_value=catalogs), \
                     patch('commulingo.classify.classify_person_card',return_value={'codes':{'citizenship':{'code':'france'},'nationalOrigin':{'code':'france'}},'person':{'groupId':'fixture-group','role':{'category':'fixture-role'}}}) as classifier, \
                     patch('commulingo.classify.classify_term',return_value={'category':'culture'}), \
                     patch('commulingo.pipeline.stages.model_call',side_effect=model):
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
                await kwargs['handler'](submission(value))
            await repair_call(kwargs, candidate())
        with patch('commulingo.pipeline.service.call',return_value=CURRENT) as rpc, \
             patch('commulingo.pipeline.stages.model_call',side_effect=model) as model_call:
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
            value = candidate(); value['fields']['body']['en'] = 'A clause — another clause.'
            with caller_scope(ctx), patch('tool_gateway.security.audit'):
                response, failed = await execute_tool(kwargs['tool']['name'],submission(value),
                    {kwargs['tool']['name']:terminal},tool_schema=kwargs['tool'])
                self.assertTrue(failed)
                self.assertIn('commulingo_pipeline_submit_draft',response)
                saved = store.save_editor_checkpoint.call_args.args[1]
                self.assertEqual(saved['draft']['args']['claims'],candidate()['claims'])
                self.assertEqual(saved['draft']['args']['fields']['body']['en'], 'A clause — another clause.')
                response, failed = await execute_tool(kwargs['tool']['name'],
                    {'changes':submission(candidate())['changes']},
                    {kwargs['tool']['name']:terminal},tool_schema=kwargs['tool'])
                self.assertFalse(failed,response)
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store)(JOB,[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')

    async def test_missing_outcomes_and_evidence_reported_together(self):
        async def model(**kwargs):
            value = candidate(); del value['issue_results']; del value['claims']
            with self.assertRaises(ValueError) as failure:
                await kwargs['handler'](submission(value))
            self.assertIn('issues',str(failure.exception))
            value['issue_results'] = candidate()['issue_results']
            with self.assertRaisesRegex(ValueError, 'evidence required for body'):
                await kwargs['handler'](submission(value))
            raise RuntimeError('end test after feedback')
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), patch('commulingo.pipeline.stages.model_call',side_effect=model):
            with self.assertRaisesRegex(RuntimeError,'end test'):
                await Editor(store_mock())(JOB,[],Usage(),.2)

    async def test_repeated_malformed_repairs_do_not_hold_editor(self):
        store = store_mock()
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['claims'] = []
            with self.assertRaises(ValueError):
                await kwargs['handler'](submission(value))
            for _ in range(3):
                with self.assertRaises(ValueError):
                    await kwargs['handler']({'changes':submission(candidate())['changes'], 'unexpected':True})
                with self.assertRaises(ValueError):
                    await kwargs['handler']({})
                self.assertEqual(store.save_editor_checkpoint.call_args.args[1]['draft']['args'], value)
            await repair_call(kwargs, candidate())
        usage = Usage()
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store)(JOB,[],usage,.2)
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(usage.tracker['repair_protocol_errors'],6)

    async def test_unchanged_year_and_its_claim_are_removed_together(self):
        async def model(**kwargs):
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['fields']['startYear'] = 1923
            value['claims'].append({'field':'startYear','claim':'The year is 1923.','passages':['P1']})
            value['issue_results'].append({'id':'requested', 'status':'resolved', 'reason':'Checked the historical context and dates.'})
            await kwargs['handler'](submission(value))
        with patch('commulingo.pipeline.service.call',return_value={**CURRENT,'startYear':1923}), patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())({**JOB, 'reason':'Verify historical context and dates'},[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')
        self.assertNotIn('startYear',result.value['draft']['fields'])
        self.assertEqual([c['field'] for c in result.value['research']['claims']],['body'])

    async def test_crash_after_rejection_resumes_saved_patch_and_passage_ids(self):
        store = store_mock()
        async def first(**kwargs):
            await kwargs['read_wrap']('fetch_url',AsyncMock(return_value=f'<external source="web">\n{BODY}\n</external>'))(url=URL)
            value = candidate(); value['claims']=[]
            with self.assertRaises(ValueError):
                await kwargs['handler'](submission(value))
            raise RuntimeError('simulated provider disconnect')
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), patch('commulingo.pipeline.stages.model_call',side_effect=first):
            with self.assertRaisesRegex(RuntimeError,'disconnect'):
                await Editor(store)(JOB,[],Usage(),.2)
        checkpoint = deepcopy(store.save_editor_checkpoint.call_args.args[1])
        page = store.save_source.call_args.args[0]
        store.job_sources.return_value = {page['id']:page}
        async def resume(**kwargs):
            self.assertIn('saved_draft',kwargs['prompt'])
            self.assertIn('"missing_evidence_fields": ["body"]', kwargs['prompt'])
            self.assertIn('"error_kind": "missing_evidence"', kwargs['prompt'])
            await repair_call(kwargs, candidate())
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), patch('commulingo.pipeline.stages.model_call',side_effect=resume):
            result = await Editor(store)(JOB,[{'stage':'editor_checkpoint','value':checkpoint}],Usage(),.2)
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(result.value['draft']['fields']['evidence'][0]['excerpt'],BODY)

    async def test_identical_failure_stops_instead_of_spending_more_rounds(self):
        store = store_mock()
        async def model(**kwargs):
            value = candidate(); value['claims']=[]
            with self.assertRaises(ValueError): await kwargs['handler'](submission(value))
            await kwargs['handler'](submission(value))
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store)(JOB,[],Usage(),.2)
        self.assertEqual(result.status,'escalated')
        self.assertIn('without progress',result.value['hold_reason'])

    async def test_no_concrete_defect_needs_no_model_call(self):
        current = {**CURRENT,'body':{'ko':'본문','en':'Body'},'evidence':[{'field':'body'}]}
        with patch('commulingo.pipeline.service.call',return_value=current), patch('commulingo.pipeline.stages.model_call') as model:
            result = await Editor(store_mock())(JOB,[],Usage(),.2)
        self.assertEqual(result.next_stage,'judge'); model.assert_not_called()


CATALOGS = ([{'id':'bolshevik','title_ko':'볼셰비키','blurb_ko':'설명'}], [], [{'id':'bolshevik'}])
PERSON_VERDICT = {'person':{'groupId':'bolshevik','role':{'category':'bolshevik'},
                            'confidence':{'group':0.9,'role':0.8},'low_confidence':False},
                  'codes':{'citizenship':{'code':'soviet','confidence':1.0,'low_confidence':False},
                           'nationalOrigin':{'code':'russia','confidence':0.9,'low_confidence':False}}}
PERSON_FIELDS = {'epithet':{'ko':'수식','en':'Epithet'},'bio':{'ko':'문장이다.','en':'A sentence.'},'career':[],
                 'citizenship':{'label':{'ko':'소련','en':'Soviet'}},'nationalOrigin':{'label':{'ko':'러시아','en':'Russia'}},
                 'familyName':{'ko':'성','en':'Family'}}


def person_submission(fields):
    return submission({'status':'ready','reason':'The original archive supports this new entry.','fields':fields,
        'claims':[{'field':f,'claim':'The source documents this field.','passages':['P1']}
                  for f in ('bio','citizenship','nationalOrigin')],
        'issue_results':[{'id':'register','status':'resolved','reason':'Registered a supported bilingual entry.'}]})


async def fetch_fixture(kwargs, text=BODY):
    return await kwargs['read_wrap']('fetch_url',AsyncMock(
        return_value=f'<external source="web">\n{text}\n</external>'))(url=URL)


class EditorContractTests(EditorCase):
    """Author-side contracts carried over from the removed two-RPC stages."""

    async def test_author_schema_excludes_runner_fields_and_bounds_repair_lookups(self):
        from jsonschema import Draft202012Validator
        from tool_gateway.results import ToolRejection
        async def model(**kwargs):
            changes = kwargs['tool']['input_schema']['properties']['changes']
            # The runner owns revision and evidence; a topic name is not a field.
            self.assertFalse({'expectedRevision','evidence','sources'} & set(changes['properties']))
            validator = Draft202012Validator(kwargs['tool']['input_schema'])
            body = submission(candidate())
            self.assertTrue(validator.is_valid(body))
            topic = deepcopy(body); topic['changes']['history'] = topic['changes'].pop('body')
            self.assertFalse(validator.is_valid(topic))
            await fetch_fixture(kwargs)
            value = candidate(); value['fields']['body']['en'] = 'A clause — another clause.'
            with self.assertRaises(ValueError):
                await kwargs['handler'](submission(value))
            # Format repair: registry reads only, by ID, at most three.
            call = AsyncMock(return_value='Registry entry')
            lookup = kwargs['read_wrap']('commulingo_people',call)
            with self.assertRaises(ToolRejection):
                await lookup(action='list_terms')
            with self.assertRaises(ToolRejection):
                await kwargs['read_wrap']('web_search',call)(query='fixture')
            for _ in range(3):
                await lookup(action='get_term',term_id='fixture')
            with self.assertRaises(ToolRejection):
                await lookup(action='get_term',term_id='fixture')
            self.assertEqual(call.await_count,3)
            await repair_call(kwargs, candidate())
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), \
             patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())(JOB,[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(result.value['draft']['fields']['expectedRevision'],CURRENT['revision'])
        self.assertEqual(result.value['draft']['fields']['evidence'][0]['excerpt'],BODY)

    async def test_probe_reasons_and_undisplayed_labels_are_rejected(self):
        # Throwaway calls that passed the length floor closed jobs for 90 days (#42, #120).
        probes = ('Probe only — not a real submission, checking the tool.',
                  'Investigating commissioned topics before returning a final artifact.')
        async def model(**kwargs):
            await fetch_fixture(kwargs)
            _, no_edit, _ = next(t for t in kwargs['local_tools'] if t[0]['name']=='commulingo_pipeline_no_edit')
            for probe in probes:
                with self.assertRaisesRegex(ValueError,'probe or progress note'):
                    await kwargs['handler'](submission({**candidate(),'reason':probe}))
                with self.assertRaisesRegex(ValueError,'substantively'):
                    await no_edit(status='sources_unavailable',reason=probe,
                        issues={'missing:body':{'status':'deferred','reason':'No reliable original could be retrieved.'}})
            value = candidate(); value['claims'][0]['passages'] = ['P1','P999']
            with self.assertRaisesRegex(ValueError,'passage labels not displayed: P999'):
                await kwargs['handler'](submission(value))
            await kwargs['handler'](submission(candidate()))
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), \
             patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())(JOB,[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')

    async def test_echoed_term_years_never_reach_validation(self):
        current = {**CURRENT,'startYear':2023,'endYear':None}
        async def model(**kwargs):
            await fetch_fixture(kwargs)
            value = candidate()
            value['fields'].update(startYear=2023,endYear=None)
            value['issue_results'].append({'id':'requested','status':'resolved','reason':'Checked the historical context and dates.'})
            await kwargs['handler'](submission(value))
        with patch('commulingo.pipeline.service.call',return_value=current) as rpc, \
             patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())({**JOB,'reason':'Verify historical context and dates'},[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')
        validate = next(c.args[0] for c in rpc.call_args_list if c.args[0]['command']=='validate')
        self.assertEqual(set(validate['fields']),{'body','evidence','expectedRevision'})
        self.assertEqual(set(result.value['draft']['fields']),{'body','evidence','expectedRevision'})

    def test_person_schemas_carry_catalogs_and_store_rules(self):
        from jsonschema import Draft202012Validator
        from commulingo.pipeline.decisions import Decisions
        from commulingo.pipeline.patches import schema_for
        update = {**JOB,'kind':'person','action':'update','topic':'basics','target':'fixture'}
        canonical = schema_for(update,None,CATALOGS)
        self.assertEqual(canonical['properties']['groupId']['enum'],['bolshevik'])
        self.assertEqual(canonical['properties']['role']['properties']['category']['enum'],['bolshevik'])
        validator = Draft202012Validator(canonical)
        for collection,edits in (('aliases','aliasEdits'),('career','careerEdits'),('scenes','sceneEdits')):
            errors = list(validator.iter_errors({collection:[],edits:[]}))
            self.assertTrue(any(e.validator=='not' for e in errors), collection)
        with self.assertRaisesRegex(ValueError,'catalogs unavailable'):
            schema_for(update,None,None)
        create = {**update,'action':'create','target':'new-person'}
        schema = Decisions(create,None,CATALOGS,Usage()).author_schema(schema_for(create,None,CATALOGS))
        # The writer never classifies: no group/role, and the code keys are gone from the code objects.
        self.assertFalse({'groupId','group','role','activities'} & set(schema['properties']))
        self.assertNotIn('code',schema['properties']['citizenship']['properties'])
        validator = Draft202012Validator(schema)
        self.assertTrue(validator.is_valid({**PERSON_FIELDS,'years':'1895?–1940'}),
                        list(validator.iter_errors({**PERSON_FIELDS,'years':'1895?–1940'})))
        self.assertFalse(validator.is_valid({**PERSON_FIELDS,'groupId':'bolshevik','years':'1895–1940'}))
        self.assertFalse(validator.is_valid({**PERSON_FIELDS,'years':'1900–현재'}))
        self.assertFalse(validator.is_valid({**PERSON_FIELDS,'sortOrder':None}))
        self.assertFalse(validator.is_valid({k:v for k,v in PERSON_FIELDS.items() if k!='familyName'}))

    async def test_person_classification_follows_local_checks_and_is_memoised(self):
        calls = []
        def card(fields, catalogs=None, claims=None, decide=None):
            calls.append(deepcopy(fields))
            return deepcopy(PERSON_VERDICT)
        job = {**JOB,'kind':'person','action':'create','topic':'basics','target':'new-person'}
        async def model(**kwargs):
            await fetch_fixture(kwargs)
            dashed = {**PERSON_FIELDS,'epithet':{'ko':'수식 — 부제','en':'Epithet — sub'}}
            with self.assertRaises(ValueError):   # prose bounce before any decision
                await kwargs['handler'](person_submission(dashed))
            self.assertEqual(calls,[])
            await kwargs['handler'](person_submission(PERSON_FIELDS))
        with patch('commulingo.pipeline.service.call',return_value=None), \
             patch('commulingo.classify.load_catalogs',return_value=CATALOGS), \
             patch('commulingo.classify.classify_person_card',side_effect=card), \
             patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await Editor(store_mock())(job,[],Usage(),.2)
        self.assertEqual(result.next_stage,'review')
        self.assertEqual(len(calls),1)
        fields = result.value['draft']['fields']
        self.assertEqual((fields['groupId'],fields['role']),('bolshevik',{'category':'bolshevik'}))
        self.assertEqual((fields['citizenship']['code'],fields['nationalOrigin']['code']),('soviet','russia'))
        self.assertEqual(result.value['draft']['classification']['person']['groupId'],'bolshevik')
        # The verdict is memoised on the classified inputs, not on the call.
        from commulingo.pipeline.decisions import Decisions
        page = snapshot(URL,BODY)
        claims = [{'field':f,'claim':'x','source_id':page['id'],'start':0,'end':20} for f in ('bio','citizenship')]
        decisions = Decisions(job,None,CATALOGS,Usage())
        calls.clear()
        with patch('commulingo.classify.classify_person_card',side_effect=card):
            await decisions.classify(deepcopy(PERSON_FIELDS),claims,{page['id']:page})
            await decisions.classify(deepcopy(PERSON_FIELDS),claims,{page['id']:page})
            self.assertEqual(len(calls),1)
            await decisions.classify({**PERSON_FIELDS,'bio':{'ko':'다른 문장이다.','en':'Another sentence.'}},claims,{page['id']:page})
            self.assertEqual(len(calls),2)

    async def test_person_classifier_outage_keeps_draft_without_asking_the_writer(self):
        store = store_mock()
        job = {**JOB,'kind':'person','action':'create','topic':'basics','target':'new-person'}
        async def model(**kwargs):
            await fetch_fixture(kwargs)
            await kwargs['handler'](person_submission(PERSON_FIELDS))
        with patch('commulingo.pipeline.service.call',return_value=None) as rpc, \
             patch('commulingo.classify.load_catalogs',return_value=CATALOGS), \
             patch('commulingo.classify.classify_person_card',return_value=None), \
             patch('commulingo.pipeline.stages.model_call',side_effect=model):
            with self.assertRaisesRegex(RuntimeError,'classification unavailable'):
                await Editor(store)(job,[],Usage(),.2)
        self.assertEqual([c.args[0]['command'] for c in rpc.call_args_list],['read'])
        saved = store.save_editor_checkpoint.call_args.args[1]['draft']['args']['fields']
        self.assertEqual(saved['epithet'],PERSON_FIELDS['epithet'])
        self.assertFalse({'groupId','role'} & set(saved))

    async def test_existing_person_classification_is_not_reassigned(self):
        from commulingo.pipeline.decisions import Decisions
        from commulingo.pipeline.patches import schema_for
        page = snapshot(URL,BODY)
        sources = {page['id']:page}
        job = {**JOB,'kind':'person','action':'update','topic':'basics','target':'stalin'}
        current = {'revision':'v1','groupId':'bolshevik','role':{'officeId':'party-leadership','category':''}}
        decisions = Decisions(job,current,CATALOGS,Usage())
        self.assertFalse({'role','group','groupId'} & set(decisions.author_schema(schema_for(job,current,CATALOGS))['properties']))
        fields = {'citizenship':{'label':{'ko':'소련','en':'Soviet'}}}
        claim = [{'field':'citizenship','claim':'x','source_id':page['id'],'start':0,'end':20}]
        with patch('commulingo.classify.classify_person_codes',return_value={'citizenship':{'code':'soviet'}}), \
             patch('commulingo.classify.classify_person_card') as card:
            out, _ = await decisions.classify(fields,claim,sources)
        card.assert_not_called()
        self.assertEqual(out,{'citizenship':{'label':{'ko':'소련','en':'Soviet'},'code':'soviet'}})
        # A person without any classification receives one from the runner.
        bare = Decisions({**job,'target':'new'},{'revision':'v1','role':{}},CATALOGS,Usage())
        claim = [{'field':'bio','claim':'x','source_id':page['id'],'start':0,'end':20}]
        with patch('commulingo.classify.classify_person_card',return_value=deepcopy(PERSON_VERDICT)):
            out, _ = await bare.classify({'bio':{'ko':'문장이다.','en':'A sentence.'}},claim,sources)
        self.assertEqual((out['groupId'],out['role']),('bolshevik',{'category':'bolshevik'}))

    async def test_section_is_one_topic_and_notes_stay_out_of_fields(self):
        # Job 5432 (2026-09-19): the author's work plan went live as a section.
        from commulingo.pipeline.stages import prose_problem
        prose = {'ko':'1937년 7월 예조프는 레닌 훈장을 받았고 1941년 1월 24일 모든 훈장을 박탈당했다. '*4,
                 'en':'In July 1937 Yezhov received the Order of Lenin; a decree of 24 January 1941 stripped him of all awards. '*4}
        heading = {'ko':'숭배와 말소','en':'Cult and erasure'}
        current = {'revision':'v1','name':{'ko':'니콜라이 예조프','en':'Nikolai Yezhov'},
                   'sections':[{'slug':'fall-trial-no-rehabilitation'}]}
        job = {**JOB,'id':5432,'kind':'person','topic':'enrichment','target':'yezhov',
               'reason':'Add a documented section on the cult and its erasure.',
               'payload':{'topics':['bio','sections'],'remaining_topics':['sections']}}
        def value(fields, notes=None):
            result = {'status':'ready','reason':'The decree and the award record document this section.',
                      'fields':fields,'claims':[{'field':'body','claim':'The awards and their removal are documented.','passages':['P1']}],
                      'issue_results':[{'id':'requested','status':'resolved','reason':'Added one documented section.'}]}
            if notes:
                result['notes'] = notes
            return submission(result)
        seen = {}
        async def model(**kwargs):
            seen['props'] = set(kwargs['tool']['input_schema']['properties']['changes']['properties'])
            self.assertIn('one distinct documented phase',kwargs['prompt'])
            await fetch_fixture(kwargs)
            plan = {'heading':{'ko':'구획 개정: 정정과 신설 구획 둘','en':'Sections revision: corrections and two new sections'},
                    'body':{'ko':"과제 항목은 'sections'다. 1. 'fall-trial-no-rehabilitation' 구획 본문 교체: "+prose['ko'],
                            'en':"The commissioned topic is 'sections'. 1. replace section body 'fall-trial-no-rehabilitation': "+prose['en']}}
            with self.assertRaisesRegex(ValueError,'unexpected'):
                await kwargs['handler'](value({**plan,'startYear':1937,'slug':'yezhov'}))
            with self.assertRaisesRegex(ValueError,'person name'):
                await kwargs['handler'](value({'heading':{'ko':'니콜라이 예조프','en':'Cult'},'body':prose,'startYear':1937}))
            await kwargs['handler'](value({'heading':heading,'body':prose,'startYear':1937,'startMonth':7},
                                          notes="also supported: correction of 'fall-trial-no-rehabilitation' dates"))
        with patch('commulingo.pipeline.service.call',return_value=current) as rpc, \
             patch('commulingo.pipeline.stages.model_call',side_effect=model), \
             patch('commulingo.section_slug.generate_section_slug',return_value='cult-and-erasure') as slugger:
            result = await Editor(store_mock())(job,[],Usage(),.2)
        self.assertEqual(seen['props'],{'heading','body','startYear','startMonth'})
        slugger.assert_called_once()
        draft = result.value['draft']
        self.assertEqual((draft['target'],draft['action']),('person_section','create'))
        self.assertEqual(draft['fields']['sortOrder'],193707)
        self.assertNotIn('startYear',draft['fields'])
        self.assertEqual(draft['fields']['slug'],'cult-and-erasure')
        self.assertIn('fall-trial',draft['notes'])
        validate = [c.args[0] for c in rpc.call_args_list if c.args[0]['command']=='validate'][-1]
        self.assertEqual(validate['target'],'person_section')
        self.assertNotIn('notes',validate['fields'])
        self.assertFalse(prose_problem(draft['fields']))

    async def test_section_sort_order_comes_from_start_year_not_heading(self):
        # 2026-09-22..24: editor sections had no period field and stored sort_order 0.
        prose = {'ko':'1937년 7월 예조프는 레닌 훈장을 받았고 1941년 1월 24일 모든 훈장을 박탈당했다. '*4,
                 'en':'In July 1937 Yezhov received the Order of Lenin; a decree of 24 January 1941 stripped him of all awards. '*4}
        current = {'revision':'v1','name':{'ko':'니콜라이 예조프','en':'Nikolai Yezhov'},
                   'sections':[{'slug':'fall-trial-no-rehabilitation','sortOrder':193900}]}
        job = {**JOB,'id':5433,'kind':'person','topic':'enrichment','target':'yezhov',
               'reason':'Add a documented section on the cult and its erasure.',
               'payload':{'topics':['bio','sections'],'remaining_topics':['sections']}}
        async def model(**kwargs):
            await fetch_fixture(kwargs)
            await kwargs['handler'](submission({'status':'ready','reason':'The decree and the award record document this section.',
                'fields':{'heading':{'ko':'1941년의 말소','en':'Erasure in 1941'},'body':prose,'startYear':1937},
                'claims':[{'field':'body','claim':'The awards and their removal are documented.','passages':['P1']},
                          {'field':'startYear','claim':'The award was given in 1937.','passages':['P1']}],
                'issue_results':[{'id':'requested','status':'resolved','reason':'Added one documented section.'}]}))
        with patch('commulingo.pipeline.service.call',return_value=current), \
             patch('commulingo.pipeline.stages.model_call',side_effect=model), \
             patch('commulingo.section_slug.generate_section_slug',return_value='cult-and-erasure'):
            result = await Editor(store_mock())(job,[],Usage(),.2)
        self.assertEqual(result.value['draft']['fields']['sortOrder'],193700)
        self.assertEqual({item['field'] for item in result.value['draft']['fields']['evidence']},
                         {'body', 'sortOrder'})
        self.assertNotIn('startYear', {item['field'] for item in result.value['draft']['fields']['evidence']})

    def test_section_start_year_accepts_explicit_unknown_date(self):
        from commulingo.pipeline.patches import schema_for
        job = {**JOB, 'kind':'person', 'action':'update',
               'payload':{'topics':['sections']}}
        fields = schema_for(job, {'sections':[]})
        self.assertIn('null', fields['properties']['startYear']['type'])
        self.assertIn('startYear', fields['required'])


class ReviewAndPublishTests(EditorCase):
    async def test_routing_pins_unpinned_authoring_and_rejects_other_workflows(self):
        store = Mock()
        editor = {name:AsyncMock(return_value=name) for name in ('discover','research','draft','submit')}
        with patch('commulingo.pipeline.workflow.stages',return_value=editor):
            routed = workflow.routed_stages(store)
        # A job predating the workflow field cannot resume after drafting:
        # its draft came from the removed two-RPC stages.
        job = {**JOB,'payload':{}}
        with self.assertRaisesRegex(ValueError,'post-draft stage'):
            await routed['submit'](job,[],Usage(),.2)
        store.pin_editor_workflow.assert_not_called()
        editor['submit'].assert_not_awaited()
        # Authoring stages pin it to the editor before running.
        for stage in ('research','draft','discover'):
            job = {**JOB,'payload':{}}
            store.reset_mock()
            self.assertEqual(await routed[stage](job,[],Usage(),.2),stage)
            store.pin_editor_workflow.assert_called_once_with(job)
            self.assertEqual(job['payload']['workflow'],'editor')
        self.assertEqual(await routed['submit'](job,[],Usage(),.2),'submit')
        store.reset_mock()
        with self.assertRaisesRegex(ValueError,'no longer supported'):
            await routed['research']({**JOB,'payload':{'workflow':'legacy'}},[],Usage(),.2)
        with self.assertRaisesRegex(ValueError,'no longer supported'):
            await routed['submit'].prepare({**JOB,'payload':{'workflow':'legacy'}},[],Usage())
        store.pin_editor_workflow.assert_not_called()
        with self.assertRaisesRegex(ValueError,'unknown editorial workflow'):
            workflow.routed_stages(store,'legacy')

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
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), \
             patch('runtime_tools.registry.TOOL_HANDLERS',reads), \
             patch('commulingo.review_handlers.review_risks',return_value=[]), \
             patch('commulingo.pipeline.stages.model_call',side_effect=model):
            result = await workflow.Review()(JOB,artifacts,Usage(),.2)
        self.assertEqual(result.next_stage,'submit')
        self.assertEqual(result.value['approved_patch_hash'],patch_hash(write_request(JOB,latest(artifacts,'draft'))))
        self.assertEqual(result.value['optional_suggestions'],value['optional_suggestions'])

    async def test_unchanged_rejected_patch_is_held_before_paid_review(self):
        artifacts = self.artifacts()
        digest = patch_hash(write_request(JOB,latest(artifacts,'draft')))
        artifacts.append({'stage':'review','value':{'decision':'revise','reviewed_patch_hash':digest}})
        with patch('commulingo.pipeline.service.call',return_value=CURRENT), patch('commulingo.pipeline.stages.model_call') as model:
            result = await workflow.Review()(JOB,artifacts,Usage(),.2)
        self.assertEqual(result.status,'escalated'); model.assert_not_called()

    async def test_publish_is_one_rpc_and_tampering_requires_new_review(self):
        artifacts = self.artifacts()
        digest = patch_hash(write_request(JOB,latest(artifacts,'draft')))
        decision = {'decision':'approve','approved_patch_hash':digest,'reason':'Independently verified.',
                    'checks':[{'citation':URL,'source':URL,'quote':BODY,'finding':'Verified.'}]}
        artifacts.append({'stage':'review','value':decision})
        with patch('commulingo.pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo.pipeline.service.call',return_value={'status':'approved','suggestionId':123}) as rpc:
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

    def fake_review(self, decisions, prompts=None, proposals=None):
        """Replace only the decision tool's validation; the stage's own routing runs."""
        def handlers(reads, proposal, snapshots, box, gate=None, **kwargs):
            if proposals is not None:
                proposals.append(deepcopy(proposal))
            async def decide(**value):
                box.update(value)
                return 'OK'
            return {'commulingo_review_decision':decide}
        async def model(**kwargs):
            if prompts is not None:
                prompts.append(kwargs['prompt'])
            await kwargs['handler'](decisions.pop(0))
        return handlers, model

    def review_patches(self, handlers, model):
        reads = {name:AsyncMock() for name in ('wiki_search','wiki_get','web_search','fetch_url','commulingo_people')}
        return (patch('commulingo.pipeline.service.call',return_value=CURRENT),
                patch('runtime_tools.registry.TOOL_HANDLERS',reads),
                patch('commulingo.review_handlers.review_risks',return_value=[]),
                patch('commulingo.review_handlers.make_handlers',side_effect=handlers),
                patch('commulingo.pipeline.stages.model_call',side_effect=model))

    async def run_review(self, job, artifacts, decision, prompts=None, proposals=None):
        handlers, model = self.fake_review([decision], prompts, proposals)
        a, b, c, d, e = self.review_patches(handlers, model)
        with a as rpc, b, c, d, e:
            result = await workflow.Review()(job, artifacts, Usage(), .2)
        self.review_rpcs = [call.args[0] for call in rpc.call_args_list]
        return result

    def review_notes(self):
        return [r for r in self.review_rpcs if r['command']=='note']

    def verdict(self, decision, corrections=()):
        return {'decision':decision,'reason':'Verified against the original archive.','checks':[],
                'resolved_risks':[],'required_corrections':list(corrections),'optional_suggestions':[]}

    async def test_review_routes_each_decision_and_holds_only_an_unchanged_patch(self):
        correction = [{'path':'/fields/body','reason':'The date contradicts the original.'}]
        for decision, corrections, expected in [('revise',correction,('draft','ready')),
                                                ('approve',[],('submit','ready')),
                                                ('escalate',[],('complete','escalated')),
                                                ('reject',[],('complete','complete'))]:
            with self.subTest(decision=decision):
                result = await self.run_review(JOB,self.artifacts(),self.verdict(decision,corrections))
                self.assertEqual((result.next_stage,result.status),expected)
                # A review that ends unpublished leaves its reason on the entry for the next author.
                notes = self.review_notes()
                if decision in ('escalate','reject'):
                    self.assertEqual(len(notes),1)
                    self.assertTrue(notes[0]['note'].startswith(f'검토 {decision} (작업 {JOB["id"]}'))
                    self.assertIn('Verified against the original archive.',notes[0]['note'])
                    self.assertEqual((notes[0]['target'],notes[0]['id'],notes[0]['changedBy']),
                                     (JOB['kind'],JOB['target'],'commulingo-pipeline-reviewer'))
                else:
                    self.assertEqual(notes,[])
        # Lifetime counters from older jobs do not hold a corrected draft.
        job = {**JOB,'payload':{'review_revisions':20}}
        history = self.artifacts()
        first = await self.run_review(job,history,self.verdict('revise',correction))
        self.assertEqual(first.next_stage,'draft')
        history.append({'stage':'review','value':first.value})
        # The same patch comes back: held before a paid review.
        with patch('commulingo.pipeline.service.call',return_value=CURRENT) as rpc, \
             patch('commulingo.pipeline.stages.model_call') as model:
            stalled = await workflow.Review()(job,history,Usage(),.2)
        model.assert_not_called()
        self.assertEqual(stalled.status,'escalated')
        held = [c.args[0] for c in rpc.call_args_list if c.args[0]['command']=='note']
        self.assertEqual(len(held),1)
        self.assertTrue(held[0]['note'].startswith('검토 revise (held)'))
        corrected = deepcopy(history[0]['value']['draft'])
        corrected['fields']['body']['en'] = 'A corrected documented historical context.'
        history.append({'stage':'research','value':{'editor_version':2,'research':{'baseline':CURRENT['revision']},'draft':corrected}})
        again = await self.run_review(job,history,self.verdict('revise',correction))
        self.assertEqual(again.next_stage,'draft')
        self.assertNotEqual(again.value['reviewed_patch_hash'],first.value['reviewed_patch_hash'])

    async def test_rereview_receives_verdicts_of_the_current_bundle_only(self):
        stale = {'stage':'review','value':{'decision':'revise','reason':'Earlier section verdict','checks':[]}}
        boundary = {'stage':'submit','value':{'remaining_topics':['sections']}}
        revise = {'stage':'review','value':{'decision':'revise','reason':'Fix the patronymic',
                  'required_corrections':[{'path':'/fields/body','reason':'부칭 표기 오류'}],'checks':[]}}
        corrected = deepcopy(self.artifacts()[0]['value']['draft'])
        corrected['fields']['body']['ko'] = '수정한 역사적 맥락이다.'
        artifacts = [stale, boundary, *self.artifacts(), revise,
                     {'stage':'research','value':{'editor_version':2,'research':{'baseline':CURRENT['revision']},'draft':corrected}}]
        prompts = []
        result = await self.run_review(JOB,artifacts,self.verdict('approve'),prompts)
        self.assertEqual(result.next_stage,'submit')
        first = await self.run_review(JOB,artifacts[2:3],self.verdict('approve'),prompts)
        self.assertEqual(first.next_stage,'submit')
        self.assertIn('Fix the patronymic',prompts[0])
        self.assertIn('부칭 표기 오류',prompts[0])
        self.assertIn('changes_since_previous_patch',prompts[0])
        self.assertNotIn('Earlier section verdict',prompts[0])  # previous bundle stays out
        self.assertIn('"previous_reviews": []',prompts[1])

    async def test_low_confidence_classification_becomes_a_review_risk(self):
        artifacts = self.artifacts()
        classification = {'person':{'groupId':'bolshevik','confidence':{'group':0.95,'role':0.41},'low_confidence':True}}
        artifacts[0]['value']['draft']['classification'] = classification
        prompts, proposals = [], []
        await self.run_review(JOB,artifacts,self.verdict('approve'),prompts,proposals)
        self.assertEqual(proposals[0]['classification'],classification)
        self.assertTrue(any('Jev-assigned classification' in r and 'low_confidence' in r for r in proposals[0]['risks']))
        self.assertIn('0.41',prompts[0])
        proposals.clear()
        await self.run_review(JOB,self.artifacts(),self.verdict('approve'),None,proposals)
        self.assertNotIn('classification',proposals[0])
        self.assertEqual(proposals[0]['risks'],[])

    def approved(self, job=JOB, notes=''):
        artifacts = self.artifacts()
        artifacts[0]['value']['draft']['notes'] = notes
        digest = patch_hash(write_request(job,latest(artifacts,'draft')))
        artifacts.append({'stage':'review','value':{'decision':'approve','approved_patch_hash':digest,
            'reason':'Independently verified.','checks':[{'citation':URL,'source':URL,'quote':BODY,'finding':'Verified.'}]}})
        return artifacts, digest

    async def test_publish_replay_is_identical_and_keeps_notes_out_of_fields(self):
        artifacts, digest = self.approved(notes='examples still unsupported')
        with patch('commulingo.pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo.pipeline.service.call',return_value={'status':'approved','suggestionId':12}) as rpc:
            await workflow.publish(JOB,artifacts,Usage(),.2)
            await workflow.publish(JOB,artifacts,Usage(),.2)
        first, second = [c.args[0] for c in rpc.call_args_list]
        self.assertEqual(first,second)
        self.assertEqual(first['idempotencyKey'],f"pipeline:{JOB['id']}:publish:{digest}")
        # The author's notes outlive the job: saved with the entry by the same receipt.
        self.assertEqual(first['notes'],'examples still unsupported')
        self.assertNotIn('notes',first['fields'])

    async def test_correction_replaces_the_original_only_inside_approved_publication(self):
        job = {**JOB,'topic':'review-repair:7','payload':{'workflow':'editor','replaces_suggestion_id':7}}
        artifacts, digest = self.approved(job)
        pending = {'id':7,'status':'pending','review_note':None}
        with patch('commulingo.pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo.review_queue.suggestion',return_value=pending), \
             patch('commulingo.pipeline.service.call',return_value={'status':'approved','suggestionId':8}) as rpc:
            result = await workflow.publish(job,artifacts,Usage(),.2)
        self.assertEqual(result.value['status'],'approved')
        # Replay after this job's own publish replaced the original: the same
        # request goes out again and the service returns the stored receipt.
        ours = {'id':7,'status':'rejected','review_note':workflow.REPLACED_NOTE_PREFIX+digest}
        with patch('commulingo.pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo.review_queue.suggestion',return_value=ours), \
             patch('commulingo.pipeline.service.call',return_value={'status':'approved','suggestionId':8}) as replay:
            await workflow.publish(job,artifacts,Usage(),.2)
        first, second = rpc.call_args.args[0], replay.call_args.args[0]
        self.assertEqual(first,second)
        self.assertEqual((first['command'],first['replacesSuggestionId'],first['approvedPatchHash']),('publish',7,digest))
        # An original resolved by hand (or missing) is no longer ours to replace:
        # the job completes quietly and nothing is published.
        for original in ({'id':7,'status':'approved','review_note':'looks right'},
                         {'id':7,'status':'rejected','review_note':'duplicate'},
                         None):
            with self.subTest(original=original), \
                 patch('commulingo.pipeline.config.load',return_value={'phase':'live'}), \
                 patch('commulingo.review_queue.suggestion',return_value=original), \
                 patch('commulingo.pipeline.service.call') as rpc:
                result = await workflow.publish(job,artifacts,Usage(),.2)
            rpc.assert_not_called()
            self.assertEqual((result.next_stage,result.status),('complete','complete'))
        # Without an approval bound to this patch nothing is written.
        unapproved = artifacts[:-1]+[{'stage':'review','value':{'decision':'revise'}}]
        with patch('commulingo.pipeline.config.load',return_value={'phase':'live'}), \
             patch('commulingo.pipeline.service.call') as rpc:
            result = await workflow.publish(job,unapproved,Usage(),.2)
        rpc.assert_not_called()
        self.assertEqual(result.next_stage,'review')

    async def test_validate_resume_returns_failures_to_authoring(self):
        artifacts = self.artifacts()
        for outcome, expected in [({}, 'review'),
                                  (ValueError('400: evidence required for body'), 'draft'),
                                  (ValueError('400: definition too long'), 'draft'),
                                  (ValueError('409: revision_conflict'), 'research')]:
            with self.subTest(outcome=str(outcome)), \
                 patch('commulingo.pipeline.service.call',side_effect=[outcome] if isinstance(outcome,Exception) else None,
                       return_value=outcome) as rpc:
                result = await workflow.validate(JOB,artifacts,Usage(),.2)
            self.assertEqual(result.next_stage,expected)
            self.assertEqual(rpc.call_args.args[0]['command'],'validate')
