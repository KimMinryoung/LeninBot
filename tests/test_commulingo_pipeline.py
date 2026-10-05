import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from commulingo.pipeline.evidence import snapshot, compile_evidence


class EvidenceTests(unittest.TestCase):
    def test_person_create_rejects_update_only_fields_in_tool_schema(self):
        from jsonschema import Draft202012Validator
        from commulingo.people import COMMULINGO_PERSON_CREATE_TOOL, COMMULINGO_PERSON_UPDATE_TOOL
        for field,value in [('aliasEdits',[]),('careerEdits',[]),('sceneEdits',[]),('expectedRevision','revision')]:
            with self.subTest(field=field):
                create = COMMULINGO_PERSON_CREATE_TOOL['input_schema']['properties']['fields']
                update = COMMULINGO_PERSON_UPDATE_TOOL['input_schema']['properties']['fields']
                errors = list(Draft202012Validator(create).iter_errors({field:value}))
                self.assertTrue(any(e.validator=='additionalProperties' and field in e.message for e in errors))
                self.assertIn(field,update['properties'])

    def test_source_nul_normalization_precedes_hash_and_ranges(self):
        source = snapshot('https://example.org/source','Historical\x00document with enough context.')
        self.assertNotIn('\x00',source['body'])
        same = snapshot(source['url'],source['body'])
        self.assertEqual(source['id'],same['id'])

    def test_passage_labels_resolve_to_displayed_paragraphs(self):
        from commulingo.pipeline.evidence import SourceHandles, Passages, resolve_passages
        body = ('Intro sentence here.\n•\n그는 1917년에 입당했다 — «Правда» 편집부에서 일했다.\n\n'
                'Later he was exiled to Siberia! Final sentence of the page.')
        source = snapshot('https://example.org/source', body)
        sources = {source['id']:source}
        handles = SourceHandles(sources)
        passages = Passages()
        text = passages.show(source['id'], body)
        # Each displayed paragraph has an immutable short label.
        # Every non-blank line is citable, a one-character bullet included: whether it supports a claim is the gate's call.
        self.assertEqual(list(passages.shown), ['P1', 'P2', 'P3', 'P4'])
        self.assertTrue(text.startswith('[P1] Intro sentence here.\n[P2] •\n[P3] 그는 1917년에'))
        claim = {'field':'bio','claim':'Joined in 1917','passages':['P3']}
        [resolved] = resolve_passages([claim], passages, sources)
        self.assertNotIn('passages', resolved)
        self.assertEqual(body[resolved['start']:resolved['end']], '그는 1917년에 입당했다 — «Правда» 편집부에서 일했다.')
        compiled = compile_evidence([resolved], sources, {'bio'})
        self.assertEqual(compiled[0]['excerpt'], '그는 1917년에 입당했다 — «Правда» 편집부에서 일했다.')
        # Several paragraphs of one source span from the first to the last, whatever the order given.
        [wide] = resolve_passages([{**claim,'passages':['P4','P3']}], passages, sources)
        self.assertEqual((wide['start'], wide['end']), (23, len(body)))
        other = snapshot('https://example.org/other', 'Unrelated page.\nHe was exiled to Siberia in 1930.')
        sources[other['id']] = other
        passages.show(other['id'], other['body'])
        # A label never shown is refused by claim number.
        with self.assertRaisesRegex(ValueError, "claim 1: passage labels not displayed: P999"):
            resolve_passages([{**claim,'passages':['P999']}], passages, sources)
        # Labels of two sources become two claims.
        split = resolve_passages([claim, {**claim,'passages':['P3','P6']}], passages, sources)
        self.assertEqual([(c['source_id'], c['start']) for c in split], [(source['id'], 23), (source['id'], 23), (other['id'], 16)])
        # Paragraphs too far apart for one evidence range become one claim per cluster.
        far = snapshot('https://example.org/far', 'First paragraph of a long page.\n' + 'x' * 7000 + '\nLast paragraph states the fact.')
        sources[far['id']] = far
        far_text = passages.show(far['id'], far['body'])
        far_labels = re.findall(r'\[(P\d+)\]', far_text)
        pieces = resolve_passages([{**claim,'passages':far_labels}], passages, sources)
        ranges = [(c['start'], c['end']) for c in pieces]
        self.assertTrue(len(ranges) >= 2 and ranges[0][0] == 0 and ranges[-1][1] == len(far['body']))
        self.assertTrue(all(end - start <= 6000 for start, end in ranges), ranges)
        # Text that changed since display (a restarted snapshot) is refused.
        sources[source['id']] = {**source, 'body': body.replace('1917', '1918')}
        with self.assertRaisesRegex(ValueError, 'claim 1: source snapshot changed since these passages were shown'):
            resolve_passages([claim], passages, sources)

    def test_exact_range_and_expiry(self):
        source = snapshot('https://example.org/source','A documented event happened in 1917.')
        claim = {'field':'bio','claim':'The event happened in 1917',
                 'source_id':source['id'],'start':2,'end':35}
        evidence = compile_evidence([claim],{source['id']:source},{'bio'})
        self.assertEqual(evidence[0]['excerpt'],source['body'][2:35])
        with self.assertRaises(ValueError):
            compile_evidence([claim],{source['id']:source},{'moment'})
        source['expires_at'] = datetime.now(timezone.utc)-timedelta(seconds=1)
        with self.assertRaises(ValueError):
            compile_evidence([claim],{source['id']:source},{'bio'})


class BatchTests(unittest.IsolatedAsyncioTestCase):


    async def test_bundle_judges_each_card_topic_then_advances_to_sections(self):
        from commulingo.pipeline.stages import judge
        from commulingo.pipeline.engine import Usage
        from commulingo.pipeline.bundles import work_topics
        job = {'id':7,'kind':'person','target':'p','topic':'enrichment',
               'payload':{'topics':['basics','bio','sections']}}
        research = {'current':{'revision':'r1'},'baseline':'r1','status':'complete',
                    'reason':'All card facts already supported','inspected_sources':['source']}
        with patch('commulingo.pipeline.stages.service.call') as rpc:
            result = await judge(job,[{'stage':'research','value':research}],Usage(),.2)
        self.assertEqual([c.args[0]['topic'] for c in rpc.call_args_list],['basics','bio'])
        self.assertEqual({c.args[0]['expectedRevision'] for c in rpc.call_args_list},{'r1'})
        self.assertEqual(result.next_stage,'research')
        self.assertEqual(result.value['remaining_topics'],['sections'])
        self.assertEqual(work_topics({**job,'payload':{**job['payload'],**result.value}}),['sections'])

    async def test_bundle_unavailable_does_not_skip_topics(self):
        from commulingo.pipeline.stages import judge
        from commulingo.pipeline.engine import Usage
        job = {'id':7,'kind':'person','target':'p','topic':'enrichment',
               'payload':{'topics':['bio','sections']}}
        with patch('commulingo.pipeline.stages.service.call'):
            result = await judge(job,[{'stage':'research','value':{'status':'sources_unavailable'}}],Usage(),.2)
        self.assertEqual(result.status,'deferred')
        self.assertNotIn('remaining_topics',result.value)


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_gap_contract_is_single_requested_entry(self):
        from commulingo.pipeline.stages import Discover
        from commulingo.pipeline.engine import Usage
        from jsonschema import validate, ValidationError
        payload = {'material_id':'gap:2007','requested_kind':'term','label':'독립',
                   'body':'독립과 주변 인물 및 개념들'}
        candidate = {'kind':'term','target':'independence','label':'독립','mention':'독립',
                     'reason':'A historically important concept missing from the dictionary.'}
        async def model(**kwargs):
            schema = kwargs['tool']['input_schema']
            validate({'candidates':[candidate]},schema)
            with self.assertRaises(ValidationError):
                validate({'candidates':[candidate,candidate]},schema)
            self.assertIn('zero or one', kwargs['prompt'])
            self.assertNotIn('up to four', kwargs['prompt'])
            # The runner owns kind, label and mention: a decorated label or a
            # paraphrased mention is corrected, not rejected (2026-09-17).
            decorated = {**candidate,'label':'독립 (Independence)','mention':'본문 표현: 독립','kind':'person'}
            validate({'candidates':[decorated]},schema)
            await kwargs['handler']({'candidates':[decorated]})
        with patch('commulingo.pipeline.stages.model_call',side_effect=model), patch('commulingo.mcp_client.call_tool',return_value={'existingId':None,'eventTitleMatch':None}):
            result = await Discover()({'id':1,'payload':payload},[],Usage(),.2)
        self.assertEqual(result.value['candidates'],[candidate])
        # Declining a requested entry needs a visible reason.
        async def decline(**kwargs):
            with self.assertRaises(ValueError):
                await kwargs['handler']({'candidates':[]})
            await kwargs['handler']({'candidates':[],'reason':'The entry already exists as another term.'})
        with patch('commulingo.pipeline.stages.model_call',side_effect=decline), patch('commulingo.mcp_client.call_tool',return_value={'existingId':None,'eventTitleMatch':None}):
            result = await Discover()({'id':1,'payload':payload},[],Usage(),.2)
        self.assertEqual(result.value, {'candidates':[], 'skip_reason':'The entry already exists as another term.'})


    async def test_model_stage_uses_real_dispatcher_for_artifact_schema(self):
        from commulingo.pipeline.stages import model_call, result_tool
        from commulingo.pipeline.prompts import spec
        from commulingo.pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        saved = []
        tool = result_tool({'type':'object','additionalProperties':False,
            'properties':{'reason':{'type':'string','minLength':5}},'required':['reason']})
        async def handler(value):
            saved.append(value)
            return 'OK: saved artifact'
        async def chat(*args,**kwargs):
            self.assertEqual([t['name'] for t in kwargs['tools']],['commulingo_pipeline_result'])
            self.assertTrue(kwargs['continue_on_length'])
            self.assertEqual(kwargs['max_length_continuations'], 2)
            with patch('tool_gateway.security.audit'):
                _, failed = await execute_tool(tool['name'],{'reason':2},kwargs['tool_handlers'],tool_schema=tool['input_schema'])
                self.assertTrue(failed)
                result, failed = await execute_tool(tool['name'],{'reason':'A sourced conclusion'},kwargs['tool_handlers'],tool_schema=tool['input_schema'])
                self.assertFalse(failed,result)
            kwargs['budget_tracker']['total_cost']=.01
            kwargs['budget_tracker']['jev_cost_usd']=.0001
        binding = SimpleNamespace(chat=chat,client=None,model='fixture',render_provider='deepseek',reasoning={})
        usage = Usage()
        with patch('bot_config.resolve_agent_tool_loop',return_value=binding):
            await model_call(spec=spec('research'),prompt='Fixture',tool=tool,handler=handler,
                             reads=set(),usage=usage,budget=.2)
        self.assertAlmostEqual(usage.tracker['total_cost'],.0101)
        self.assertEqual(saved,[{'reason':'A sourced conclusion'}])

    async def test_local_tool_is_a_terminal_including_forced_finalization(self):
        from commulingo.pipeline.stages import model_call, result_tool
        from commulingo.pipeline.prompts import spec
        from commulingo.pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        tool = result_tool({'type':'object','properties':{}})
        repair = {'name':'commulingo_pipeline_no_edit','input_schema':{
            'type':'object','additionalProperties':False,'properties':{'repairs':{'type':'array'}},'required':['repairs']}}
        saved = []
        async def edit(repairs):
            saved.extend(repairs)
            return 'validated'
        async def chat(*args,**kwargs):
            self.assertIn(repair['name'],kwargs['terminal_tools'])
            self.assertIn(repair['name'],kwargs['finalization_tools'])
            with patch('tool_gateway.security.audit'):
                result, failed = await execute_tool(repair['name'],{'repairs':['fixture']},
                    kwargs['tool_handlers'],tool_schema=repair)
                self.assertFalse(failed,result)
        binding = SimpleNamespace(chat=chat,client=None,model='fixture',render_provider='deepseek',reasoning={})
        with patch('bot_config.resolve_agent_tool_loop',return_value=binding):
            await model_call(spec=spec('research'),prompt='Fixture',tool=tool,handler=AsyncMock(),
                reads=set(),usage=Usage(),budget=.2,local_tools=[(repair,edit,True)])
        self.assertEqual(saved,['fixture'])

    async def test_partial_submission_saves_progress_without_ending_the_stage(self):
        from commulingo.pipeline.stages import model_call, result_tool, StageContinues
        from commulingo.pipeline.prompts import spec
        from commulingo.pipeline.engine import Usage
        from tool_gateway.dispatcher import execute_tool
        tool = result_tool({'type':'object','properties':{'part':{'type':'string'}}})
        calls = []
        async def handler(value):
            calls.append(value['part'])
            if value['part'] == 'first':
                raise StageContinues('Saved to the draft. Still needed before validation: reason.')
            return 'OK: validated'
        usage = Usage()
        async def chat(*args, **kwargs):
            with patch('tool_gateway.security.audit'):
                first, failed = await execute_tool(tool['name'], {'part':'first'}, kwargs['tool_handlers'], tool_schema=tool)
                self.assertTrue(failed, first)  # protocol must keep the terminal loop open
                self.assertIn('Still needed', first)
                second, failed = await execute_tool(tool['name'], {'part':'second'}, kwargs['tool_handlers'], tool_schema=tool)
                self.assertFalse(failed, second)
                third, failed = await execute_tool(tool['name'], {'part':'third'}, kwargs['tool_handlers'], tool_schema=tool)
                self.assertTrue(failed)
                self.assertIn('stage already completed', third)
        binding = SimpleNamespace(chat=chat,client=None,model='fixture',render_provider='deepseek',reasoning={})
        with patch('bot_config.resolve_agent_tool_loop',return_value=binding):
            await model_call(spec=spec('research'),prompt='Fixture',tool=tool,handler=handler,
                reads=set(),usage=usage,budget=.2)
        self.assertEqual(calls, ['first', 'second'])
        self.assertNotIn('rejections', usage.tracker)

    async def test_model_stage_reruns_on_gpt_when_deepseek_refuses_content(self):
        from commulingo.pipeline.stages import model_call, result_tool
        from commulingo.pipeline.prompts import spec
        from commulingo.pipeline.engine import Usage
        tool = result_tool({'type':'object','properties':{'reason':{'type':'string'}},'required':['reason']})
        saved, seen = [], []
        async def handler(value):
            saved.append(value)
            return 'OK'
        async def refuse(*args,**kwargs):
            raise RuntimeError("Error code: 400 - {'error': {'message': 'Content Exists Risk'}}")
        async def accept(messages,**kwargs):
            self.assertEqual(messages[0]['role'],'user')
            await kwargs['tool_handlers'][tool['name']](reason='sourced')
        def resolve(agent_spec,policy):
            seen.append(agent_spec.provider)
            if agent_spec.provider=='openai':
                self.assertEqual(agent_spec.model,'gpt6')
            return SimpleNamespace(chat=refuse if agent_spec.provider=='deepseek' else accept,
                                   client=None,model='fixture',render_provider=agent_spec.provider,reasoning={})
        usage = Usage()
        with patch('bot_config.resolve_agent_tool_loop',side_effect=resolve):
            await model_call(spec=replace(spec('research'),provider='deepseek'),prompt='Fixture',tool=tool,
                             handler=handler,reads=set(),usage=usage,budget=.2,job={'id':1,'payload':{}})
        self.assertEqual(seen,['deepseek','openai'])
        self.assertEqual(saved,[{'reason':'sourced'}])
        self.assertEqual(usage.tracker['provider_fallback'],'openai')
        self.assertEqual(usage.tracker['model_calls'],2)
        # A job that already fell back starts on GPT; any other error surfaces unchanged.
        seen.clear()
        with patch('bot_config.resolve_agent_tool_loop',side_effect=resolve):
            await model_call(spec=replace(spec('research'),provider='deepseek'),prompt='Fixture',tool=tool,
                             handler=handler,reads=set(),usage=Usage(),budget=.2,
                             job={'id':1,'payload':{'provider_fallback':'openai'}})
        self.assertEqual(seen,['openai'])
        async def other(*args,**kwargs):
            raise RuntimeError('upstream HTTP 500')
        with patch('bot_config.resolve_agent_tool_loop',return_value=SimpleNamespace(
                chat=other,client=None,model='fixture',render_provider='deepseek',reasoning={})):
            with self.assertRaisesRegex(RuntimeError,'HTTP 500'):
                await model_call(spec=replace(spec('research'),provider='deepseek'),prompt='Fixture',tool=tool,
                                 handler=handler,reads=set(),usage=Usage(),budget=.2)

    def test_term_fact_echoes_are_dropped_but_real_changes_kept(self):
        # End to end through the editor: test_commulingo_editor
        # EditorContractTests.test_echoed_term_years_never_reach_validation.
        from commulingo.pipeline.stages import drop_unchanged_term_facts
        current = {'startYear':2023,'endYear':None,'period':{'ko':'2023년–현재','en':'2023–present'}}
        fields = {'definition':{'ko':'정의','en':'Definition'},'startYear':2023,'endYear':None,
                  'period':{'ko':'2023년–현재','en':'2023–present'}}
        drop_unchanged_term_facts(fields,current,'update')
        self.assertEqual(set(fields),{'definition'})
        changed = {'startYear':2018,'endYear':None,'period':{'ko':'개념','en':'Concept'}}
        drop_unchanged_term_facts(changed,current,'update')
        self.assertEqual(set(changed),{'startYear','period'})
        cleared = {'startYear':None}
        drop_unchanged_term_facts(cleared,current,'update')
        self.assertEqual(set(cleared),{'startYear'})
        created = {'term':{'ko':'용어','en':'Term'},'startYear':None,'endYear':None,'period':{'ko':'개념','en':'Concept'}}
        drop_unchanged_term_facts(created,{},'create')
        self.assertEqual(set(created),{'term','period'})


