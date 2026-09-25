"""Shared stage helpers, the model-call runner, discovery and the no-edit judge.

The editor workflow (editor.py, workflow.py) owns authoring, review and
publication; the legacy two-RPC Research/Draft/Review/submit stages were
removed on 2026-09-24.
"""
import asyncio
from copy import deepcopy
from dataclasses import replace
import logging
import re

from .engine import Result
from . import service
from .bundles import work_topics, advance

logger = logging.getLogger(__name__)

# DeepSeek's server-side input filter rejects the whole request (HTTP 400) once
# the context holds politically sensitive source text (job #489 "decoupling":
# zh.wikipedia 中美关系 + trade-war pages). Retrying the same sources fails the
# same way, so the stage reruns once on GPT and the job stays on GPT afterwards.
CONTENT_RISK = 'Content Exists Risk'
FALLBACK_PROVIDER = 'openai'
# Terra, not Luna: the stage has to read the refused sources and judge them.
FALLBACK_MODEL = 'gpt6'

READS = {'wiki_search','wiki_get','web_search','fetch_url','commulingo_people'}


def latest(artifacts, stage):
    for artifact in reversed(current_artifacts(artifacts)):
        value = artifact['value']
        if value.get('editor_version') == 2 and stage in {'research','draft'} and stage in value:
            return value[stage]
        if artifact['stage'] == stage:
            return value
    return {}


def current_artifacts(artifacts):
    for index in range(len(artifacts)-1,-1,-1):
        if artifacts[index]['value'].get('remaining_topics'):
            return artifacts[index+1:]
    return artifacts


def write_request(job, draft):
    return {'target':draft.get('target',job['kind']),
            'action':draft.get('action',job['action']), 'id':job['target'],
            'fields':draft['fields'],'sources':draft['sources'],'changedBy':'commulingo-pipeline'}


TERM_FACT_FIELDS = ('startYear','endYear','period')

def drop_unchanged_term_facts(fields, current, action):
    """Remove year/period keys the draft merely echoed.

    The term service demands evidence for every fact key present in the edit,
    including ``endYear: null``. Drafts copy the current years back (or fill
    the nullable keys with null), research cannot ground "no end year", and
    the job cycles validate→research→draft until it escalates (#1470, #1707,
    #2049, #16869). A value equal to the current record is no edit; a null on
    create is the column default. A real change still needs its claim.
    """
    for field in TERM_FACT_FIELDS:
        if field not in fields:
            continue
        value = fields[field]
        if action=='update' and field in current and value==current[field]:
            del fields[field]
        elif action=='create' and value is None:
            del fields[field]


# Throwaway text a model submits while probing the tool. Such a reason closed
# jobs as sources_unavailable (#42, #120), and such a claim carried over into
# every later attempt of #120.
PROBE_RE = re.compile(r'\W*(placeholder|probe|test|testing|checking|investigating|interim|in progress|'
                      r'todo|tbd|dummy|lorem)\b', re.I)


class StageContinues(Exception):
    """A terminal tool call that saved progress without finishing the stage.

    Not a rejection: the author split a large submission across calls.
    """


def is_probe(text):
    return bool(PROBE_RE.match(str(text or ''))) or 'not a real submission' in str(text or '').lower()


def result_tool(schema):
    return {'name':'commulingo_pipeline_result',
            'description':'Save this stage result. This does not publish dictionary content.',
            'input_schema':schema}


def stage_evidence(payload):
    from llm.execution_context import context_record, render_context_records
    return render_context_records([context_record(
        'stage_artifacts', 'commulingo_pipeline_store', payload,
        coverage='source snapshots and draft/review artifacts; stage completion is not publication',
    )])


def content_risk(exc):
    return CONTENT_RISK in str(exc)


async def model_call(*, spec, prompt, tool, handler, reads, usage, budget, read_wrap=None, scope_id=None, read_tools=None, max_rounds=12, job=None, local_tools=()):
    from bot_config import resolve_agent_tool_loop
    from runtime_tools.registry import TOOLS, TOOL_HANDLERS
    from tool_gateway.inference import resolve_agent_inference_policy
    from tool_gateway.security import caller_scope, new_run_context
    from tool_gateway.results import ToolRejection
    fallback = ((job or {}).get('payload') or {}).get('provider_fallback')
    if fallback:
        spec = replace(spec,provider=fallback,model=FALLBACK_MODEL)
        usage.tracker['provider_fallback'] = fallback
    policy = resolve_agent_inference_policy(spec)
    tools = [deepcopy((read_tools or {}).get(t['name'],t)) for t in TOOLS if t['name'] in reads]
    handlers = {name: TOOL_HANDLERS[name] for name in reads}
    if read_wrap:
        handlers = {name:read_wrap(name,h) for name,h in handlers.items()}
    completed = False
    rejections = []
    usage.tracker['model_calls'] = usage.tracker.get('model_calls',0) + 1
    async def terminal(**value):
        nonlocal completed
        if completed:
            raise ToolRejection('stage already completed')
        try:
            result = await handler(value)
        except StageContinues as progress:
            return str(progress)
        except ValueError as exc:
            rejections.append(str(exc))
            usage.tracker.setdefault('rejections', []).append(str(exc)[:500])
            raise ToolRejection(str(exc)) from exc
        completed = True
        return result
    handlers[tool['name']] = terminal
    terminal_names = [tool['name']]
    for definition, callback, is_terminal in local_tools:
        tools.append(definition)
        if is_terminal:
            async def extra_terminal(_callback=callback, **value):
                nonlocal completed
                if completed:
                    raise ToolRejection('stage already completed')
                try:
                    result = await _callback(**value)
                except ValueError as exc:
                    rejections.append(str(exc))
                    usage.tracker.setdefault('rejections', []).append(str(exc)[:500])
                    raise ToolRejection(str(exc)) from exc
                completed = True
                return result
            handlers[definition['name']] = extra_terminal
            terminal_names.append(definition['name'])
        else:
            handlers[definition['name']] = callback
    context = new_run_context(interface='autonomous',agent_name=spec.name,is_owner=True,
                              scope_type='maintenance_job',scope_id=scope_id or 'commulingo_pipeline:standalone')
    # Artifact handlers populate an in-memory box; they must run in every attempt.
    # Durable public-write idempotency belongs to the RPC receipt, not this tool.
    context = replace(context,task_id=None,session_id=None)
    from llm.execution_context import attach_context, context_record
    def messages():
        # The loop appends to this list, so every run starts from a fresh copy.
        return attach_context([{'role':'user','content':prompt}], [context_record(
            'pipeline_stage', 'commulingo_pipeline_runtime', {
                'terminal_tools': terminal_names,
                'stage_result': 'not recorded yet',
                'publication': 'not implied by stage completion; only submit/review receipts establish application',
                'evidence': 'source IDs/hash/ranges and baseline revision belong to the supplied artifacts; do not invent or refresh them',
            }, scope=scope_id or 'commulingo_pipeline:standalone', temporal_scope='current stage',
        )])
    async def run(spec):
        binding = resolve_agent_tool_loop(spec,policy)
        with caller_scope(context):
            usage.started = True
            await binding.chat(messages(),
                client=binding.client,model=binding.model,tools=[*tools,tool],tool_handlers=handlers,
                system_prompt=spec.render_prompt(provider=binding.render_provider),
                max_rounds=min(policy.max_rounds,max_rounds),max_tokens=policy.max_output_tokens,
                max_input_tokens=policy.max_input_tokens,budget_usd=budget,budget_tracker=usage.tracker,
                agent_name=spec.name,finalization_tools=terminal_names,terminal_tools=terminal_names,
                terminal_required=True,continue_on_length=policy.max_output_continuations > 0,
                max_length_continuations=policy.max_output_continuations,**binding.reasoning)
    jev_before = usage.tracker.get('jev_cost_usd',0)
    auxiliary_before = usage.tracker.get('auxiliary_cost_usd',0)
    observed_before = usage.tracker.get('observed_llm_cost_usd',0)
    cost_before = usage.tracker.get('total_cost',0)
    try:
        await run(spec)
    except Exception as exc:
        if completed or not content_risk(exc) or spec.effective_provider()==FALLBACK_PROVIDER:
            raise
        logger.warning('%s: %s refused the stage input (%s); rerunning on %s',
                       scope_id, spec.effective_provider(), CONTENT_RISK, FALLBACK_PROVIDER)
        usage.tracker['provider_fallback'] = FALLBACK_PROVIDER
        usage.tracker['model_calls'] = usage.tracker.get('model_calls',0) + 1
        await run(replace(spec,provider=FALLBACK_PROVIDER,model=FALLBACK_MODEL))
    finally:
        # Provider loops replace total_cost on return; add sidecar decisions
        # afterwards so their usage is neither lost nor charged twice.
        # LoopState accumulates received responses even across a provider
        # fallback; adapter finalization otherwise replaces the first cost.
        loop_cost = (cost_before + usage.tracker['observed_llm_cost_usd'] - observed_before
                     if 'observed_llm_cost_usd' in usage.tracker else usage.tracker.get('total_cost',0))
        usage.tracker['total_cost'] = loop_cost + (
            usage.tracker.get('jev_cost_usd',0)-jev_before +
            usage.tracker.get('auxiliary_cost_usd',0)-auxiliary_before)
    usage.complete = True
    if not completed:
        detail = '; '.join(dict.fromkeys(rejections[-3:]))
        raise RuntimeError('stage ended without validated result; collected sources are retained'
                           + (f'; last rejections: {detail}' if detail else ''))


async def judge(job, artifacts, usage, budget):
    research = latest(artifacts,'research')
    if (job.get('payload') or {}).get('replaces_suggestion_id'):
        return Result({'research':research,'hold_reason':'correction research produced no supported edit'},'complete','escalated')
    if research.get('current'):
        try:
            for topic in work_topics(job):
                suffix = ':'+topic if (job.get('payload') or {}).get('topics') else ''
                await asyncio.to_thread(service.call,{'command':'enrichment','target':job['kind'],
                    'id':job['target'],'topic':topic,'status':research['status'],'reason':research['reason'],
                    'sources':research['inspected_sources'],'expectedRevision':research['baseline'],
                    'idempotencyKey':f'pipeline:{job["id"]}:{len(artifacts)}:enrichment{suffix}'})
        except ValueError as exc:
            if 'revision_conflict' in str(exc):
                return Result({'reason':str(exc)},'research')
            raise
    if research['status']=='sources_unavailable':
        return Result({'status':research['status']},'research','deferred',90*86400)
    value = advance(job, {'status':research['status']})
    return Result(value,'research' if value.get('remaining_topics') else 'complete',
                  'ready' if value.get('remaining_topics') else 'complete')


class Discover:
    uses_llm = True

    async def __call__(self, job, artifacts, usage, budget):
        from .prompts import spec as stage_spec
        spec = stage_spec('discover')
        from db import query_one
        schema = {'type':'object','additionalProperties':False,'properties':{'candidates':{
            'type':'array','maxItems':4,'items':{'type':'object','additionalProperties':False,
                'properties':{'kind':{'type':'string','enum':['person','term']},
                    'target':{'type':'string','pattern':'^[a-z0-9]+(?:-[a-z0-9]+)*$'},
                    'label':{'type':'string','minLength':2},'mention':{'type':'string','minLength':2},
                    'reason':{'type':'string','minLength':20}},
                'required':['kind','target','label','mention','reason']}}},'required':['candidates']}
        box = {}
        from .config import load
        overlap_allow = load()['term_event_overlap_allow']
        explicit_gap = job['payload']['material_id'].startswith('gap:')
        if explicit_gap:
            # The runner already knows kind, label and mention for a requested
            # entry. They stay optional free strings and are overwritten below:
            # enum-exact copies were rejected five times per run and the model
            # then gave up with an empty result, stranding 31 requests (2026-09-17).
            schema['properties']['candidates']['maxItems'] = 1
            props = schema['properties']['candidates']['items']['properties']
            for key in ('kind', 'label', 'mention'):
                props[key] = {'type':'string'}
            required = schema['properties']['candidates']['items']['required']
            schema['properties']['candidates']['items']['required'] = [k for k in required if k not in {'kind','label','mention'}]
            schema['properties']['reason'] = {'type':'string','minLength':20,
                'description':'Required when candidates is empty: why the requested entry should not be registered.'}
        async def finish(value):
            accepted = []
            if explicit_gap and len(value['candidates'])>1:
                raise ValueError('an explicit gap commissions only one requested entry')
            if explicit_gap and not value['candidates']:
                if len((value.get('reason') or '').strip()) < 20:
                    raise ValueError('an empty result for a requested entry needs a reason of at least 20 characters')
                box['skip_reason'] = value['reason'].strip()
            for candidate in value['candidates']:
                if explicit_gap:
                    candidate = {**candidate, 'kind':job['payload']['requested_kind'],
                                 'label':job['payload']['label'], 'mention':job['payload']['label']}
                if candidate['mention'] not in job['payload']['body']:
                    raise ValueError('candidate mention must occur exactly in this material')
                table,aliases,foreign,label = ('commulingo_people','commulingo_person_aliases','person_id','name') if candidate['kind']=='person' else ('commulingo_terms','commulingo_term_aliases','term_id','term')
                existing = await asyncio.to_thread(query_one,f'''SELECT id FROM {table}
                    WHERE id=%(id)s OR lower({label}_ko)=lower(%(label)s) OR lower({label}_en)=lower(%(label)s)
                    UNION SELECT {foreign} FROM {aliases} WHERE lower(alias)=lower(%(label)s) LIMIT 1''',
                    {'id':candidate['target'],'label':candidate['label']})
                if candidate['kind']=='term' and candidate['target'] not in overlap_allow:
                    from .store import EVENT_TITLE_MATCH_SQL
                    event = await asyncio.to_thread(query_one, 'SELECT 1 AS hit WHERE ' + EVENT_TITLE_MATCH_SQL.format(
                        label_ko='%(label)s', label_en='%(label)s'), {'label':candidate['label']})
                    if event:
                        continue  # the events lane owns this name; not a glossary entry
                if not existing:
                    accepted.append(candidate)
            box['candidates'] = accepted
            return 'OK: candidates recorded; no public content changed'
        commission = ('DISCOVERY ONLY: check only the explicitly requested entry. Return zero or one candidate; '
            f'kind={job["payload"]["requested_kind"]!r}, label and mention={job["payload"]["label"]!r}. '
            'The target is a lowercase hyphenated dictionary slug, never a gap ID. '
            'The runner supplies kind, label and mention; submit target and reason only. Do not propose neighboring names or concepts. '
            'If the entry should not be registered (already present under another name, not a dictionary subject), '
            'return empty candidates with a top-level reason. '
            if explicit_gap else 'DISCOVERY ONLY: identify up to four historically useful missing people or concept terms ')
        prompt = (commission +
            'explicitly mentioned in this public material. Check current dictionary aliases. '
            'Do not register events or institutions as concept terms. Empty candidates is valid. '
            'The runner will research and independently review each accepted candidate later.\n'
            + stage_evidence(job['payload']))
        await model_call(spec=spec,prompt=prompt,tool=result_tool(schema),handler=finish,
                         reads={'commulingo_people'},usage=usage,budget=budget,
                         scope_id=f'commulingo_pipeline:{job["id"]}:discover',job=job)
        return Result(box,'complete','complete')


def review_note_checks(checks):
    """Checks as the editorial note shows them: the citation gate's verdict
    numbers stay in the artifact, not in the suggestion history."""
    return [{k:v for k,v in c.items() if k!='citation_check'} if isinstance(c,dict) else c for c in checks]


def prose_problem(fields):
    from commulingo.people import _em_dash_problem, _script_leak_problem, _contains_north_korea
    prose = {k:v for k,v in fields.items() if k not in {'evidence','sources'}}
    return '; '.join(e for e in (_em_dash_problem(prose), _script_leak_problem(prose),
        'Use 조선민주주의인민공화국 or 조선 in Korean text' if _contains_north_korea(prose) else None) if e)


def stages(store, workflow='editor'):
    from .workflow import routed_stages
    return routed_stages(store, workflow)
