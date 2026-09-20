"""Patch-centred workflow over the existing durable queue, leases and budget."""
import asyncio
from copy import deepcopy

from . import service
from .editor import Editor
from .engine import Result
from .patches import patch_hash, changes
from .bundles import advance


class Review:
    uses_llm = True

    def __init__(self, store=None):
        self.store = store

    async def __call__(self, job, artifacts, usage, budget):
        from .stages import latest, current_artifacts, write_request, model_call, stage_evidence, READS
        from agents.commulingo_reviewer import COMMULINGO_REVIEWER
        from scripts.commulingo_person_reviewer import make_handlers, review_risks
        from runtime_tools.commulingo_review_policy import DECISION_TOOL
        from runtime_tools.registry import TOOL_HANDLERS
        draft = latest(artifacts,'draft')
        research = latest(artifacts,'research')
        current = await asyncio.to_thread(service.call, {'command':'read','target':job['kind'],'id':job['target']})
        if (current or {}).get('revision','') != research.get('baseline',''):
            return Result({'reason':'revision changed'}, 'research')
        digest = patch_hash(write_request(job,draft))
        previous = [a['value'] for a in current_artifacts(artifacts)
                    if a['stage']=='review' and a['value'].get('decision')=='revise']
        if any(v.get('reviewed_patch_hash')==digest for v in previous):
            return Result({'hold_reason':'review requested correction but patch and evidence are unchanged',
                           'patch_hash':digest}, 'complete', 'escalated')
        proposal = {'target_type':draft.get('target',job['kind']),'action':draft.get('action',job['action']),
                    'target_id':job['target'],'patch_json':draft['fields'],'source_refs':draft['sources']}
        proposal['risks'] = review_risks(proposal,current)
        if draft.get('classification'):
            proposal['classification'] = draft['classification']
            proposal['risks'].append('Verify Jev-assigned classification, particularly any low_confidence decisions.')
        snapshots, box = {}, {}
        from .decisions import Decisions
        from .citation_gate import check_review_checks, annotate
        decisions = Decisions(job,current,None,usage)
        async def gate(value):
            checks = value.get('checks',[])
            return {**value,'checks':annotate(checks,await check_review_checks(checks,usage=usage,
                decide=decisions.decide,cache=decisions.citations))}
        from .fetch_backoff import FetchBackoff
        from .store import Store
        backoff = FetchBackoff(self.store if self.store is not None else Store(), job, usage, artifacts)
        handlers = make_handlers({k:backoff.wrap(k,TOOL_HANDLERS[k]) for k in READS},proposal,snapshots,box,gate=gate)
        tool = deepcopy(DECISION_TOOL)
        tool['input_schema']['properties'].update({
            'required_corrections':{'type':'array','items':{'type':'object','additionalProperties':False,
                'properties':{'path':{'type':'string','pattern':'^/fields/'},
                              'reason':{'type':'string','minLength':10}}, 'required':['path','reason']}},
            'optional_suggestions':{'type':'array','items':{'type':'string'}}})
        tool['input_schema']['required'] += ['required_corrections','optional_suggestions']

        async def finish(value):
            corrections = value.get('required_corrections', [])
            if value['decision']=='revise' and not corrections:
                raise ValueError('revise requires specific factual corrections; optional suggestions alone warrant approval')
            if value['decision']=='approve' and corrections:
                raise ValueError('approval cannot contain required corrections')
            for correction in corrections:
                key = correction['path'].split('/')[2].replace('~1','/').replace('~0','~')
                if key not in draft['fields']:
                    raise ValueError('required correction must identify a field in the reviewed patch')
            decision = {k:v for k,v in value.items() if k not in {'required_corrections','optional_suggestions'}}
            result = await handlers[DECISION_TOOL['name']](**decision)
            box.update(required_corrections=corrections, optional_suggestions=value.get('optional_suggestions',[]),
                       reviewed_patch_hash=digest, baseline=research.get('baseline',''), editor_version=2)
            if value['decision']=='approve':
                box['approved_patch_hash'] = digest
            return result

        previous_drafts = [a['value']['draft'] for a in current_artifacts(artifacts)
                           if a['value'].get('editor_version')==2 and a['value'].get('draft')]
        delta = changes(current, draft['fields'])
        since_review = changes(previous_drafts[-2]['fields'], draft['fields']) if len(previous_drafts)>1 else delta
        from .review_context import context, context_tool
        compact = context(proposal, current, previous_drafts[-2]['fields'] if len(previous_drafts)>1 else None)
        shared = {'issues':draft.get('issues',[]), 'issue_results':draft.get('issue_results',[]),
                  'previous_reviews':previous}
        old_context = {'suggestion':proposal,'current':current,'changes':delta,
                       'changes_since_previous_patch':since_review, **shared}
        compact.update(shared)
        usage.tracker['review_context_original_chars'] = len(stage_evidence(old_context))
        usage.tracker['review_context_chars'] = len(stage_evidence(compact))
        prompt = ('Independently verify the changed facts, bilingual equivalence and their original sources. '
            'Fetch the relevant originals yourself; author excerpts are leads, not independent confirmation. '
            'Review the actual delta. Do not expand the article or request stylistic rewrites. '
            'On re-review first check previous required corrections and newly changed facts; preserve prior '
            'accepted conclusions unless new conflicting evidence appears. Required corrections are only material '
            'factual errors, unsupported core assertions or bilingual contradictions. Put optional improvements '
            'in optional_suggestions; they must not prevent approval. If an unchanged field is included solely '
            'to attach missing evidence, verify that evidence without requiring additional prose. '
            'Proposed values appear once in suggestion.patch_json; changes lists their old values by JSON pointer. '
            'Use commulingo_pipeline_review_context to inspect other current fields for contradictions or duplicate sections. '
            'Missing context is not evidence of absence.\n' + stage_evidence(compact))
        await model_call(spec=COMMULINGO_REVIEWER,prompt=prompt,tool=tool,handler=finish,reads=READS,
            read_wrap=lambda name,call:handlers[name],usage=usage,budget=budget,
            scope_id=f'commulingo_pipeline:{job["id"]}:review',job=job,
            local_tools=[context_tool(current)])
        if box['decision']=='revise':
            return Result(box, 'draft')
        return Result(box, 'submit' if box['decision']=='approve' else 'complete',
                      'ready' if box['decision']=='approve' else 'escalated')


async def validate(job, artifacts, usage, budget):
    """Compatibility resume for queued legacy drafts; schema failures return to the editor."""
    from .stages import latest, write_request
    draft = latest(artifacts,'draft')
    try:
        result = await asyncio.to_thread(service.call, {'command':'validate', **write_request(job,draft)})
        return Result(result, 'review')
    except ValueError as exc:
        return Result({'error':str(exc)}, 'research' if 'revision_conflict' in str(exc) else 'draft')


async def publish(job, artifacts, usage, budget):
    from .stages import latest, write_request, review_note_checks
    from .config import load
    from .store import Store
    config = load()
    if config['phase']=='draft':
        raise ValueError('publication disabled during draft evaluation')
    draft, decision = latest(artifacts,'draft'), latest(artifacts,'review')
    request = write_request(job,draft)
    digest = patch_hash(request)
    if decision.get('decision')!='approve' or decision.get('approved_patch_hash')!=digest:
        return Result({'reason':'independent approval does not bind this exact patch'}, 'review')
    if config['phase']=='canary':
        await asyncio.to_thread(Store().publication_slot,job,config['canary_per_group_per_day'])
    request.update(command='publish', approvedPatchHash=digest,
        review={'decision':'approve','reason':decision['reason'],'checks':review_note_checks(decision['checks'])},
        notes=draft.get('notes',''), jobRef=f'job {job["id"]}',
        idempotencyKey=f'pipeline:{job["id"]}:publish:{digest}')
    deferred = [i for i in draft.get('issue_results',[]) if i['status']=='deferred']
    if deferred:
        request['notes'] = (request['notes'] + '\nDeferred issues:\n' + '\n'.join(
            f"{i['id']}: {i['reason']}" for i in deferred)).strip()[:4000]
    replaced = (job.get('payload') or {}).get('replaces_suggestion_id')
    if replaced:
        request['replacesSuggestionId'] = replaced
    try:
        receipt = await asyncio.to_thread(service.call, request)
    except ValueError as exc:
        if 'revision_conflict' not in str(exc):
            raise
        return Result({'error':str(exc)}, 'research')
    if receipt.get('status')!='approved':
        raise ValueError('atomic publication returned no approved receipt')
    value = advance(job, {**receipt,'patch_hash':digest,
        'resolved_issues':[i['id'] for i in draft.get('issue_results',[]) if i['status']=='resolved'],
        'deferred_issues':[i for i in draft.get('issue_results',[]) if i['status']=='deferred']})
    return Result(value,'research' if value.get('remaining_topics') else 'complete',
                  'ready' if value.get('remaining_topics') else 'complete')


def stages(store):
    from .stages import Discover, judge
    editor = Editor(store)
    return {'discover':Discover(), 'research':editor, 'draft':editor, 'judge':judge,
            'validate':validate, 'review':Review(store), 'submit':publish}


def routed_stages(store, legacy, preferred):
    """A durable job never falls back to two-RPC publication on a later tick."""
    if preferred not in {'legacy','editor'}:
        raise ValueError('unknown editorial workflow')
    editor = stages(store)
    routed = {}
    async def select(job, stage):
        selected = (job.get('payload') or {}).get('workflow', preferred)
        if selected not in {'legacy','editor'}:
            raise ValueError('unknown job workflow')
        if (selected=='editor' and not (job.get('payload') or {}).get('workflow')
                and stage not in {'research','draft','discover'}):
            selected = 'legacy'
        if selected=='editor' and not (job.get('payload') or {}).get('workflow'):
            await asyncio.to_thread(store.pin_editor_workflow,job)
            job['payload'] = {**job.get('payload',{}),'workflow':'editor'}
        return (editor if selected=='editor' else legacy)[stage]

    for name in legacy:
        async def execute(job, artifacts, usage, budget, stage=name):
            selected = await select(job, stage)
            return await selected(job,artifacts,usage,budget)
        async def prepare(job, artifacts, usage, stage=name):
            selected = await select(job, stage)
            check = getattr(selected, 'prepare', None)
            return await check(job, artifacts, usage) if check else None
        execute.prepare = prepare
        execute.uses_llm = getattr(legacy[name], 'uses_llm', False)
        routed[name] = execute
    return routed
