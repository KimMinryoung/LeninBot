"""Independent review session of the CommuLingo pipeline (worker kind
commulingo_review). The queue, validation and publication run in the frontend."""
import asyncio
import logging

from . import service
from .engine import Result
from .patches import patch_hash, changes
from .bundles import work_topics

class Review:
    uses_llm = True

    def __init__(self, store=None):
        self.store = store

    async def __call__(self, job, artifacts, usage, budget):
        from .stages import latest, current_artifacts, write_request, model_call, stage_evidence, READS
        from agents.commulingo_reviewer import COMMULINGO_REVIEWER
        from commulingo.review_handlers import make_handlers, review_risks
        from commulingo.review_policy import DECISION_TOOL, decision_tool
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
            await self.leave_note(job, artifacts, 'revise (held)', previous[-1].get('reason',''))
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
        tool = decision_tool(proposal)
        tool['input_schema']['properties'].update({
            'required_corrections':{'type':'array','items':{'type':'object','additionalProperties':False,
                'properties':{'path':{'type':'string','pattern':'^/fields/',
                    'description':'Select a field in this proposed patch: ' + ', '.join('/fields/' + k.replace('~','~0').replace('/','~1') for k in draft['fields'])},
                              'reason':{'type':'string','minLength':10}}, 'required':['path','reason']}},
            'optional_suggestions':{'type':'array','items':{'type':'string'}},
            'coverage': {'type':'object','additionalProperties':False,
                'properties': {'sufficient':{'type':'boolean'},
                    'reason':{'type':'string','minLength':20,
                        'description':'Assess whether the title and commissioned topic are answered, using the original sources and existing text. Identify material omissions, not optional expansion.'}},
                'required':['sufficient','reason']}})
        tool['input_schema']['required'] += ['required_corrections','optional_suggestions','coverage']

        async def finish(value):
            corrections = value.get('required_corrections', [])
            coverage = value['coverage']
            if not coverage['sufficient'] and (value['decision']=='approve' or not corrections):
                raise ValueError('Material omissions require non-approval and evidence-backed required_corrections.')
            if value['decision']=='revise' and not corrections:
                raise ValueError('revise requires specific factual corrections; optional suggestions alone warrant approval')
            if value['decision']=='approve' and corrections:
                raise ValueError('approval cannot contain required corrections')
            for correction in corrections:
                key = correction['path'].split('/')[2].replace('~1','/').replace('~0','~')
                if key not in draft['fields']:
                    raise ValueError('required correction must identify a field in the reviewed patch')
            decision = {k:v for k,v in value.items() if k not in {'required_corrections','optional_suggestions','coverage'}}
            result = await handlers[DECISION_TOOL['name']](**decision)
            box.update(coverage=coverage, required_corrections=corrections, optional_suggestions=value.get('optional_suggestions',[]),
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
            'Bilingual equivalence means the proposed ko and en values of each field state the same claims; '
            'the languages of the sources are irrelevant. An empty, placeholder or partial translation, or a '
            'claim present in only one language, is a required correction. '
            'On re-review first check previous required corrections and newly changed facts; preserve prior '
            'accepted conclusions unless new conflicting evidence appears. Required corrections are only material '
            'factual errors, unsupported core assertions, bilingual contradictions, or material omissions. '
            'Assess coverage explicitly: does the body answer its title and commissioned topic, and add '
            'useful supported information beyond the existing biography? A missing central role, action, '
            'outcome or causal link that makes the account misleading requires correction backed by a '
            'retrieved source and an existing /fields/... path. Archive catalog descriptions and general '
            'caveats cannot substitute for the subject’s documented actions. Do not demand length, '
            'extra background or unsupported speculation. Put optional improvements '
            'in optional_suggestions; they must not prevent approval. If an unchanged field is included solely '
            'to attach missing evidence, verify that evidence without requiring additional prose. '
            'Proposed values appear once in suggestion.patch_json; changes lists their old values with '
            '/fields/... correction paths. Copy those paths into required_corrections. '
            'Use commulingo_pipeline_review_context to inspect other current fields for contradictions or duplicate sections. '
            'Missing context is not evidence of absence.\n' + stage_evidence(compact))
        await model_call(spec=COMMULINGO_REVIEWER,prompt=prompt,tool=tool,handler=finish,reads=READS,
            read_wrap=lambda name,call:handlers[name],usage=usage,budget=budget,
            scope_id=f'commulingo_pipeline:{job["id"]}:review',job=job,
            local_tools=[context_tool(current, proposal["target_type"])])
        if box['decision']=='revise':
            return Result(box, 'draft')
        if box['decision']!='approve':
            await self.leave_note(job, artifacts, box['decision'], box.get('reason',''))
        # reject is a finished verdict; only escalate asks for a human.
        return Result(box, 'submit' if box['decision']=='approve' else 'complete',
                      'ready' if box['decision']=='approve' else 'escalated' if box['decision']=='escalate' else 'complete')

    async def leave_note(self, job, artifacts, decision, reason):
        """A job that ends without publishing tells the entry's next author why.

        Without it the verdict lives only in this job's review artifact, and the
        next commission on the same entry runs into the same unresolved conflict.
        """
        text = f'검토 {decision} (작업 {job["id"]}, {work_topics(job)}): {(reason or "").strip()}'[:4000]
        try:
            await asyncio.to_thread(service.call,{'command':'note','target':job['kind'],'id':job['target'],
                'note':text,'changedBy':'commulingo-pipeline-reviewer','jobRef':f'job {job["id"]} review',
                'idempotencyKey':f'pipeline:{job["id"]}:{len(artifacts)}:review-note'})
        except Exception as exc:  # the verdict itself is already persisted as an artifact
            logging.getLogger(__name__).warning('review note not saved for %s: %s', job['target'], exc)
