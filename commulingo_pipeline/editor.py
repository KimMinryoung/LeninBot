"""One author session reads original pages, drafts, and repairs a minimal patch."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib

from llm.prompt_renderer import SystemPrompt
from . import service
from .bundles import work_topics
from .draft_repair import RepairProtocolError
from .author_draft import AuthorDraft, structured_args
from .diagnostics import prose_errors
from .engine import Result
from .evidence import compile_evidence, resolve_passages, PASSAGE_PATTERN, MAX_PASSAGES
from .issues import commission, FACTS
from .patches import canonical, changes, patch_hash, schema_for
from .source_session import Sources
from .editor_context import RepairReads, prose_budgets, work_status
from .decisions import Decisions

INSTRUCTIONS = """Edit only the commissioned issues with the smallest supported bilingual patch.
Use the input as follows:
- issues: scope and completion criteria. Address required review corrections; optional suggestions are not obligations.
- current, surrounding_context: existing content to preserve. Read notes and sections to avoid duplication.
- tool schemas, prose_budgets: typed changes and output limits, never length targets.
- source_cache: available originals. Cite only displayed P-labels, using the shortest sufficient passages per changed factual field.
- work_status: current research access, submission tool and next action. Follow the latest tool response.
- saved_draft: retained work, not approved content. Resubmit changed fields with their evidence; omitted fields remain saved.
Research only missing or conflicting facts. Submit when the commissioned claims have adequate support,
leaving time to repair validation errors. Put deferred work in private notes, not public prose.
Provide classification labels; the runner assigns codes, citations and revision, then obtains independent review.
"""


class Editor:
    uses_llm = True

    def __init__(self, store):
        self.store = store

    async def prepare(self, job, artifacts, usage):
        """Read/commission once under the engine lease, before any paid reservation."""
        usage.tracker['workflow'] = 'editor'
        from .stages import latest
        current = await asyncio.to_thread(service.call, {'command':'read','target':job['kind'],'id':job['target']})
        if job['action']=='create' and current:
            return Result({'reason':'target already exists'}, 'complete', 'complete')
        if job['action']=='update' and not current:
            return Result({'reason':'target no longer exists'}, 'complete', 'escalated')
        baseline = (current or {}).get('revision','')
        issues = commission(job, current)
        previous_review = latest(artifacts, 'review') or (job.get('payload') or {}).get('review_feedback') or {}
        if previous_review.get('decision')=='revise':
            issues.append({'id':'review-corrections','field':'*','problem':previous_review.get('required_corrections') or previous_review.get('reason'),
                           'done_when':'Resolve the factual errors required by the independent review.'})
        research = {'current':current, 'baseline':baseline, 'issues':issues, 'claims':[], 'inspected_sources':[]}
        if not issues:
            return Result({**research, 'status':'complete', 'reason':'No concrete missing field or evidence defect remains in the commissioned topics.'}, 'judge')
        usage.prepared['editor'] = (current, baseline, issues, previous_review, research)
        return None

    async def __call__(self, job, artifacts, usage, budget):
        from .stages import (current_artifacts, latest, model_call, result_tool, stage_evidence,
                             write_request, prose_problem, is_probe,
                             drop_unchanged_term_facts, READS)
        from .prompts import EDITOR_POLICY
        from agents.commulingo_curator import COMMULINGO_CURATOR
        if 'editor' not in usage.prepared:
            early = await self.prepare(job, artifacts, usage)
            if early is not None:
                return early
        current, baseline, issues, previous_review, research = usage.prepared.pop('editor')
        checkpoint = next((a['value'] for a in reversed(current_artifacts(artifacts))
                           if a['stage']=='editor_checkpoint' and a['value'].get('baseline')==baseline), {})
        session = await Sources.load(self.store, job, usage, checkpoint)
        from .fetch_backoff import FetchBackoff
        session.backoff = FetchBackoff(self.store, job, usage, artifacts)
        catalogs = None
        section = job['kind']=='person' and job['action']=='update' and work_topics(job)==['sections']
        if job['kind']=='person' and not section:
            from runtime_tools.commulingo_classify import load_catalogs
            catalogs = await asyncio.to_thread(load_catalogs)
        decisions = Decisions(job,current,catalogs,usage,deepcopy(checkpoint.get('classification_cache',{})))
        field_schema = decisions.author_schema(schema_for(job, current, catalogs))
        field_schema['properties']['notes'] = {'type':'string','maxLength':4000,
            'description':'Private working notes; moved out of published fields.'}
        reads = RepairReads(session,repair_only=checkpoint.get('repair_only',False))
        reads.missing_fields = list(checkpoint.get('missing_fields', []))
        tool = result_tool({'type':'object','additionalProperties':False,
            'properties': {
                'status':{'type':'string','enum':['ready','complete','not_applicable','sources_unavailable']},
                'reason':{'type':'string','minLength':20},
                'issue_results':{'type':'array','items':{'type':'object','additionalProperties':False,
                    'properties':{'id':{'type':'string','enum':[i['id'] for i in issues]},
                                  'status':{'type':'string','enum':['resolved','deferred']},
                                  'reason':{'type':'string','minLength':10}},
                    'required':['id','status','reason']}},
                'fields':field_schema,
                'claims':{'type':'array','items':{'type':'object','additionalProperties':False,
                    'properties':{'field':{'type':'string','enum':list(field_schema['properties'])},
                        'claim':{'type':'string','minLength':1},
                        'passages':{'type':'array','minItems':1,'maxItems':MAX_PASSAGES,
                                    'items':{'type':'string','pattern':PASSAGE_PATTERN}},
                        'stance':{'type':'string','enum':['supports','disputes']}},
                    'required':['field','claim','passages']}},
                'notes':{'type':'string','maxLength':4000}},
            'required':['status','reason']})
        repair = AuthorDraft(tool, capture_invalid=True)
        repair.draft = deepcopy(checkpoint.get('draft'))
        # Earlier editor checkpoints used sentence arrays for bilingual prose.
        # Joining preserves their text; canonical length checks still apply.
        if repair.draft and structured_args(repair.draft['args']):
            saved_args = repair.draft['args']
            # Older checkpoints asked the author for a slug. The server now
            # owns it, so retain the draft prose and regenerate at validation.
            if section:
                saved_args.get('fields', {}).pop('slug', None)
                saved_args['claims'] = [c for c in saved_args.get('claims', []) if c.get('field') != 'slug']
            nested_notes = saved_args.get('fields', {}).pop('notes', None)
            if isinstance(nested_notes, str):
                saved_args['notes'] = '\n\n'.join(dict.fromkeys(
                    n.strip() for n in (saved_args.get('notes'), nested_notes) if n and n.strip()))
            decisions.strip_assigned(saved_args.get('fields',{}))
            for field, value in repair.draft['args'].get('fields', {}).items():
                properties = field_schema['properties'].get(field, {}).get('properties', {})
                if isinstance(value, dict):
                    for lang in ('ko','en'):
                        if properties.get(lang, {}).get('type')=='string' and isinstance(value.get(lang), list) and all(isinstance(s,str) for s in value[lang]):
                            value[lang] = ' '.join(value[lang])
        previous_patch = latest(artifacts,'draft') or {}
        needed = {i['field'] for i in issues}
        saved_fields = (repair.draft or {}).get('args', {}).get('fields', {})
        saved_fields = saved_fields if isinstance(saved_fields, dict) else {}
        needed.update(saved_fields)
        needed.update(previous_patch.get('fields',{}))
        needed.update(field_schema.get('required',[]))
        if 'role' in needed:
            needed.update({'bio', 'career'})
        if '*' in needed:
            needed = set(field_schema['properties']) - {'aliasEdits', 'careerEdits', 'sceneEdits'}
            needed.update(saved_fields)
        focused_contract = deepcopy(tool['input_schema'])
        focused_contract['properties']['fields']['properties'] = {
            k:v for k,v in field_schema['properties'].items() if k in needed and k != 'notes'}
        repair.configure(focused_contract['properties']['fields'], issues)
        failures = dict(checkpoint.get('failures') or {})
        section_slug_cache = dict(checkpoint.get('section_slug_cache') or {})
        box = {}
        error_kind = checkpoint.get('error_kind', '')
        last_error = repair.author_error(checkpoint.get('error', '') or '')
        def status():
            state = work_status(issues, repair.draft, reads, error=last_error, error_kind=error_kind)
            if repair.draft and not structured_args(repair.draft['args']):
                state.update(submission_tool='commulingo_pipeline_result', next_tool='commulingo_pipeline_result',
                             next_action='Submit a complete replacement for the malformed legacy draft, or use the no-edit tool. History is retained.')
            return state
        reads.status = status

        async def save_checkpoint(error=None):
            if repair.draft or session.passages.shown:
                await asyncio.to_thread(self.store.save_editor_checkpoint, job, {
                    'baseline':baseline, 'draft':repair.draft, 'passages':session.passages.shown,
                    'source_requests':session.requests, 'failures':failures,
                    'error':last_error if error is None else repair.author_error(error),
                    'error_kind':error_kind,
                    'repair_only':reads.repair_only,'missing_fields':reads.missing_fields,
                    'classification_cache':decisions.cache,
                    'section_slug_cache':section_slug_cache})

        async def finish(value, *, update=False):
            nonlocal error_kind, last_error
            error_kind = 'schema'
            try:
                value = repair.prepare(repair.submission(value, update=update))
                error_kind = 'validation'
                await save_checkpoint()
                if is_probe(value['reason']) or any(is_probe(c['claim']) for c in value.get('claims',[])):
                    raise ValueError('reason is a probe or progress note; submit a substantive editorial decision')
                research['inspected_sources'] = sorted({s['url'] for s in session.sources.values()
                    if s.get('body') and s['expires_at']>datetime.now(timezone.utc)})
                outcomes = value.get('issue_results') or []
                problems = []
                if len(outcomes)!=len(issues) or {r['id'] for r in outcomes}!={i['id'] for i in issues}:
                    problems.append('ready requires one issue_results entry per commissioned issue, resolved or deferred with a reason; set /issue_results to the complete list for issue IDs: '
                                    + ', '.join(i['id'] for i in issues))
                fields = deepcopy(value.get('fields') or {})
                nested_notes = fields.pop('notes',None)
                notes = '\n\n'.join(dict.fromkeys(n.strip() for n in (value.get('notes'),nested_notes) if n and n.strip()))
                if len(notes)>4000:
                    raise ValueError('/notes: combined private notes exceed 4000 characters; shorten notes only')
                if not fields:
                    raise ValueError('ready requires a non-empty fields patch')
                submitted_fields = set(fields)
                if job['kind']=='term':
                    drop_unchanged_term_facts(fields, current or {}, job['action'])
                if not fields:
                    raise ValueError('patch has no changes; use commulingo_pipeline_no_edit')
                problem = prose_errors(fields)
                if problem:
                    problems.append(problem)
                try:
                    claims = resolve_passages(value.get('claims') or [], session.passages, session.sources,
                                              draft_paths=True)
                except ValueError as exc:
                    error_kind = 'passages'
                    problems.append(str(exc))
                    claims = []
                # Validate every passage before dropping claims for unchanged
                # term metadata that the runner itself removed from the patch.
                removed_fields = submitted_fields - set(fields)
                claims = [c for c in claims if c['field'] not in removed_fields]
                extras = {c['field'] for c in claims} - set(fields)
                if extras:
                    problems.append('claims must support fields in this patch: ' + ', '.join(sorted(extras)))
                factual = {'body'} if section else FACTS[job['kind']]
                missing = (set(fields) & factual) - {c['field'] for c in claims}
                if missing:
                    if error_kind != 'passages':
                        error_kind = 'missing_evidence'
                    problems.append('evidence required for ' + ', '.join(sorted(missing)) + '; repair those claims in this session')
                reads.missing_fields = sorted(missing)
                reads.repair_only = not missing and bool(claims)
                if missing:
                    usage.tracker['targeted_research'] = sorted(missing)
                    problems.append('Research only the missing fields. All other draft fields and claims remain saved.')
                if problems:
                    raise ValueError('\n'.join(problems))
                from .citation_gate import check_claims, annotate
                try:
                    claims = annotate(claims,await check_claims(claims,session.sources,usage=usage,
                        decide=decisions.decide,cache=decisions.citations))
                except ValueError:
                    reads.repair_only = False
                    error_kind = 'citation'
                    raise
                evidence = compile_evidence(claims, session.sources, set(fields))
                if not evidence:
                    raise ValueError('ready requires original-text evidence for the proposed change')
                resolved = {r['id'] for r in outcomes if r['status']=='resolved'}
                if not resolved:
                    raise ValueError('ready must resolve at least one commissioned issue')
                classification = {}
                if not section:
                    fields, classification = await decisions.classify(fields,claims,session.sources)
                    await save_checkpoint()
                for issue in issues:
                    if issue['id'] in resolved and issue['field']!='*' and issue['field'] not in fields:
                        raise ValueError(f"issue {issue['id']} cannot be resolved without its field or evidence in the patch")
                if section:
                    name = (current or {}).get('name') or {}
                    if any(fields['heading'].get(lang) and fields['heading'][lang].strip()==name.get(lang) for lang in ('ko','en')):
                        raise ValueError('fields.heading must name the section topic, not repeat the person name')
                    original = (job.get('payload') or {}).get('original_proposal') or {}
                    original_slug = (original.get('patch_json') or {}).get('slug')
                    if original and not original_slug:
                        raise ValueError('correction is missing its original section slug')
                    if original_slug:
                        fields['slug'] = original_slug
                    else:
                        slug_key = canonical(fields.get('heading'))
                        if slug_key not in section_slug_cache:
                            from runtime_tools.commulingo_section_slug import generate_section_slug
                            section_slug_cache[slug_key] = await asyncio.to_thread(
                                generate_section_slug, job['target'], fields.get('heading'), fields.get('body'),
                                (current or {}).get('sections', []), usage=usage)
                            await save_checkpoint()
                        fields['slug'] = section_slug_cache[slug_key]
                        if not isinstance(fields.get('sortOrder'), int):
                            from runtime_tools.commulingo_section_slug import section_sort_order
                            fields['sortOrder'] = section_sort_order(
                                fields.get('heading'), (current or {}).get('sections', []))
                    if fields['slug']==job['target']:
                        raise ValueError('generated section slug must name the topic, not the person')
                exists = section and any(s['slug']==fields.get('slug') for s in (current or {}).get('sections', []))
                fields['evidence'] = evidence
                if job['action']=='update':
                    fields['expectedRevision'] = baseline
                candidate = {'fields':fields,'sources':list(dict.fromkeys(e['source'] for e in evidence)),
                             'changes':changes(current, fields), 'issues':issues,
                             'issue_results':outcomes, 'notes':notes}
                candidate['classification'] = classification
                if section:
                    candidate.update(target='person_section', action='update' if exists else 'create')
                usage.tracker['preflight_checks'] = usage.tracker.get('preflight_checks',0)+1
                await asyncio.to_thread(service.call, {'command':'validate', **write_request(job,candidate)})
                usage.tracker['preflight_passed'] = True
                candidate['patch_hash'] = patch_hash(write_request(job,candidate))
                research.update(claims=claims,status='ready',reason=value['reason'])
                box.update(editor_version=2, draft=candidate, research=research)
                return 'OK: validated patch recorded for independent review'
            except ValueError as exc:
                last_error = repair.author_error(str(exc))
                usage.tracker['preflight_failures'] = usage.tracker.get('preflight_failures',0)+1
                if 'revision_conflict' in str(exc):
                    box.update(rebase=True, error=str(exc))
                    return 'Revision changed; restart from current record with cached sources.'
                if isinstance(exc, RepairProtocolError):
                    usage.tracker['repair_protocol_errors'] = usage.tracker.get('repair_protocol_errors',0)+1
                    await save_checkpoint(str(exc))
                    raise RepairProtocolError(reads.with_status(repair.feedback(str(exc)))) from exc
                fingerprint = hashlib.sha256(canonical({'draft':repair.draft,'error':str(exc)}).encode()).hexdigest()
                failures[fingerprint] = failures.get(fingerprint,0)+1
                await save_checkpoint(str(exc))
                if failures[fingerprint]>=2:
                    box.update(hold_reason='same rejected patch and error repeated without progress', error=str(exc))
                    return 'Held: identical failed patch repeated; retained for diagnosis.'
                raise ValueError(reads.with_status(repair.feedback(str(exc)))) from exc

        async def edit(**value):
            return await finish(value, update=True)

        async def no_edit(**value):
            try:
                repair.validate_call(value, repair.no_edit_tool)
                if is_probe(value['reason']) or any(is_probe(item['reason']) for item in value['issues'].values()):
                    raise ValueError('Explain the no-edit decision substantively for every commissioned issue.')
            except ValueError as exc:
                raise ValueError(str(exc) + '; retry commulingo_pipeline_no_edit with the corrected decision.') from exc
            # Preserve the last draft and its evidence for audit/recovery. This
            # decision is a distinct artifact, never a mutation of draft status.
            await save_checkpoint()
            inspected = sorted({source['url'] for source in session.sources.values()
                                if source.get('body') and source['expires_at'] > datetime.now(timezone.utc)})
            box.update(editor_version=2, research={**research, 'status':value['status'],
                'reason':value['reason'], 'inspected_sources':inspected,
                'issue_results':[{'id':key, **item} for key, item in value['issues'].items()]})
            return 'OK: no-edit judgment recorded; saved draft retained in history'

        focused_current = {k:v for k,v in (current or {}).items()
                           if k in needed | {'id','revision','name','term','evidence','notes','sections'}}
        background_fields = ({'definition','original','aliases','period','startYear','endYear'}
                             if job['kind']=='term' else
                             {'givenName','familyName','cyrillic','years','epithet','bio','role'})
        surrounding_context = {k:v for k,v in (current or {}).items()
                               if k in background_fields and k not in focused_current}
        context_fields = set(field_schema['properties']) | {'sections','notes'}
        async def read_context(fields):
            unknown = set(fields)-context_fields
            if unknown:
                raise ValueError('unknown editable fields: '+', '.join(sorted(unknown)))
            return stage_evidence({'current':{k:(current or {}).get(k) for k in fields},
                                   'change_schema':{k:repair.submit_tool['input_schema']['properties']['changes']['properties'][k]
                                                    for k in fields if k in repair.field_names}})
        context_tool = {'name':'commulingo_pipeline_context',
            'description':'Read current values and exact schema for additional fields only when needed for the commissioned correction.',
            'input_schema':{'type':'object','additionalProperties':False,
                'properties':{'fields':{'type':'array','minItems':1,'maxItems':len(context_fields),
                    'uniqueItems':True,
                    'items':{'type':'string','enum':sorted(context_fields)}}},'required':['fields']}}
        initial_status = status()
        # The commission is already present in issues; responses keep their own scope.
        initial_status.pop('scope')
        prompt = ('Complete the commissioned edit using the task data below. '
                  'Use commulingo_pipeline_context for additional current values. Editable changes are defined by the tools.\n'
                  + ('For a person section, submit changes.heading and changes.body. The server generates '
                     'the section topic slug; do not supply it. Write one distinct documented phase or theme '
                     'that the current sections do not cover. Give it a specific bilingual heading and a '
                     'substantive bilingual body with original evidence. If sources do not support a useful '
                     'section, submit a reasoned no-edit decision; length and section count are not targets.\n'
                     if section else '')
                  + stage_evidence({'job':{k:job[k] for k in ('id','kind','action','target')},
                      'current':focused_current,'issues':issues,
                      'surrounding_context':surrounding_context,'work_status':initial_status,
                      'prose_budgets':prose_budgets(focused_contract['properties']['fields']),
                      'original_proposal':(job.get('payload') or {}).get('original_proposal'),
                      'source_cache':session.context(),'saved_draft':repair.view(),
                      'previous_patch':previous_patch if not repair.draft else None,'review_feedback':previous_review}))
        spec = replace(COMMULINGO_CURATOR, prompt_ir=SystemPrompt(identity=EDITOR_POLICY+INSTRUCTIONS))
        await model_call(spec=spec,prompt=prompt,tool=repair.submit_tool,handler=finish,reads=READS,
            usage=usage,budget=budget,read_wrap=reads.wrap,max_rounds=12,
            local_tools=[(repair.update_tool,edit,True),(repair.no_edit_tool,no_edit,True),session.cached_tool(on_read=save_checkpoint),(context_tool,read_context,False),
                         reads.tool(field_schema['properties'],usage,on_reopen=save_checkpoint)],
            scope_id=f'commulingo_pipeline:{job["id"]}:editor',job=job)
        if box.get('rebase'):
            return Result(box,'research')
        if box.get('hold_reason'):
            return Result(box,'complete','escalated')
        return Result(box, 'review' if box.get('draft') else 'judge')
