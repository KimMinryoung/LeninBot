"""One author session reads original pages, drafts, and repairs a minimal patch."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json

from llm.prompt_renderer import SystemPrompt
from . import service
from .bundles import work_topics
from .draft_repair import RepairProtocolError
from .author_draft import AuthorDraft, structured_args
from .editor_checkpoint import restore_draft
from .diagnostics import prose_errors
from .engine import Result
from .evidence import compile_evidence, resolve_passages
from .issues import commission, FACTS
from .patches import canonical, changes, patch_hash, schema_for
from .source_session import Sources
from .editor_context import prose_budgets, work_status
from .decisions import ClassificationUnavailable, Decisions

EDITOR_MAX_ROUNDS = 24

INSTRUCTIONS = """Edit only the commissioned issues with the smallest supported bilingual patch.
Use the input as follows:
- issues: scope and completion criteria. Address required review corrections; optional suggestions are not obligations.
- current, surrounding_context: existing content to preserve. Read notes and sections to avoid duplication.
- tool schemas, prose_budgets: typed changes and output limits, never length targets.
- source_cache: available originals. Cite only displayed P-labels, using the shortest sufficient passages per changed factual field.
- work_status: saved draft, submission tool and next action. Follow the latest tool response.
- saved_draft: retained work, not approved content. Send only changed fields or evidence; omitted values remain saved.
Keep P-labels only in evidence, never in public prose. Existing dictionary text is context, not independent source verification.
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
        from .stages import (current_artifacts, latest, model_call, stage_evidence,
                             write_request, prose_problem, is_probe, StageContinues,
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
            from commulingo.classify import load_catalogs
            catalogs = await asyncio.to_thread(load_catalogs)
        decisions = Decisions(job,current,catalogs,usage,deepcopy(checkpoint.get('classification_cache',{})))
        field_schema = decisions.author_schema(schema_for(job, current, catalogs))
        field_schema['properties']['notes'] = {'type':'string','maxLength':4000,
            'description':'Private working notes; moved out of published fields.'}
        saved_draft = restore_draft(checkpoint.get('draft'), field_schema, section=section)
        if saved_draft and structured_args(saved_draft['args']):
            decisions.strip_assigned(saved_draft['args'].get('fields', {}))
        previous_patch = latest(artifacts,'draft') or {}
        needed = {i['field'] for i in issues}
        saved_fields = (saved_draft or {}).get('args', {}).get('fields', {})
        saved_fields = saved_fields if isinstance(saved_fields, dict) else {}
        needed.update(saved_fields)
        needed.update(previous_patch.get('fields',{}))
        needed.update(field_schema.get('required',[]))
        if 'activities' in needed:
            needed.update({'bio', 'career'})
        if '*' in needed:
            needed = set(field_schema['properties']) - {'aliasEdits', 'careerEdits', 'sceneEdits'}
            needed.update(saved_fields)
        focused_fields = deepcopy(field_schema)
        focused_fields['properties'] = {
            k:v for k,v in field_schema['properties'].items() if k in needed and k != 'notes'}
        factual = {'body'} if section else FACTS[job['kind']]
        author = AuthorDraft(field_schema, issues, editable_fields=focused_fields,
                             factual_fields=factual, draft=saved_draft)
        failures = dict(checkpoint.get('failures') or {})
        section_slug_cache = dict(checkpoint.get('section_slug_cache') or {})
        box = {}
        error_kind = checkpoint.get('error_kind', '')
        last_error = author.author_error(checkpoint.get('error', '') or '')
        def pending_evidence(args):
            fields = deepcopy(args.get('fields') or {})
            if job['kind'] == 'term':
                drop_unchanged_term_facts(fields, current or {}, job['action'])
            return (set(fields) & factual) - {c['field'] for c in args.get('claims') or []}

        def missing_parts(args):
            return author.missing(args, factual_fields=pending_evidence(args))

        def status():
            state = work_status(issues, author.draft, error=last_error, error_kind=error_kind)
            missing = missing_parts(author.draft['args']) if author.draft and structured_args(author.draft['args']) else []
            if missing and not last_error:
                state.update(missing_before_validation=missing, next_tool=author.submit_tool['name'],
                             next_action='Send the remaining parts; the draft is validated once they are present: '
                                         + ', '.join(missing) + '.')
            if author.draft and not structured_args(author.draft['args']):
                state.update(draft_saved=False,
                             next_action='Replace the malformed legacy draft with a new submission, or use the no-edit tool. History is retained.')
            return state
        def feedback(text):
            return text + '\n' + json.dumps({'work_status': status()}, ensure_ascii=False)

        async def save_checkpoint(error=None):
            if author.draft or session.passages.shown:
                await asyncio.to_thread(self.store.save_editor_checkpoint, job, {
                    'baseline':baseline, 'draft':author.draft, 'passages':session.passages.shown,
                    'source_requests':session.requests, 'failures':failures,
                    'error':last_error if error is None else author.author_error(error),
                    'error_kind':error_kind,
                    'classification_cache':decisions.cache,
                    'section_slug_cache':section_slug_cache})

        async def finish(value):
            nonlocal error_kind, last_error
            error_kind = 'schema'
            try:
                merged = author.submission(value)
                missing = missing_parts(merged)
                if missing:
                    # A split submission: keep the part and ask for the rest
                    # without counting a rejection or ending the stage.
                    author.save(merged)
                    error_kind, last_error = '', ''
                    await save_checkpoint()
                    raise StageContinues(feedback(
                        'Saved to the draft. Still needed before validation: ' + ', '.join(missing) + '.'))
                value = author.prepare(merged)
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
                problem = prose_errors(fields, passage_labels=session.passages.shown)
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
                missing = (set(fields) & factual) - {c['field'] for c in claims}
                if missing:
                    if error_kind != 'passages':
                        error_kind = 'missing_evidence'
                    problems.append('evidence required for ' + ', '.join(sorted(missing)) + '; repair those claims in this session')
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
                    error_kind = 'citation'
                    raise
                # Section dates are authored as startYear/startMonth but stored
                # as sortOrder. Keep their cited source attached to the stored
                # field, which is the only one the editorial RPC accepts.
                if section:
                    claims = [{**claim, 'field': 'sortOrder'}
                              if claim['field'] in {'startYear', 'startMonth'} else claim
                              for claim in claims]
                evidence = compile_evidence(claims, session.sources,
                                            set(fields) | ({'sortOrder'} if section else set()))
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
                            from commulingo.section_slug import generate_section_slug
                            section_slug_cache[slug_key] = await asyncio.to_thread(
                                generate_section_slug, job['target'], fields.get('heading'), fields.get('body'),
                                (current or {}).get('sections', []), usage=usage)
                            await save_checkpoint()
                        fields['slug'] = section_slug_cache[slug_key]
                    from commulingo.section_slug import section_sort_order
                    fields['sortOrder'] = section_sort_order(
                        fields.pop('startYear', None), fields.pop('startMonth', None),
                        (current or {}).get('sections', []))
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
            except ClassificationUnavailable as exc:
                error_kind, last_error = 'classification_unavailable', str(exc)
                await save_checkpoint()
                box['dependency_error'] = str(exc)
                # Complete the tool loop before raising to Engine. A tool-level
                # exception is otherwise fed back to the author as a repair.
                return 'Stopped: classification unavailable; draft saved for a later retry.'
            except ValueError as exc:
                last_error = author.author_error(str(exc))
                usage.tracker['preflight_failures'] = usage.tracker.get('preflight_failures',0)+1
                if 'revision_conflict' in str(exc):
                    box.update(rebase=True, error=str(exc))
                    return 'Revision changed; restart from current record with cached sources.'
                if isinstance(exc, RepairProtocolError):
                    usage.tracker['repair_protocol_errors'] = usage.tracker.get('repair_protocol_errors',0)+1
                    await save_checkpoint(str(exc))
                    raise RepairProtocolError(feedback(author.feedback(str(exc)))) from exc
                fingerprint = hashlib.sha256(canonical({'draft':author.draft,'error':str(exc)}).encode()).hexdigest()
                failures[fingerprint] = failures.get(fingerprint,0)+1
                await save_checkpoint(str(exc))
                if failures[fingerprint]>=2:
                    box.update(hold_reason='same rejected patch and error repeated without progress', error=str(exc))
                    return 'Held: identical failed patch repeated; retained for diagnosis.'
                raise ValueError(feedback(author.feedback(str(exc)))) from exc

        async def no_edit(**value):
            try:
                author.validate_call(value, author.no_edit_tool)
                if is_probe(value['reason']) or any(is_probe(item['reason']) for item in value['issues'].values()):
                    raise ValueError('Explain the no-edit decision substantively for every commissioned issue.')
            except ValueError as exc:
                raise ValueError(str(exc) + '; retry commulingo_pipeline_no_edit with the corrected decision.') from exc
            inspected = sorted({source['url'] for source in session.sources.values()
                                if source.get('body') and source['expires_at'] > datetime.now(timezone.utc)})
            # A draft written from sources this session read cannot then report
            # that no original was accessible. Job 63829 (Lafayette, 2026-09-26)
            # did exactly that after four prose values ran a few characters over
            # their limits: the false status deferred a sourced entry 90 days.
            if value['status']=='sources_unavailable' and inspected and author.draft:
                raise ValueError('sources_unavailable contradicts the sources read in this session ('
                                 + ', '.join(inspected[:3]) + ') and the draft saved from them. '
                                 'Fix the values the last validation rejected and resubmit the draft.')
            # complete means "already satisfied", which a create job whose target
            # does not exist can never be. Job 63954 (Nathalie Le Mel, 2026-09-30)
            # reported its unvalidated draft as the registered entry and the job
            # closed with nothing published.
            if value['status']=='complete' and job['action']=='create' and not current:
                raise ValueError('complete contradicts a create job: the entry does not exist and nothing is registered'
                                 + (f' (last validation error: {last_error})' if last_error else '')
                                 + '. Fix the rejected values and resubmit the draft, or use not_applicable/sources_unavailable.')
            # Preserve the last draft and its evidence for audit/recovery. This
            # decision is a distinct artifact, never a mutation of draft status.
            await save_checkpoint()
            box.update(editor_version=2, research={**research, 'status':value['status'],
                'reason':value['reason'], 'inspected_sources':inspected,
                'issue_results':[{'id':key, **item} for key, item in value['issues'].items()],
                **({'notes':value['notes']} if value.get('notes') else {})})
            return 'OK: no-edit judgment recorded; saved draft retained in history'

        focused_current = {k:v for k,v in (current or {}).items()
                           if k in needed | {'id','revision','name','term','evidence','notes','sections'}}
        background_fields = ({'definition','original','aliases','period','startYear','endYear'}
                             if job['kind']=='term' else
                             {'givenName','familyName','cyrillic','years','epithet','bio','activities'})
        surrounding_context = {k:v for k,v in (current or {}).items()
                               if k in background_fields and k not in focused_current}
        context_fields = set(field_schema['properties']) | {'sections','notes'}
        async def read_context(fields):
            unknown = set(fields)-context_fields
            if unknown:
                raise ValueError('unknown editable fields: '+', '.join(sorted(unknown)))
            return stage_evidence({'current':{k:(current or {}).get(k) for k in fields},
                                   'change_schema':{k:author.submit_tool['input_schema']['properties']['fields']['properties'][k]
                                                    for k in fields if k in author.field_names}})
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
                  + ('For a person section, use commulingo_pipeline_submit_draft '
                     'with fields.heading, fields.startYear, fields.body and evidence.body, plus issues and reason. '
                     'One call is sufficient; split only if the text is long, preserving already saved parts. '
                     'startYear is always a year: for a theme, the year it begins; startMonth only when known. '
                     'Cite a passage for a factual year. '
                     'The server generates the section topic slug; do not supply it. Write one distinct documented phase or theme '
                     'that the current sections do not cover. Give it a specific bilingual heading and a '
                     'substantive bilingual body with original evidence. If sources do not support a useful '
                     'section, submit a reasoned no-edit decision; length and section count are not targets.\n'
                     if section else '')
                  + stage_evidence({'job':{k:job[k] for k in ('id','kind','action','target')},
                      'current':focused_current,'issues':issues,
                      'surrounding_context':surrounding_context,'work_status':initial_status,
                      'prose_budgets':prose_budgets(focused_fields),
                      'original_proposal':(job.get('payload') or {}).get('original_proposal'),
                      'source_cache':session.context(),'saved_draft':author.view(),
                      'previous_patch':previous_patch if not author.draft else None,'review_feedback':previous_review}))
        # 12 rounds ended 27 of 32 failed attempts and capped 80 of 257
        # successes (2026-09-22..25) while a run cost $0.0135 on average
        # against the $0.20 stage budget; split submissions add rounds too.
        spec = replace(COMMULINGO_CURATOR, prompt_ir=SystemPrompt(identity=EDITOR_POLICY+INSTRUCTIONS),
                       max_rounds=EDITOR_MAX_ROUNDS)
        await model_call(spec=spec,prompt=prompt,tool=author.submit_tool,handler=finish,reads=READS,
            usage=usage,budget=budget,read_wrap=session.wrap,max_rounds=EDITOR_MAX_ROUNDS,
            local_tools=[(author.no_edit_tool,no_edit,True),session.cached_tool(on_read=save_checkpoint),(context_tool,read_context,False)],
            scope_id=f'commulingo_pipeline:{job["id"]}:editor',job=job)
        if box.get('dependency_error'):
            raise ClassificationUnavailable(box['dependency_error'])
        if box.get('rebase'):
            return Result(box,'research')
        if box.get('hold_reason'):
            return Result(box,'complete','escalated')
        return Result(box, 'review' if box.get('draft') else 'judge')
