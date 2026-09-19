"""PostgreSQL queue with fenced leases and conservative cost reservations."""
from contextlib import contextmanager
from decimal import Decimal
import uuid
import hashlib
import json
from .bundles import advance, bundle_candidates, gap_ids

from psycopg2.extras import Json, RealDictCursor


class LostLease(RuntimeError):
    pass


class BudgetUnavailable(RuntimeError):
    pass


# Importance tier (operator decision 2026-09-17): people linked to at least this
# many history events get the deep section budget and the short re-enrichment grace.
IMPORTANT_EVENTS = 6
GRACE_DAYS_IMPORTANT, GRACE_DAYS_OTHER = 14, 90
# Linked events -> maximum number of detail sections commissioned.
SECTION_CAP_SQL = 'CASE WHEN {events}>=6 THEN 12 WHEN {events}>=3 THEN 5 WHEN {events}>=1 THEN 3 ELSE 2 END'
PERSON_EVENTS_SQL = '(SELECT count(DISTINCT e.event_id) FROM commulingo_history_event_people e WHERE e.person_id={person})'
# True while a person's last pipeline-applied edit is younger than their grace window.
PERSON_IN_GRACE_SQL = ('EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a JOIN commulingo_pipeline_jobs g ON g.id=a.job_id '
    "WHERE g.kind='person' AND g.target={person} AND a.stage='submit' AND a.value->>'status'='approved' "
    'AND a.created_at > now() - (CASE WHEN ' + PERSON_EVENTS_SQL + ' >= %(important)s '
    "THEN %(grace_important)s ELSE %(grace_other)s END) * interval '1 day')")
GRACE_PARAMS = {'important':IMPORTANT_EVENTS,'grace_important':GRACE_DAYS_IMPORTANT,'grace_other':GRACE_DAYS_OTHER}
# Terms (operator decisions 2026-09-17): a pipeline-applied edit earns a fixed
# grace; a body this long is left alone rather than rewritten whole (the term
# body has no schema ceiling, so this is an absolute size); a term that
# coincides with a history event by id, title or alias belongs to the events
# lane unless allowlisted in config.
TERM_GRACE_DAYS = 90
TERM_BODY_ENOUGH_KO, TERM_BODY_ENOUGH_EN = 2000, 4500
TERM_IN_GRACE_SQL = ('EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a JOIN commulingo_pipeline_jobs g ON g.id=a.job_id '
    "WHERE g.kind='term' AND g.target={term} AND a.stage='submit' AND a.value->>'status'='approved' "
    "AND a.created_at > now() - %(term_grace)s * interval '1 day')")
TERM_BODY_ENOUGH_SQL = '(length({t}.body_ko) >= %(body_ko)s OR length({t}.body_en) >= %(body_en)s)'
# A gap or discovery label that names an existing history event is not a new
# glossary entry (operator decision 2026-09-17); ids in the allowlist still are.
EVENT_TITLE_MATCH_SQL = ('EXISTS (SELECT 1 FROM commulingo_history_events ev WHERE lower(ev.title_ko)=lower({label_ko}) '
    "OR ({label_en}<>'' AND lower(ev.title_en)=lower({label_en})))")


def term_params(exclude):
    return {'term_grace':TERM_GRACE_DAYS,'body_ko':TERM_BODY_ENOUGH_KO,'body_en':TERM_BODY_ENOUGH_EN,
            'term_exclude':list(exclude or [])}


def term_priority(body_empty, mentions):
    """Body-less terms first, then by linked public reports; always in the non-urgent range."""
    return (21 if body_empty else 51) + max(0, 29 - min(int(mentions or 0), 29))



class Store:
    def __init__(self, connect=None):
        if connect is None:
            from db import get_conn
            connect = get_conn
        self.connect = connect

    @contextmanager
    def transaction(self):
        with self.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur

    def enqueue(self, *, kind, action, target, topic, baseline='', reason,
                priority=50, payload=None, stage='research'):
        with self.transaction() as cur:
            cur.execute('''INSERT INTO commulingo_pipeline_jobs
                (kind,action,target,topic,baseline,reason,priority,payload,stage)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (kind,target,topic) WHERE status IN
                ('ready','running','deferred','escalated') DO NOTHING RETURNING id''',
                (kind, action, target, topic, baseline, reason, priority, Json(payload or {}),stage))
            row = cur.fetchone()
            return row['id'] if row else None

    def reprioritize_people(self):
        """Refresh untouched person enrichment priorities from current event links."""
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs j SET priority=x.priority, updated_at=now()
                FROM (SELECT p.id, 100 - LEAST((SELECT count(DISTINCT e.event_id)
                        FROM commulingo_history_event_people e WHERE e.person_id=p.id),79) AS priority
                      FROM commulingo_people p) x
                WHERE j.kind='person' AND j.action='update' AND j.topic='enrichment'
                  AND j.status='ready' AND j.priority>=20
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=j.id)
                  AND j.target=x.id AND j.priority!=x.priority''')
            return cur.rowcount

    def retire_people_in_grace(self):
        """Untouched enrichment bundles for recently edited people wait out their grace window."""
        with self.transaction() as cur:
            cur.execute("""UPDATE commulingo_pipeline_jobs j SET status='cancelled', updated_at=now(),
                last_error='re-enrichment grace after an applied edit'
                WHERE j.kind='person' AND j.action='update' AND j.topic='enrichment'
                  AND j.status='ready' AND j.stage='research' AND j.priority>=20
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=j.id)
                  AND """ + PERSON_IN_GRACE_SQL.format(person='j.target'), GRACE_PARAMS)
            return cur.rowcount

    def reprioritize_terms(self, mentions):
        """Refresh untouched term enrichment priorities from body state and report mentions."""
        ids = list(mentions or {})
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs j SET priority=x.priority, updated_at=now()
                FROM (SELECT t.id, (CASE WHEN t.body_ko='' OR t.body_en='' THEN 21 ELSE 51 END)
                             + GREATEST(0, 29 - LEAST(COALESCE(m.mentions,0), 29)) AS priority
                      FROM commulingo_terms t
                      LEFT JOIN unnest(%s::text[], %s::int[]) AS m(id, mentions) ON m.id=t.id) x
                WHERE j.kind='term' AND j.action='update' AND j.topic='enrichment'
                  AND j.status='ready' AND j.stage='research' AND j.priority>=20
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=j.id)
                  AND j.target=x.id AND j.priority!=x.priority''',
                (ids, [int(mentions[i]) for i in ids]))
            return cur.rowcount

    def retire_unqualified_terms(self, exclude):
        """Untouched term bundles that no longer qualify wait or leave the queue."""
        params = term_params(exclude)
        untouched = '''j.kind='term' AND j.action='update' AND j.topic='enrichment'
                  AND j.status='ready' AND j.stage='research' AND j.priority>=20
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=j.id)'''
        retired = {}
        with self.transaction() as cur:
            for reason, predicate in (
                    ('re-enrichment grace after an applied edit', TERM_IN_GRACE_SQL.format(term='j.target')),
                    ('body already substantial', 'EXISTS (SELECT 1 FROM commulingo_terms t WHERE t.id=j.target AND '
                        + TERM_BODY_ENOUGH_SQL.format(t='t') + ')'),
                    ('excluded from enrichment by operator', 'j.target = ANY(%(term_exclude)s::text[])')):
                cur.execute(f'''UPDATE commulingo_pipeline_jobs j SET status='cancelled', updated_at=now(), last_error=%(reason)s
                    WHERE {untouched} AND {predicate}''', {**params, 'reason':reason})
                retired[reason] = cur.rowcount
        return retired

    def restore_terms(self, targets, reasons):
        """Reopen untouched bundles cancelled for a reason that no longer applies."""
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs j SET status='ready', last_error='', updated_at=now()
                WHERE j.kind='term' AND j.action='update' AND j.topic='enrichment' AND j.status='cancelled'
                  AND j.last_error = ANY(%s::text[]) AND j.target = ANY(%s::text[])
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_jobs o WHERE o.kind=j.kind AND o.target=j.target
                      AND o.id<>j.id AND o.status IN ('ready','running','deferred','escalated'))''',
                (list(reasons), list(targets)))
            return cur.rowcount

    def cancel_discovery(self):
        """Discovery is switched off: retire queued material jobs without deleting history."""
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs SET status='cancelled', updated_at=now(),
                last_error='discovery disabled in config/commulingo_pipeline.json'
                WHERE stage='discover' AND status IN ('ready','deferred','escalated')
                  AND payload->>'material_id' NOT LIKE 'gap:%%' ''')
            return cur.rowcount

    def enqueue_review_repair(self, proposal, decision):
        """One durable correction per original proposal, even after completion/replay."""
        if proposal['action'] not in {'create','update'}:
            return None
        section = proposal['target_type']=='person_section'
        kind = 'person' if section else proposal['target_type']
        topic = f"review-repair:{proposal['id']}"
        payload = {'replaces_suggestion_id':proposal['id'], 'original_proposal':proposal,
                   'review_feedback':decision,'review_revisions':1,
                   'topics':['sections'] if section else ['review_correction']}
        # Store only JSON editorial data; DB timestamps are not part of the commission.
        payload['original_proposal'] = {k:proposal[k] for k in
            ('id','target_type','target_id','action','patch_json','source_refs')}
        with self.transaction() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",(topic,))
            cur.execute('SELECT id FROM commulingo_pipeline_jobs WHERE topic=%s ORDER BY id LIMIT 1',(topic,))
            existing = cur.fetchone()
            if existing:
                return existing['id']
            cur.execute('''INSERT INTO commulingo_pipeline_jobs
                (kind,action,target,topic,reason,priority,payload)
                VALUES (%s,%s,%s,%s,%s,10,%s) RETURNING id''',
                (kind,'update' if section else proposal['action'],proposal['target_id'],topic,
                 'Correct independently reviewed proposal',Json(payload)))
            return cur.fetchone()['id']

    def claim(self, *, lease_seconds=120, group=None, job_id=None, stages=None):
        token = str(uuid.uuid4())
        with self.transaction() as cur:
            cur.execute('SELECT cursor FROM commulingo_pipeline_scheduler WHERE id=1 FOR UPDATE')
            cursor = cur.fetchone()['cursor']
            cur.execute('''WITH candidate AS (
                SELECT id FROM commulingo_pipeline_jobs
                WHERE ((status IN ('ready','deferred') AND available_at <= now())
                    OR (status='running' AND lease_until < now()))
                  AND (%s::text IS NULL OR kind || ':' || action = %s)
                  AND (%s::bigint IS NULL OR id = %s)
                  AND (%s::text[] IS NULL OR stage=ANY(%s::text[]))
                ORDER BY CASE WHEN stage IN ('review','submit') THEN 0 WHEN priority<20 THEN 1 ELSE 2 END,
                         mod(CASE kind || ':' || action
                            WHEN 'person:create' THEN 0 WHEN 'person:update' THEN 1
                            WHEN 'term:create' THEN 2 ELSE 3 END - %s + 4,4),
                         priority, created_at, id
                FOR UPDATE SKIP LOCKED LIMIT 1)
                UPDATE commulingo_pipeline_jobs j SET status='running',
                    lease_token=%s, lease_until=now()+%s*interval '1 second',
                    attempts=attempts+1, updated_at=now()
                FROM candidate c WHERE j.id=c.id RETURNING j.*''',
                (group, group, job_id, job_id, stages, stages, cursor, token, lease_seconds))
            job = cur.fetchone()
            if job:
                groups = ['person:create','person:update','term:create','term:update']
                cur.execute('UPDATE commulingo_pipeline_scheduler SET cursor=%s WHERE id=1',
                            ((groups.index(job['kind']+':'+job['action'])+1)%4,))
            return job

    def consolidate(self, *, apply=False):
        """Merge only untouched update jobs; preserve every old row for audit."""
        with self.transaction() as cur:
            # Serialize against claim/enqueue, including manual workers.
            if apply:
                cur.execute("SET LOCAL lock_timeout='5s'")
                cur.execute('LOCK TABLE commulingo_pipeline_jobs IN SHARE ROW EXCLUSIVE MODE')
            cur.execute('''SELECT j.* FROM commulingo_pipeline_jobs j
                WHERE j.action='update' AND j.status='ready' AND j.stage='research'
                  AND j.attempts=0 AND j.topic!='enrichment'
                  AND NOT (j.payload ? 'replaces_suggestion_id')
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=j.id)
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_job_sources s WHERE s.job_id=j.id)
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_budget b WHERE b.job_id=j.id)
                  AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_jobs other
                    WHERE other.kind=j.kind AND other.target=j.target AND other.id!=j.id
                      AND other.status IN ('ready','running','deferred','escalated')
                      AND (other.action!='update' OR other.topic='enrichment'
                        OR other.payload ? 'replaces_suggestion_id'
                        OR other.status!='ready' OR other.stage!='research' OR other.attempts!=0
                        OR EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=other.id)
                        OR EXISTS (SELECT 1 FROM commulingo_pipeline_job_sources s WHERE s.job_id=other.id)
                        OR EXISTS (SELECT 1 FROM commulingo_pipeline_budget b WHERE b.job_id=other.id)))
                ORDER BY j.priority,j.id''')
            rows = [dict(r) for r in cur.fetchall()]
            groups = {}
            for row in rows:
                groups.setdefault((row['kind'],row['target']), []).append(row)
            bundles = bundle_candidates(rows)
            if apply:
                for bundle in bundles:
                    members = groups[bundle['kind'],bundle['target']]
                    parent = members[0]['id']
                    children = [r['id'] for r in members[1:]]
                    payload = {**bundle['payload'], 'bundled_job_ids':[r['id'] for r in members]}
                    cur.execute('''UPDATE commulingo_pipeline_jobs SET topic='enrichment',payload=%s,
                        baseline=%s,reason=%s,updated_at=now() WHERE id=%s''',
                        (Json(payload),bundle['baseline'],bundle['reason'],parent))
                    if children:
                        cur.execute('''UPDATE commulingo_pipeline_jobs SET status='cancelled',
                            payload=payload || %s::jsonb,last_error='Consolidated into target bundle',
                            updated_at=now() WHERE id=ANY(%s)''',(Json({'bundled_into':parent}),children))
            return {'applied':apply, 'jobs_before':len(rows), 'bundles':len(bundles),
                    'merged_jobs':len(rows)-len(bundles)}

    def release_publication_waits(self):
        """Live mode releases only the waits caused by the former canary cap."""
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs SET status='ready',available_at=now(),
                last_error='',updated_at=now() WHERE status='deferred' AND stage='submit'
                AND last_error='canary publication slots exhausted' RETURNING id''')
            return len(cur.fetchall())

    def start_attempt(self, job):
        attempt = str(uuid.uuid4())
        with self.transaction() as cur:
            cur.execute("INSERT INTO commulingo_pipeline_attempts(id,job_id,stage) VALUES (%s,%s,%s)",
                        (attempt,job['id'],job['stage']))
        return attempt

    def link_attempt_budget(self, attempt, reservation):
        with self.transaction() as cur:
            cur.execute('UPDATE commulingo_pipeline_attempts SET budget_id=%s WHERE id=%s',
                        (reservation,attempt))

    def finish_attempt(self, attempt, outcome, next_stage, error, duration, metrics):
        # Do not duplicate tool transcripts/source text or ledger cost in metrics.
        terminal_calls = sum(isinstance(line,str) and '] commulingo_pipeline_result(' in line
                             for line in metrics.get('tool_work_details',[]))
        metrics = {k:v for k,v in metrics.items() if k in {
            'rounds_used','input_tokens','output_tokens','model_calls','pipeline_cache_hits',
            'preflight_checks','preflight_failures','preflight_passed','rejections','provider_fallback',
            'targeted_research'}}
        metrics['terminal_calls'] = terminal_calls
        if 'rejections' in metrics:
            metrics['rejections'] = metrics['rejections'][-12:]
        with self.transaction() as cur:
            cur.execute("""UPDATE commulingo_pipeline_attempts SET finished_at=now(),
                duration_seconds=%s,outcome=%s,next_stage=%s,error=%s,metrics=%s WHERE id=%s""",
                (duration,outcome,next_stage,str(error)[:2000],Json(metrics),attempt))

    def release_budget_waits(self, *, cap, amount, review_fraction):
        """Reconsider only known budget waits, never error or evidence holds."""
        cap, amount, fraction = map(lambda x: Decimal(str(x)), (cap,amount,review_fraction))
        with self.transaction() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('commulingo-pipeline-budget'))")
            cur.execute("""WITH spent AS (
                SELECT coalesce(sum(coalesce(actual,reserved)),0) AS total,
                    coalesce(sum(coalesce(actual,reserved)) FILTER (WHERE lane!='review'),0) AS author
                FROM commulingo_pipeline_budget WHERE day=(now() AT TIME ZONE 'UTC')::date)
                UPDATE commulingo_pipeline_jobs SET status='ready',available_at=now(),last_error='',updated_at=now()
                FROM spent WHERE status='deferred' AND last_error='daily budget reserved or spent'
                AND (stage IN ('validate','judge','submit') OR
                    (spent.total+%s<=%s AND (stage='review' OR spent.author+%s<=%s*(1-%s))))
                RETURNING id""",(amount,cap,amount,cap,fraction))
            return len(cur.fetchall())

    def heartbeat(self, job, *, lease_seconds=120):
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs
                SET lease_until=now()+%s*interval '1 second'
                WHERE id=%s AND lease_token=%s AND status='running'
                  AND lease_until>now() RETURNING id''',
                (lease_seconds, job['id'], job['lease_token']))
            if not cur.fetchone():
                raise LostLease(str(job['id']))

    def finish_stage(self, job, value, *, next_stage, status='ready', usage=None, delay_seconds=0):
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs SET stage=%s,status=%s,
                lease_token=NULL,lease_until=NULL,last_error='',updated_at=now(),attempts=0,
                available_at=now()+%s*interval '1 second'
                WHERE id=%s AND lease_token=%s AND status='running'
                  AND lease_until>now() RETURNING id''',
                (next_stage, status, delay_seconds,job['id'], job['lease_token']))
            if not cur.fetchone():
                raise LostLease(str(job['id']))
            metrics = {k:v for k,v in (usage or {}).items() if k in
                       {'total_cost','rounds_used','input_tokens','output_tokens','pipeline_cache_hits','provider_fallback'}}
            if (usage or {}).get('provider_fallback'):
                # Later stages see the same sources; skip the provider that refused them.
                cur.execute('''UPDATE commulingo_pipeline_jobs
                    SET payload=payload || %s::jsonb WHERE id=%s''',
                    (Json({'provider_fallback':usage['provider_fallback']}),job['id']))
            if value.get('remaining_topics'):
                cur.execute('''UPDATE commulingo_pipeline_jobs
                    SET payload=payload || %s::jsonb WHERE id=%s''',
                    (Json({'remaining_topics':value['remaining_topics']}),job['id']))
            cur.execute('''INSERT INTO commulingo_pipeline_artifacts(job_id,stage,value,metrics)
                VALUES (%s,%s,%s,%s)''', (job['id'], job['stage'], Json(value),Json(metrics)))
            if job['stage']=='discover':
                for candidate in value.get('candidates',[]):
                    payload = {'candidate':candidate,'material_id':job['payload']['material_id']}
                    if payload['material_id'].startswith('gap:'):
                        payload['gap_id'] = int(payload['material_id'].split(':')[1])
                    cur.execute('''INSERT INTO commulingo_pipeline_mentions(kind,target,material_id,mention)
                        VALUES (%s,%s,%s,%s) ON CONFLICT(kind,target,material_id) DO UPDATE SET mention=EXCLUDED.mention''',
                        (candidate['kind'],candidate['target'],job['payload']['material_id'],candidate['mention']))
                    cur.execute('SELECT count(*) AS n FROM commulingo_pipeline_mentions WHERE kind=%s AND target=%s',
                                (candidate['kind'],candidate['target']))
                    priority = 50-min(10,cur.fetchone()['n'])
                    cur.execute('''INSERT INTO commulingo_pipeline_jobs
                        (kind,action,target,topic,reason,priority,payload)
                        VALUES (%s,'create',%s,'basics',%s,%s,%s)
                        ON CONFLICT (kind,target,topic) WHERE status IN
                        ('ready','running','deferred','escalated') DO UPDATE
                        SET priority=LEAST(commulingo_pipeline_jobs.priority,EXCLUDED.priority)''',
                        (candidate['kind'],candidate['target'],candidate['reason'],priority,
                         Json(payload)))
                cur.execute('''INSERT INTO commulingo_pipeline_materials(material_id,content_hash)
                    VALUES (%s,%s) ON CONFLICT(material_id) DO UPDATE
                    SET content_hash=EXCLUDED.content_hash,processed_at=now()''',
                    (job['payload']['material_id'],job['payload']['content_hash']))
                if job['payload']['material_id'].startswith('gap:') and not value.get('candidates'):
                    # A requested entry the model declined stays visible as skipped,
                    # never as a pending request nothing will pick up again.
                    cur.execute('''UPDATE commulingo_curation_gaps SET status='skipped',
                        resolution=%s,updated_at=now() WHERE id=%s AND status='pending' ''',
                        ('Pipeline discovery declined: ' + (value.get('skip_reason') or 'no reason recorded'),
                         int(job['payload']['material_id'].split(':')[1])))
            if (job['stage']=='research' and status=='complete' and job['action']=='create'
                    and value.get('reason')=='target already exists' and gap_ids(job['payload'])):
                cur.execute('''UPDATE commulingo_curation_gaps SET status='done',resolved_id=%s,
                    resolution='Entry already existed when the pipeline researched it',updated_at=now()
                    WHERE id=ANY(%s) AND status='pending' ''',(job['target'],gap_ids(job['payload'])))
            if job['stage']=='submit' and value.get('status')=='approved' and gap_ids(job['payload']):
                cur.execute('''UPDATE commulingo_curation_gaps SET status='done',resolved_id=%s,
                    resolution='Approved through durable pipeline',updated_at=now()
                    WHERE id=ANY(%s) AND status='pending' ''',(job['target'],gap_ids(job['payload'])))

    def defer(self, job, error, *, seconds=3600, escalate=False, failed=True):
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs SET status=%s,
                available_at=now()+%s*interval '1 second',last_error=%s,
                attempts=GREATEST(0,attempts-%s),
                lease_token=NULL,lease_until=NULL,updated_at=now()
                WHERE id=%s AND lease_token=%s AND status='running'
                  AND lease_until>now() RETURNING id''',
                ('escalated' if escalate else 'deferred', seconds, str(error)[:2000],0 if failed else 1,
                 job['id'], job['lease_token']))
            if not cur.fetchone():
                raise LostLease(str(job['id']))

    def detail(self, job_id):
        with self.transaction() as cur:
            cur.execute('SELECT * FROM commulingo_pipeline_jobs WHERE id=%s', (job_id,))
            job = cur.fetchone()
            if not job:
                return None
            cur.execute('''SELECT stage,value,metrics,created_at FROM commulingo_pipeline_artifacts
                WHERE job_id=%s ORDER BY id''', (job_id,))
            return {'job': job, 'artifacts': cur.fetchall()}

    def list_jobs(self, limit=50):
        with self.transaction() as cur:
            cur.execute('''SELECT * FROM commulingo_pipeline_jobs
                ORDER BY priority,created_at,id LIMIT %s''', (limit,))
            return cur.fetchall()

    def efficiency(self, since):
        from .efficiency import query
        with self.transaction() as cur:
            cur.execute(query('%s::timestamptz'),(since,))
            return next(iter(cur.fetchone().values()))

    def metrics(self):
        with self.transaction() as cur:
            cur.execute('''SELECT j.kind,j.action,j.status,count(*) AS jobs
                FROM commulingo_pipeline_jobs j GROUP BY j.kind,j.action,j.status ORDER BY 1,2,3''')
            jobs = cur.fetchall()
            cur.execute('''SELECT stage,count(*) AS artifacts,
                count(*) FILTER (WHERE value ? 'error') AS rejected,
                sum((metrics->>'total_cost')::numeric) AS cost_usd,
                sum((metrics->>'pipeline_cache_hits')::integer) AS cache_hits
                FROM commulingo_pipeline_artifacts GROUP BY stage ORDER BY stage''')
            stages = cur.fetchall()
            cur.execute('''SELECT count(*) AS cases,
                count(*) FILTER (WHERE NOT (value ? 'error')) AS first_passed
                FROM (SELECT DISTINCT ON (job_id) value FROM commulingo_pipeline_artifacts
                    WHERE stage='validate' ORDER BY job_id,id) first_validation''')
            validation = dict(cur.fetchone())
            validation['first_pass_rate'] = validation['first_passed']/validation['cases'] if validation['cases'] else None
            cur.execute('''SELECT count(DISTINCT a.job_id) AS applied FROM commulingo_pipeline_artifacts a
                WHERE (a.stage='submit' AND a.value->>'status'='approved') OR
                    (a.stage='review' AND EXISTS (SELECT 1 FROM commulingo_agent_suggestions s
                        WHERE s.id=(a.value->>'suggestionId')::bigint AND s.status='approved'))''')
            applied = cur.fetchone()['applied']
        return {'jobs':jobs,'stages':stages,'validation':validation,'applied':applied,'budget':self.costs()}

    def reserve(self, amount, *, lane, job_id=None, cap='3.39', review_fraction='0.30'):
        amount, cap = Decimal(str(amount)), Decimal(str(cap))
        fraction = Decimal(str(review_fraction))
        if not amount.is_finite() or amount <= 0 or not cap.is_finite() or cap <= 0:
            raise ValueError('positive finite reservation and cap required')
        if not fraction.is_finite() or not 0 <= fraction <= 1:
            raise ValueError('invalid review fraction')
        token = str(uuid.uuid4())
        with self.transaction() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('commulingo-pipeline-budget'))")
            cur.execute('''SELECT COALESCE(sum(COALESCE(actual,reserved)),0) AS total,
                COALESCE(sum(COALESCE(actual,reserved)) FILTER (WHERE lane!='review'),0) AS author
                FROM commulingo_pipeline_budget
                WHERE day=(now() AT TIME ZONE 'UTC')::date''')
            spent = cur.fetchone()
            if spent['total'] + amount > cap or (lane != 'review' and
                    spent['author'] + amount > cap * (1 - fraction)):
                raise BudgetUnavailable('daily budget reserved or spent')
            cur.execute('''INSERT INTO commulingo_pipeline_budget(id,lane,job_id,reserved)
                VALUES (%s,%s,%s,%s)''', (token, lane, job_id, amount))
        return token

    def settle(self, token, actual):
        actual = Decimal(str(actual))
        if not actual.is_finite() or actual < 0:
            raise ValueError('nonnegative finite cost required')
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_budget SET actual=%s,settled_at=now()
                WHERE id=%s AND actual IS NULL RETURNING id''', (actual, token))
            if not cur.fetchone():
                cur.execute('SELECT actual FROM commulingo_pipeline_budget WHERE id=%s', (token,))
                row = cur.fetchone()
                if not row or row['actual'] != actual:
                    raise ValueError('unknown reservation or conflicting settlement')

    def costs(self):
        with self.transaction() as cur:
            cur.execute('''SELECT day,lane,sum(actual) AS actual,
                sum(reserved) FILTER (WHERE actual IS NULL) AS outstanding,
                sum(GREATEST(COALESCE(actual,reserved)-reserved,0)) AS overrun
                FROM commulingo_pipeline_budget GROUP BY day,lane ORDER BY day DESC,lane''')
            return cur.fetchall()

    def save_source(self, source):
        with self.transaction() as cur:
            cur.execute('''INSERT INTO commulingo_pipeline_sources
                (id,url,content_hash,fetched_at,expires_at,body)
                VALUES (%(id)s,%(url)s,%(content_hash)s,%(fetched_at)s,%(expires_at)s,%(body)s)
                ON CONFLICT(id) DO UPDATE SET fetched_at=EXCLUDED.fetched_at,
                expires_at=EXCLUDED.expires_at,body=EXCLUDED.body''', source)

    def sources(self, ids):
        with self.transaction() as cur:
            cur.execute('SELECT * FROM commulingo_pipeline_sources WHERE id=ANY(%s)', (list(ids),))
            return {row['id']: row for row in cur.fetchall()}

    def expire_sources(self):
        with self.transaction() as cur:
            cur.execute('UPDATE commulingo_pipeline_sources SET body=NULL WHERE expires_at<now() AND body IS NOT NULL')
            return cur.rowcount

    def retry(self, job_id):
        with self.transaction() as cur:
            cur.execute('''UPDATE commulingo_pipeline_jobs SET status='ready',
                available_at=now(),attempts=0,last_error='',updated_at=now()
                WHERE id=%s AND status IN ('deferred','escalated') RETURNING id''', (job_id,))
            return bool(cur.fetchone())

    def link_source(self, job_id, source_id):
        with self.transaction() as cur:
            cur.execute('''INSERT INTO commulingo_pipeline_job_sources(job_id,source_id)
                VALUES (%s,%s) ON CONFLICT DO NOTHING''', (job_id, source_id))

    def job_sources(self, job_id):
        """The job's snapshots. An oversize body (a runaway merge, see
        evidence.MAX_SNAPSHOT_CHARS) comes back with body=NULL so it is never
        loaded into the process; job 2523 held 1.3 GB of such rows."""
        from .evidence import MAX_SNAPSHOT_CHARS
        with self.transaction() as cur:
            cur.execute('''SELECT s.id, s.url, s.content_hash, s.fetched_at, s.expires_at,
                    CASE WHEN length(s.body) > %s THEN NULL ELSE s.body END AS body
                FROM commulingo_pipeline_sources s
                JOIN commulingo_pipeline_job_sources j ON j.source_id=s.id WHERE j.job_id=%s''',
                (MAX_SNAPSHOT_CHARS, job_id))
            return {row['id']: row for row in cur.fetchall()}

    def cached_source(self, tool, args):
        key = hashlib.sha256(json.dumps(args,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self.transaction() as cur:
            cur.execute('''SELECT s.* FROM commulingo_pipeline_sources s
                JOIN commulingo_pipeline_fetch_cache c ON c.source_id=s.id
                WHERE c.tool=%s AND c.args_hash=%s AND s.expires_at>now() AND s.body IS NOT NULL''',(tool,key))
            return cur.fetchone()

    def cache_source(self, tool, args, source_id):
        key = hashlib.sha256(json.dumps(args,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self.transaction() as cur:
            cur.execute('''INSERT INTO commulingo_pipeline_fetch_cache(tool,args_hash,source_id)
                VALUES (%s,%s,%s) ON CONFLICT(tool,args_hash) DO UPDATE SET source_id=EXCLUDED.source_id''',
                (tool,key,source_id))

    def publication_slot(self, job, limit):
        with self.transaction() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('commulingo-pipeline-publication'))")
            cur.execute("SELECT day=(now() AT TIME ZONE 'UTC')::date AS current_day FROM commulingo_pipeline_publications WHERE job_id=%s",(job['id'],))
            slot = cur.fetchone()
            if slot and slot['current_day']:
                return
            cur.execute('''SELECT count(*) AS n FROM commulingo_pipeline_publications
                WHERE day=(now() AT TIME ZONE 'UTC')::date AND kind=%s AND action=%s''',
                (job['kind'],job['action']))
            if cur.fetchone()['n']>=limit:
                raise BudgetUnavailable('canary publication slots exhausted')
            cur.execute('''INSERT INTO commulingo_pipeline_publications(job_id,kind,action) VALUES (%s,%s,%s)
                ON CONFLICT(job_id) DO UPDATE SET day=EXCLUDED.day''',
                        (job['id'],job['kind'],job['action']))

    def reconcile_reviews(self):
        with self.transaction() as cur:
            cur.execute('''SELECT j.*,s.status AS review_status FROM commulingo_pipeline_jobs j
                JOIN LATERAL (SELECT id,value FROM commulingo_pipeline_artifacts
                    WHERE job_id=j.id AND stage='review' ORDER BY id DESC LIMIT 1) a ON true
                JOIN commulingo_agent_suggestions s ON s.id=(a.value->>'suggestionId')::bigint
                WHERE j.status='escalated' AND s.status IN ('approved','rejected')
                    AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts newer
                        WHERE newer.job_id=j.id AND newer.id>a.id AND newer.value ? 'remaining_topics')
                FOR UPDATE OF j''')
            rows = cur.fetchall()
            for row in rows:
                value = advance(row,{'status':row['review_status']}) if row['review_status']=='approved' else {}
                remaining = value.get('remaining_topics')
                cur.execute('''UPDATE commulingo_pipeline_jobs SET status=%s,stage=%s,
                    payload=payload || %s::jsonb,available_at=now(),updated_at=now() WHERE id=%s''',
                    ('ready' if remaining else 'complete','research' if remaining else 'complete',
                     Json({'remaining_topics':remaining} if remaining else {}),row['id']))
                if remaining:
                    cur.execute('''INSERT INTO commulingo_pipeline_artifacts(job_id,stage,value)
                        VALUES (%s,'handoff',%s)''',(row['id'],Json(value)))
                if row['review_status']=='approved' and gap_ids(row['payload']):
                    cur.execute('''UPDATE commulingo_curation_gaps SET status='done',resolved_id=%s,
                        resolution='Approved after pipeline handoff',updated_at=now()
                        WHERE id=ANY(%s) AND status='pending' ''',(row['target'],gap_ids(row['payload'])))
            return len(rows)
