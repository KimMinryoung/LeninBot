"""PostgreSQL queue with fenced leases and conservative cost reservations."""
from contextlib import contextmanager
from decimal import Decimal
import uuid
import hashlib
import json

from psycopg2.extras import Json, RealDictCursor


class LostLease(RuntimeError):
    pass


class BudgetUnavailable(RuntimeError):
    pass


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

    def claim(self, *, lease_seconds=120, group=None, job_id=None):
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
                (group, group, job_id, job_id, cursor, token, lease_seconds))
            job = cur.fetchone()
            if job:
                groups = ['person:create','person:update','term:create','term:update']
                cur.execute('UPDATE commulingo_pipeline_scheduler SET cursor=%s WHERE id=1',
                            ((groups.index(job['kind']+':'+job['action'])+1)%4,))
            return job

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
                       {'total_cost','rounds_used','input_tokens','output_tokens','pipeline_cache_hits'}}
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
            if job['stage']=='submit' and value.get('status')=='approved' and job['payload'].get('gap_id'):
                cur.execute('''UPDATE commulingo_curation_gaps SET status='done',resolved_id=%s,
                    resolution='Approved through durable pipeline',updated_at=now()
                    WHERE id=%s AND status='pending' ''',(job['target'],job['payload']['gap_id']))

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
        with self.transaction() as cur:
            cur.execute('''SELECT s.* FROM commulingo_pipeline_sources s
                JOIN commulingo_pipeline_job_sources j ON j.source_id=s.id WHERE j.job_id=%s''', (job_id,))
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
            cur.execute('''UPDATE commulingo_pipeline_jobs j SET status='complete',stage='complete',updated_at=now()
                FROM commulingo_pipeline_artifacts a JOIN commulingo_agent_suggestions s
                    ON s.id=(a.value->>'suggestionId')::bigint
                WHERE j.id=a.job_id AND j.status='escalated' AND a.stage='review'
                    AND s.status IN ('approved','rejected')
                RETURNING j.target,j.payload,s.status''')
            rows = cur.fetchall()
            for row in rows:
                if row['status']=='approved' and row['payload'].get('gap_id'):
                    cur.execute('''UPDATE commulingo_curation_gaps SET status='done',resolved_id=%s,
                        resolution='Approved after pipeline handoff',updated_at=now()
                        WHERE id=%s AND status='pending' ''',(row['target'],row['payload']['gap_id']))
            return len(rows)
