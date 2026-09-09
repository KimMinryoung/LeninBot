"""Durable leases and operator handoff for pending person suggestions."""
import json
import uuid
from db import get_conn
from psycopg2.extras import RealDictCursor


def query(sql, params=(), *, one=False):
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall() if cur.description else []
    return (dict(rows[0]) if rows else None) if one else [dict(row) for row in rows]


def synchronize():
    query("""INSERT INTO commulingo_person_review_jobs(suggestion_id)
        SELECT id FROM commulingo_agent_suggestions WHERE status='pending' AND suggested_by!='commulingo-pipeline' AND
            (target_type IN ('person','person_section') OR (target_type='term' AND patch_json ? 'evidence'))
        ON CONFLICT DO NOTHING""")
    query("""UPDATE commulingo_person_review_jobs j SET status=s.status,lease_token=NULL,lease_until=NULL,updated_at=NOW()
        FROM commulingo_agent_suggestions s WHERE s.id=j.suggestion_id AND s.status IN ('approved','rejected') AND j.status<>s.status""")
    query("""UPDATE commulingo_person_review_jobs SET status=CASE WHEN attempts>=3 THEN 'escalated' ELSE 'retry' END,
        lease_token=NULL,lease_until=NULL,last_error='Review process stopped before completion',updated_at=NOW()
        WHERE status='reviewing' AND lease_until<NOW()""")


def claim():
    return query("""WITH candidate AS (
        SELECT j.suggestion_id FROM commulingo_person_review_jobs j JOIN commulingo_agent_suggestions s ON s.id=j.suggestion_id
        WHERE j.status IN ('queued','retry') AND j.next_attempt_at<=NOW() AND s.status='pending'
        ORDER BY j.next_attempt_at,j.suggestion_id FOR UPDATE OF j SKIP LOCKED LIMIT 1)
        UPDATE commulingo_person_review_jobs j SET status='reviewing',attempts=attempts+1,
        lease_token=%s,lease_until=NOW()+INTERVAL '20 minutes',updated_at=NOW()
        FROM candidate c WHERE j.suggestion_id=c.suggestion_id RETURNING j.*""", (uuid.uuid4().hex,), one=True)


def suggestion(sid):
    return query('SELECT * FROM commulingo_agent_suggestions WHERE id=%s', (sid,), one=True)


def owned(job):
    return query("SELECT suggestion_id FROM commulingo_person_review_jobs WHERE suggestion_id=%s AND lease_token=%s AND status='reviewing' AND lease_until>NOW()",
                 (job['suggestion_id'], job['lease_token']), one=True) is not None


def save_decision(job, decision, fetched):
    return query("""UPDATE commulingo_person_review_jobs SET decision=%s::jsonb,research=%s::jsonb,updated_at=NOW()
        WHERE suggestion_id=%s AND lease_token=%s AND status='reviewing' RETURNING suggestion_id""",
        (json.dumps(decision, ensure_ascii=False), json.dumps(fetched, ensure_ascii=False), job['suggestion_id'], job['lease_token']), one=True)


def finish(job, status, error=''):
    if status not in {'approved','rejected','escalated','retry'}:
        raise ValueError('invalid review job state')
    query("""UPDATE commulingo_person_review_jobs SET status=%s,last_error=%s,
        lease_token=NULL,lease_until=NULL,next_attempt_at=NOW()+INTERVAL '1 hour',updated_at=NOW()
        WHERE suggestion_id=%s AND lease_token=%s AND status='reviewing'""",
        (status, error[:2000], job['suggestion_id'], job['lease_token']))


def retry(sid):
    return query("""UPDATE commulingo_person_review_jobs j SET status='queued',attempts=0,decision=NULL,research='{}',
        last_error='',lease_token=NULL,lease_until=NULL,next_attempt_at=NOW(),notification_after=NOW(),updated_at=NOW()
        FROM commulingo_agent_suggestions s WHERE s.id=j.suggestion_id AND s.status='pending'
        AND j.suggestion_id=%s AND j.status IN ('retry','escalated') RETURNING j.suggestion_id""", (sid,), one=True)


def defer_budget(job):
    query("""UPDATE commulingo_person_review_jobs SET status='retry',attempts=GREATEST(attempts-1,0),
        lease_token=NULL,lease_until=NULL,next_attempt_at=NOW()+INTERVAL '1 hour',
        last_error='Daily budget deferred',updated_at=NOW()
        WHERE suggestion_id=%s AND lease_token=%s AND status='reviewing'""",
        (job['suggestion_id'],job['lease_token']))


def pending():
    return query("""SELECT s.id,s.target_id,s.target_type,s.action,s.review_note,
        COALESCE(j.status,'queued') AS review_status,j.last_error,j.decision
        FROM commulingo_agent_suggestions s LEFT JOIN commulingo_person_review_jobs j ON j.suggestion_id=s.id
        WHERE s.status='pending' AND (s.target_type IN ('person','person_section')
            OR (s.target_type='term' AND s.patch_json ? 'evidence')) ORDER BY s.id LIMIT 30""")


def detail(sid):
    row = suggestion(sid)
    if not row or row['target_type'] not in {'person','person_section','term'}:
        return None
    row['review_job'] = query('SELECT * FROM commulingo_person_review_jobs WHERE suggestion_id=%s', (sid,), one=True)
    return row


def notifications():
    # Reserving a short delivery lease avoids duplicate sends by overlapping workers.
    return query("""WITH due AS (
        SELECT j.suggestion_id FROM commulingo_person_review_jobs j JOIN commulingo_agent_suggestions s ON s.id=j.suggestion_id
        WHERE j.status='escalated' AND s.status='pending' AND j.notification_after<=NOW()
        ORDER BY j.suggestion_id FOR UPDATE OF j SKIP LOCKED LIMIT 5)
        UPDATE commulingo_person_review_jobs j SET notification_after=NOW()+INTERVAL '10 minutes'
        FROM due WHERE j.suggestion_id=due.suggestion_id RETURNING j.*""")


def notification_sent(sid):
    query("UPDATE commulingo_person_review_jobs SET notified_at=NOW(),notification_after=NOW()+INTERVAL '1 day' WHERE suggestion_id=%s", (sid,))
