"""leninbot's research source cache for the CommuLingo agent sessions.

The enrichment queue moved to the frontend (frontend
dev_docs/commulingo-agent-pipeline.md, 2026-10-05). What stays here is the
cache of pages the author and review sessions fetched (commulingo_pipeline_sources,
fetch_cache, job_sources): research infrastructure, reused across a job's
attempts so a retry does not pay for the same page twice.
"""
from contextlib import contextmanager
import hashlib
import json

from psycopg2.extras import RealDictCursor


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
