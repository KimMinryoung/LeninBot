"""Scheduled pipeline observations; never claim/retry/translate/publish a job."""
from __future__ import annotations

import asyncio
from ops.readonly_query import query as _query
from ops.diagnostics import observe, report, systemd_properties, SERVICE_UNITS

PIPELINES = ('commulingo', 'translation', 'kg_sync')
EXCLUSIONS = ['archival translation batches', 'manual static-page translation', 'manual Markdown file translation']


def commulingo_queue(limit):
    counts = _query('''SELECT status, stage, count(*) AS jobs,
        count(*) FILTER (WHERE status='running' AND lease_until < now()) AS expired_leases,
        count(*) FILTER (WHERE status IN ('ready','deferred') AND available_at < now() - interval '24 hours') AS stalled_24h,
        min(created_at) AS oldest_created_at FROM commulingo_pipeline_jobs GROUP BY status,stage ORDER BY 1,2''', ())
    recent = _query('''SELECT id,kind,action,target,stage,status,attempts,updated_at,available_at,lease_until,
        left(last_error, 1000) AS last_error FROM commulingo_pipeline_jobs ORDER BY updated_at DESC,id DESC LIMIT %s''', (limit,))
    failures = _query('''SELECT id,stage,status,updated_at,left(last_error,1000) AS last_error
        FROM commulingo_pipeline_jobs WHERE COALESCE(last_error,'') <> '' ORDER BY updated_at DESC,id DESC LIMIT %s''', (limit,))
    return {'counts': counts, 'recent_jobs': recent, 'recent_errors_or_deferrals': failures,
            'stalled_definition': 'expired running lease or ready/deferred eligible for over 24 hours'}


def translation_queue(kind, limit):
    from translation_runtime.freshness import (TABLES, missing_translation_sql, changed_source_sql,
                                             pending_translation_sql, cooldown_fingerprint_sql)
    from translation_runtime.batch_state import BatchState
    scope = "status = 'public' AND " if kind == 'research' else ''
    # Only identities, flags and fingerprints cross this read boundary, never source bodies.
    rows = _query(f'''SELECT id, ({missing_translation_sql(kind)}) AS missing,
        ({changed_source_sql(kind)}) AS source_changed, {cooldown_fingerprint_sql(kind)} AS fingerprint
        FROM {TABLES[kind]} WHERE {scope}({pending_translation_sql(kind)}) ORDER BY id''', ())
    state = BatchState()
    pending = []
    deferred = 0
    validation_failures = []
    for row in rows:
        held = state.deferred(f"{kind}:{row['id']}", row['fingerprint'])
        deferred += int(held)
        record = state.get(f"{kind}:{row['id']}")
        if record.get("source") == row["fingerprint"] and record.get("count", 0):
            validation_failures.append({"id": row["id"], "failure_count": record["count"],
                                        "retry_after_epoch": record.get("retry_after"), "deferred": held})
        pending.append({'id': row['id'], 'missing': row['missing'], 'source_changed': row['source_changed'], 'deferred': held})
    return {'pending': len(rows), 'missing': sum(bool(r['missing']) for r in rows),
            'source_changed': sum(bool(r['source_changed']) for r in rows), 'deferred': deferred,
            'eligible': len(rows) - deferred, 'sample': pending[:limit],
            'validation_failures': validation_failures[:limit],
            'note': 'Missing and source_changed may overlap; deferred matches the current source fingerprint.'}


def kg_queue():
    from kg_runtime.metrics import sync_metrics, sync_unhealthy
    states = sync_metrics(query=_query)
    if states.get('error'):
        raise RuntimeError(states['error'])
    if not states:
        raise RuntimeError('KG sync state unavailable (missing table or no recorded runs)')
    return {name: {**state, 'unhealthy': sync_unhealthy(state)} for name, state in states.items()}


def failure_logs(name, limit):
    from ops.logs import fetch_server_logs
    rows = fetch_server_logs(name, hours_back=168, grep=['error', 'failed', 'traceback'], limit=1000)
    if rows and rows[0].get('error'):
        raise RuntimeError(rows[0]['error'])
    return {'rows': [{'raw': row['raw'][:1000]} for row in rows[-limit:]],
            'scope': 'error/failed/traceback text in latest 1000 journal rows within 168 hours'}


async def snapshot(pipeline, limit):
    names = PIPELINES if pipeline == 'all' else (pipeline,)
    async def one(name):
        unit = SERVICE_UNITS[name]
        parts = dict(zip(('last_run', 'timer', 'recent_failure_logs'), await asyncio.gather(
            observe(lambda: systemd_properties(unit)),
            observe(lambda: systemd_properties(unit.replace('.service', '.timer'), timer=True)),
            observe(lambda: failure_logs(name, limit), timeout=20))))
        if name == 'translation':
            kinds = ('research', 'posts', 'diary', 'curation')
            parts['queue'] = dict(zip(kinds, await asyncio.gather(*(
                observe(lambda k=k: translation_queue(k, limit), timeout=30) for k in kinds))))
        else:
            parts['queue'] = await observe(lambda: commulingo_queue(limit) if name == 'commulingo' else kg_queue())
        return name, parts
    return report(dict(await asyncio.gather(*(one(name) for name in names))), exclusions=EXCLUSIONS)
