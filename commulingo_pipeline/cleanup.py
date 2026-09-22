"""Bounded retirement of untouched automatic commissions; no model or RPC calls."""
from .issues import commission
from .bundles import topics

REASON = 'automatic commission has no remaining missing fields'
TABLES = ('commulingo_pipeline_jobs', 'commulingo_pipeline_artifacts',
          'commulingo_pipeline_attempts', 'commulingo_pipeline_budget',
          'commulingo_pipeline_job_sources', 'commulingo_people', 'commulingo_terms',
          'commulingo_person_roles', 'commulingo_person_career_entries',
          'commulingo_person_sections')


def has_work(job, current):
    # Evaluate every remaining topic, including sections deferred behind the card.
    return any(commission({**job, 'topic': topic,
                           'payload': {**(job.get('payload') or {}), 'remaining_topics': [topic]}}, current)
               for topic in topics(job))


def retire(store, *, apply=False, limit=200):
    if not 1 <= limit <= 200:
        raise ValueError('cleanup limit must be 1..200')
    with store.transaction() as cur:
        # All decisions use a single DB snapshot. Applying holds short NOWAIT
        # locks against both target edits and worker claims/history writes.
        # No network or model calls occur while these locks are held.
        if apply:
            cur.execute('LOCK TABLE ' + ','.join(sorted(TABLES)) + ' IN SHARE ROW EXCLUSIVE MODE NOWAIT')
        cur.execute('''SELECT j.*, CASE WHEN j.kind='person' THEN
            jsonb_build_object('years',p.years_label,
                'epithet',jsonb_build_object('ko',p.epithet_ko,'en',p.epithet_en),
                'bio',jsonb_build_object('ko',p.bio_ko,'en',p.bio_en),
                'moment',jsonb_build_object('ko',p.moment_ko,'en',p.moment_en),
                'citizenship',jsonb_build_object('label',jsonb_build_object('ko',p.citizenship_label_ko,'en',p.citizenship_label_en)),
                'nationalOrigin',jsonb_build_object('label',jsonb_build_object('ko',p.origin_label_ko,'en',p.origin_label_en)),
                'role',EXISTS(SELECT 1 FROM commulingo_person_roles r WHERE r.person_id=p.id),
                'career',EXISTS(SELECT 1 FROM commulingo_person_career_entries c WHERE c.person_id=p.id),
                'sections',EXISTS(SELECT 1 FROM commulingo_person_sections s WHERE s.person_id=p.id))
            ELSE jsonb_build_object('definition',jsonb_build_object('ko',t.definition_ko,'en',t.definition_en),
                'body',jsonb_build_object('ko',t.body_ko,'en',t.body_en)) END AS current
            FROM commulingo_pipeline_jobs j
            LEFT JOIN commulingo_people p ON j.kind='person' AND p.id=j.target
            LEFT JOIN commulingo_terms t ON j.kind='term' AND t.id=j.target
            WHERE j.action='update' AND j.status='ready' AND j.stage='research'
              AND j.attempts=0 AND j.lease_token IS NULL
              AND (p.id IS NOT NULL OR t.id IS NOT NULL)
              AND (j.reason LIKE 'Commissioned %%' OR j.reason LIKE 'Bundled enrichment:%%')
              AND NOT (j.payload ?| ARRAY['gap_id','review_feedback','original_proposal','replaces_suggestion_id'])
              AND coalesce(j.payload->'gap_ids','[]'::jsonb)='[]'::jsonb
              AND coalesce(j.payload->>'workflow','editor')='editor'
              AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=j.id)
              AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_attempts a WHERE a.job_id=j.id)
              AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_budget b WHERE b.job_id=j.id)
              AND NOT EXISTS (SELECT 1 FROM commulingo_pipeline_job_sources s WHERE s.job_id=j.id)
            ORDER BY j.updated_at,j.id LIMIT %s''', (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        retired = [r for r in rows if automatic(r) and not has_work(r, r['current'])]
        if apply:
            # Rotate retained candidates so one full batch cannot starve later
            # jobs. No attempt, cost, artifact or public content is created.
            if rows:
                cur.execute('UPDATE commulingo_pipeline_jobs SET updated_at=now() WHERE id=ANY(%s)',
                            ([r['id'] for r in rows],))
            if retired:
                cur.execute("""UPDATE commulingo_pipeline_jobs SET status='cancelled',last_error=%s
                    WHERE id=ANY(%s)""", (REASON, [r['id'] for r in retired]))
        return {'applied': apply, 'examined': len(rows), 'retired': len(retired) if apply else 0,
                'candidates': [{'id':r['id'], 'kind':r['kind'], 'target':r['target'],
                                'topics':topics(r)} for r in retired]}


def automatic(job):
    payload = job.get('payload') or {}
    if any(payload.get(k) for k in ('gap_id','gap_ids','review_feedback','original_proposal','replaces_suggestion_id')):
        return False
    if not job['reason'].startswith(('Commissioned ', 'Bundled enrichment:')):
        return False
    return all(automatic(c) for c in payload.get('commissions', []))
