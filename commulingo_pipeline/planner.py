"""Incremental source intake and deterministic missing-content commissions."""
import hashlib
from collections import deque


class Planner:
    def __init__(self, store):
        self.store = store

    def candidates(self, limit=40):
        with self.store.transaction() as cur:
            cur.execute('''SELECT 'person' AS kind,'update' AS action,p.id AS target,
                topic.name AS topic,topic.priority,
                'Commissioned missing information or evidence: ' || topic.name AS reason,
                concat_ws(':',p.updated_at::text,
                    (SELECT max(e.created_at)::text FROM commulingo_person_evidence e WHERE e.person_id=p.id),
                    (SELECT max(s.updated_at)::text FROM commulingo_person_sections s WHERE s.person_id=p.id)) AS baseline
                FROM commulingo_people p CROSS JOIN LATERAL (VALUES
                    ('basics',20,p.years_label='' OR p.epithet_ko='' OR p.epithet_en=''
                        OR NOT EXISTS (SELECT 1 FROM commulingo_person_roles r WHERE r.person_id=p.id)
                        OR NOT EXISTS (SELECT 1 FROM commulingo_person_career_entries c WHERE c.person_id=p.id)
                        OR NOT EXISTS (SELECT 1 FROM commulingo_person_evidence e WHERE e.person_id=p.id AND e.field='years')),
                    ('bio',30,p.bio_ko='' OR p.bio_en='' OR NOT EXISTS
                        (SELECT 1 FROM commulingo_person_evidence e WHERE e.person_id=p.id AND e.field='bio')),
                    ('nationality',35,p.citizenship_code='' OR p.origin_code='' OR NOT EXISTS
                        (SELECT 1 FROM commulingo_person_evidence e WHERE e.person_id=p.id AND e.field='citizenship')
                        OR NOT EXISTS (SELECT 1 FROM commulingo_person_evidence e WHERE e.person_id=p.id AND e.field IN ('nationalOrigin','origin'))),
                    ('moment',40,p.moment_ko='' OR p.moment_en='' OR NOT EXISTS
                        (SELECT 1 FROM commulingo_person_evidence e WHERE e.person_id=p.id AND e.field='moment')),
                    ('sections',50,(SELECT count(*) FROM commulingo_person_sections s WHERE s.person_id=p.id)<12)
                ) AS topic(name,priority,needed)
                WHERE topic.needed AND NOT EXISTS (SELECT 1 FROM commulingo_person_enrichment e
                    WHERE e.person_id=p.id AND e.topic=topic.name AND e.status!='open' AND e.review_after>now())
                AND NOT EXISTS (SELECT 1 FROM commulingo_agent_suggestions s
                    WHERE s.target_id=p.id AND s.target_type IN ('person','person_section') AND s.status='pending')
                UNION ALL
                SELECT 'term','update',t.id,topic.name,topic.priority,
                    'Commissioned glossary explanation: ' || topic.name,t.updated_at::text
                FROM commulingo_terms t CROSS JOIN LATERAL (VALUES
                    ('definition',30,t.definition_ko='' OR t.definition_en='' OR NOT EXISTS
                        (SELECT 1 FROM commulingo_term_evidence e WHERE e.term_id=t.id AND e.field='definition')),
                    ('history',40,t.body_ko='' OR t.body_en='' OR NOT EXISTS
                        (SELECT 1 FROM commulingo_term_evidence e WHERE e.term_id=t.id AND e.field='body')),
                    ('distinctions',50,true),('examples',50,true),
                    ('relations',50,NOT EXISTS (SELECT 1 FROM commulingo_term_people r WHERE r.term_id=t.id)
                        OR NOT EXISTS (SELECT 1 FROM commulingo_term_events r WHERE r.term_id=t.id))
                ) AS topic(name,priority,needed)
                WHERE topic.needed AND NOT EXISTS (SELECT 1 FROM commulingo_term_enrichment e
                    WHERE e.term_id=t.id AND e.topic=topic.name AND e.status!='open' AND e.review_after>now())
                AND NOT EXISTS (SELECT 1 FROM commulingo_agent_suggestions s
                    WHERE s.target_id=t.id AND s.target_type='term' AND s.status='pending')
                ORDER BY priority,target''')
            candidates = [dict(row) for row in cur.fetchall()]
            cur.execute('''SELECT kind,CASE WHEN
                    (kind='person' AND EXISTS (SELECT 1 FROM commulingo_people p WHERE p.id=COALESCE(NULLIF(g.target_id,''),NULLIF(g.resolved_id,'')))) OR
                    (kind='term' AND EXISTS (SELECT 1 FROM commulingo_terms t WHERE t.id=COALESCE(NULLIF(g.target_id,''),NULLIF(g.resolved_id,''))))
                    THEN 'update' ELSE 'create' END AS action,COALESCE(NULLIF(target_id,''),
                    NULLIF(resolved_id,'')) AS target,'basics' AS topic,
                    'Explicit gap: ' || label_ko AS reason,10 AS priority,'' AS baseline,
                    id AS gap_id,label_ko,label_en FROM commulingo_curation_gaps g
                WHERE status='pending' AND kind IN ('person','term')
                ORDER BY priority DESC,id LIMIT %s''',(limit,))
            for row in cur.fetchall():
                row = dict(row)
                if row['target']:
                    if row['kind']=='term':
                        row['topic']='definition'
                    row['payload'] = {k:row.pop(k) for k in ('gap_id','label_ko','label_en')}
                    candidates.append(row)
            # Completed judgements are suppressed until changed content or TTL expiry.
            cur.execute('''SELECT kind,target,topic,baseline FROM commulingo_pipeline_jobs
                WHERE status IN ('ready','running','deferred','escalated') OR
                    (status='complete' AND updated_at>now() - CASE WHEN EXISTS (
                        SELECT 1 FROM commulingo_pipeline_artifacts a WHERE a.job_id=commulingo_pipeline_jobs.id
                        AND a.stage='research' AND a.value->>'status'='sources_unavailable')
                        THEN interval '90 days' ELSE interval '180 days' END)''')
            covered = {(r['kind'],r['target'],r['topic'],r['baseline']) for r in cur.fetchall()}
            available = [r for r in candidates if (r['kind'],r['target'],r['topic'],r['baseline']) not in covered]
            urgent = sorted((r for r in available if r['priority']<20),key=lambda r:(r['priority'],r['target']))
            groups = [deque(r for r in available if r['priority']>=20 and (r['kind'],r['action'])==g)
                      for g in [('person','create'),('person','update'),('term','create'),('term','update')]]
            selected = urgent[:limit]
            while len(selected)<limit and any(groups):
                for group in groups:
                    if group and len(selected)<limit:
                        selected.append(group.popleft())
            return selected

    def materials(self, limit=10):
        with self.store.transaction() as cur:
            cur.execute('''WITH materials AS (
                SELECT 'report:' || slug AS material_id,title AS label,markdown AS body,NULL::text AS requested_kind
                FROM research_documents WHERE status='public'
                UNION ALL SELECT 'event:' || id,title_ko,
                    concat_ws(E'\\n',summary_ko,outcome_ko,timeline::text),NULL FROM commulingo_history_events
                UNION ALL SELECT 'person:' || id,name_ko,concat_ws(E'\\n',bio_ko,moment_ko),NULL
                FROM commulingo_people
                UNION ALL SELECT 'gap:' || id::text,label_ko,concat_ws(E'\\n',label_ko,label_en,reason),kind
                FROM commulingo_curation_gaps WHERE status='pending' AND kind IN ('person','term')
                    AND NULLIF(target_id,'') IS NULL AND NULLIF(resolved_id,'') IS NULL)
                SELECT m.* FROM materials m LEFT JOIN commulingo_pipeline_materials p USING(material_id)
                WHERE COALESCE(body,'')!='' AND (p.material_id IS NULL OR p.content_hash!=md5(m.body))
                ORDER BY CASE WHEN material_id LIKE 'gap:%%' THEN 0 ELSE 1 END,material_id LIMIT %s''',(limit,))
            rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            # md5 is a change detector only, never an evidence integrity hash.
            row['content_hash'] = hashlib.md5(row['body'].encode(),usedforsecurity=False).hexdigest()
        return rows

    def plan(self, *, apply=False, limit=40):
        candidates = self.candidates(limit)
        materials = self.materials(min(limit,10))
        if apply:
            for candidate in candidates:
                self.store.enqueue(**candidate)
            for material in materials:
                target = 'material-'+hashlib.sha256((material['material_id']+material['content_hash']).encode()).hexdigest()[:32]
                self.store.enqueue(kind='term',action='create',target=target,topic='discovery',
                    reason='New or changed public material',priority=10 if material['material_id'].startswith('gap:') else 60,payload=material,stage='discover')
        return {'commissions':candidates,'materials':[{k:v for k,v in m.items() if k!='body'} for m in materials],
                'applied':apply}
