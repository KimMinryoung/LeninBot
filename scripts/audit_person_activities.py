#!/usr/bin/env python3
"""Read-only review of every person's function and actual service affiliation.

Jev receives public dictionary biographies/careers, never user records. Output
is a candidate report, NOT evidence or an approved edit. In particular stored
biography text is not presented as a quotation from an external source.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from commulingo.activities import load_catalog, activity_questions
from scripts.commulingo_classification_audit import state_of
from llm.call_registry import decide_detailed
from db import query


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--concurrency',type=int,default=8)
    args=ap.parse_args()
    catalog=load_catalog()
    people=query("""SELECT p.id,p.name_ko,p.years_label,p.epithet_ko,p.bio_ko,p.bio_en,p.group_id,p.citizenship_code,
        p.updated_at,r.category_id,r.office_id,
        (SELECT json_agg(json_build_object('t',e.role_ko,'y',e.period_label) ORDER BY e.sort_order)
         FROM commulingo_person_career_entries e WHERE e.person_id=p.id) career,
        (SELECT count(*) FROM commulingo_person_evidence e WHERE e.person_id=p.id) evidence_count,
        (SELECT count(*) FROM commulingo_person_sections s WHERE s.person_id=p.id AND sources<>'[]'::jsonb) sourced_sections
        FROM commulingo_people p LEFT JOIN commulingo_person_roles r ON r.person_id=p.id ORDER BY p.id""")
    if args.limit: people=people[:args.limit]
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps({'catalog':catalog,'people':people},sort_keys=True,default=str).encode()).hexdigest()
    existing={}
    if out.exists():
        for line in out.read_text().splitlines():
            r=json.loads(line)
            if r.get('fingerprint')==fingerprint and not r.get('error'):existing[r['id']]=r
    def judge(p):
        if p['id'] in existing:return existing[p['id']]
        state=state_of(p)
        questions=activity_questions(catalog,[{'excerpt':json.dumps(state,ensure_ascii=False)}])
        questions.pop('activity_basis')
        result=decide_detailed('commulingo_classification_audit',state,questions,label='activity-affiliation-audit')
        row={'id':p['id'],'name':p['name_ko'],'legacy':p['office_id'] or p['category_id'],
             'source_records':p['evidence_count']+p['sourced_sections'],'fingerprint':fingerprint}
        if result.decision is None:return {**row,'error':result.error}
        d=result.decision
        function=d.choice('activity_function');affiliation=d.choice('activity_affiliation')
        if function not in {f['id'] for f in catalog['functions']} or affiliation not in {a['id'] for a in catalog['affiliations']}|{'unresolved','independent'}:
            return {**row,'error':'out-of-catalog answer'}
        legacy=catalog['legacy'].get(row['legacy'])
        return {**row,'candidate':{'functionId':function,'affiliationId':affiliation},
                'confidence':{'function':d.confidence('activity_function'),'affiliation':d.confidence('activity_affiliation')},
                'legacy_matches':bool(legacy and legacy[0]==function and (not legacy[1] or legacy[1]==affiliation)),
                'status':'candidate_requires_source_review','cost':d.cost_usd or 0.0}
    rows=[]
    with out.open('w') as file,ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for r in pool.map(judge,people):
            rows.append(r);file.write(json.dumps(r,ensure_ascii=False)+'\n');file.flush()
            if len(rows)%100==0:print(f'{len(rows)}/{len(people)} reviewed',flush=True)
    errors=[r for r in rows if 'error' in r]
    summary={'people':len(rows),'errors':len(errors),'legacy_matches':sum(r.get('legacy_matches',False) for r in rows),'cost':sum(r.get('cost',0) for r in rows)}
    out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    return bool(errors)

if __name__=='__main__':sys.exit(main())
