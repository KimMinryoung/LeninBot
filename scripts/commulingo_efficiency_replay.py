#!/usr/bin/env python3
"""Offline regression replay of 20 saved drafts. Never calls models or publishes.

Input: JSON array, five cases per person/term x create/update, each containing
job_id, kind, action, draft (the persisted draft artifact). No credentials needed.
This checks draft compatibility and repair continuity, NOT LLM quality or cost.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def replay(cases, repair_class=None):
    from commulingo.pipeline.draft_repair import DraftRepair
    from scripts.commulingo_write_session import draft_id
    from commulingo.people import (COMMULINGO_PERSON_CREATE_TOOL,
        COMMULINGO_PERSON_UPDATE_TOOL,COMMULINGO_TERM_CREATE_TOOL,COMMULINGO_TERM_UPDATE_TOOL)
    tools={('person','create'):COMMULINGO_PERSON_CREATE_TOOL,
           ('person','update'):COMMULINGO_PERSON_UPDATE_TOOL,
           ('term','create'):COMMULINGO_TERM_CREATE_TOOL,
           ('term','update'):COMMULINGO_TERM_UPDATE_TOOL}
    repair_class = repair_class or DraftRepair
    counts=Counter((c['kind'],c['action']) for c in cases)
    if counts!=Counter({key:5 for key in tools}):
        raise ValueError('exactly five cases per person/term x create/update required')
    if len({c['job_id'] for c in cases})!=20:
        raise ValueError('case job IDs must be distinct')
    results=[]
    for case in cases:
        schema=deepcopy(tools[case['kind'],case['action']]['input_schema']['properties']['fields'])
        attached={'evidence','expectedRevision','sources','confidence'}
        for field in attached:
            schema['properties'].pop(field,None)
        schema['required']=[f for f in schema.get('required',[]) if f not in attached]
        tool={'name':'commulingo_pipeline_result','input_schema':{'type':'object',
            'properties':{'fields':schema},'required':['fields'],'additionalProperties':False}}
        session=repair_class(tool)
        fields={k:v for k,v in case['draft']['fields'].items() if k not in attached}
        row={'job_id':case['job_id'],'kind':case['kind'],'action':case['action'],'compatible':False}
        try:
            prepared=session.prepare({'fields':fields})
            row['compatible'] = True
            # An external storage rejection must leave the exact prepared draft
            # available. Replay an idempotent repair, without changing any prose.
            key=next(iter(fields))
            current=draft_id(session.draft)
            repaired=session.prepare({'draft_id':current,'repairs':[
                {'op':'set','path':'/fields/'+key,'value':deepcopy(fields[key])}]})
            if prepared!=repaired or repaired!={'fields':fields}:
                raise ValueError('repair changed unrequested content')
            row.update(compatible=True,repair_continuity=True)
        except (ValueError,KeyError,TypeError) as exc:
            row.update(repair_continuity=False,error=str(exc))
        results.append(row)
    return {'mode':'offline_saved_draft_replay','cases':results,
            'compatible':sum(r['compatible'] for r in results),
            'repair_continuity':sum(r['repair_continuity'] for r in results),
            'paid_calls':0,'quality_gate':'not_measured','cost_reduction_gate':'not_measured',
            'production_rollout_eligible':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixtures',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--baseline-ref',help='Trusted repository commit to compare its DraftRepair implementation')
    args=parser.parse_args()
    cases=json.loads(args.fixtures.read_text())
    result=replay(cases)
    if args.baseline_ref:
        import subprocess
        source=subprocess.run(['git','show',args.baseline_ref+':commulingo/pipeline/draft_repair.py'],
            cwd=ROOT,check=True,capture_output=True,text=True).stdout
        namespace={}
        exec(compile(source,'baseline_draft_repair.py','exec'),namespace)
        baseline=replay(cases,namespace['DraftRepair'])
        result['baseline']={k:v for k,v in baseline.items() if k!='cases'}
        result['baseline_ref']=args.baseline_ref
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
    return 0 if result['compatible']==20 and result['repair_continuity']==20 else 1


if __name__=='__main__':
    raise SystemExit(main())
