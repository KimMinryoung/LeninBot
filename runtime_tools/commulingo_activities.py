"""Shared activity catalogue and evidence-bound Jev decisions.

The frontend owns the versioned JSON catalogue and write schema. Citizenship
never supplies an affiliation. A selected evidence excerpt must support both
function and affiliation in the same activity; unsure affiliation stays null.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

CATALOG_PATH = Path(os.environ.get('COMMULINGO_ACTIVITY_CATALOG', '/home/grass/frontend/data/commulingo/activity-catalog.json'))
SCHEMA_PATH = Path(os.environ.get('COMMULINGO_ACTIVITY_SCHEMA', '/home/grass/frontend/data/commulingo/activity-schema.json'))

def load_catalog():
    return json.loads(CATALOG_PATH.read_text())


def activity_evidence(fields, claims=None):
    out = []
    for e in fields.get('evidence') or []:
        if e.get('field') not in {'bio','career','moment','role','activities','epithet'}:
            continue
        if all(isinstance(e.get(k), str) and e[k].strip() for k in ('source','locator','claim','excerpt')):
            out.append({k: e[k] for k in ('source','locator','claim','excerpt')})
    for field in ('career','bio','moment','activities'):
        for e in (claims or {}).get(field, []):
            if all(isinstance(e.get(k), str) and e[k].strip() for k in ('source','locator','claim','excerpt')):
                value = {k: e[k] for k in ('source','locator','claim','excerpt')}
                if value not in out:
                    out.append(value)
    return out[:16]


def activity_questions(catalog, evidence):
    return {
        'activity_function': {'type':'choice', 'criteria':{f['id']:f"{f['label']['en']}: {f['criteria']}" for f in catalog['functions']},
            'instructions':'Choose the defining documented activity, not citizenship, highest incidental title, victimhood or political sympathy. Choose an activity supported by ONE of the evidence excerpts.'},
        'activity_affiliation': {'type':'choice', 'criteria':{
            **{a['id']:f"{a['label']['en']} ({a['kind']}): {a['criteria']}" for a in catalog['affiliations']},
            'independent':'The chosen evidence explicitly establishes independent/unaffiliated activity.',
            'unresolved':'The evidence does not establish the organization served in the chosen activity, or it is absent from the catalogue.'},
            'instructions':'Choose the actual organization served or joined in the SAME activity selected above. Citizenship, ethnicity, residence and research subject are NOT affiliation. Prefer a named party/force over a generic state. For scholars and artists do not infer state service from nationality or a public university. Select unresolved when evidence is insufficient.'},
        'activity_basis': {'type':'choice', 'criteria':{str(i):e['excerpt'][:2400] for i,e in enumerate(evidence)} | {'unsupported':'No single excerpt supports the selected activity and affiliation together.'},
            'instructions':'Which excerpt supports BOTH the chosen function and affiliation in the SAME career? If affiliation is unresolved, support the function alone. Select unsupported if the two selections belong to different careers or periods.'},
    }


def activity_person_from(decision, catalog, evidence, group_ids, accept=0.7):
    group = decision.choice('group')
    function = decision.choice('activity_function')
    affiliation = decision.choice('activity_affiliation')
    basis = decision.choice('activity_basis')
    functions = {f['id']: f for f in catalog['functions']}
    affiliations = {a['id']: a for a in catalog['affiliations']}
    if group not in group_ids or function not in functions or affiliation not in {*affiliations,'independent','unresolved'}:
        return None
    if not isinstance(basis,str) or not basis.isdigit() or int(basis) >= len(evidence):
        return None
    e = evidence[int(basis)]
    if not all(e.get(k) for k in ('source','locator','claim','excerpt')):
        return None
    status = affiliation if affiliation in {'independent','unresolved'} else 'confirmed'
    kind = affiliations.get(affiliation,{}).get('kind')
    relation = status if status != 'confirmed' else ('membership' if kind == 'party' else 'employment' if kind == 'institution' else 'service')
    confidence = {k: round(decision.confidence(k) or 0.0,3) for k in ('group','activity_function','activity_affiliation','activity_basis')}
    activity = {'functionId':function,'affiliationId':affiliation if status=='confirmed' else None,
                'affiliationStatus':status,'relation':relation,'primary':True,
                'startYear':None,'endYear':None,'evidence':[e]}
    return {'groupId':group,'role':{'icon':functions[function]['icon']},'activities':[activity],
            'confidence':confidence,'low_confidence':min(confidence.values())<accept,'model':decision.model}
