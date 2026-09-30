"""Shared activity catalogue and evidence-bound Jev decisions.

The frontend owns the versioned JSON catalogue and write schema. Citizenship
never supplies an affiliation. A selected evidence excerpt must support both
function and affiliation in the same activity; unsure affiliation stays null.
"""
from __future__ import annotations
import json
import re

from ops.paths import commulingo_data_file

CATALOG_PATH = commulingo_data_file('activity-catalog.json', 'COMMULINGO_ACTIVITY_CATALOG')
SCHEMA_PATH = commulingo_data_file('activity-schema.json', 'COMMULINGO_ACTIVITY_SCHEMA')

def load_catalog():
    return json.loads(CATALOG_PATH.read_text())


def activity_evidence(fields, claims=None):
    out = []
    for e in fields.get('evidence') or []:
        if e.get('field') not in {'bio','career','moment','activities','epithet'}:
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


_YEAR_RE = re.compile(r'(?<![\d\[])(1[0-9]{3}|20[0-9]{2})(?![\d\]])')


def excerpt_years(text, span=None):
    """Four-digit years an excerpt names, within the person's life when known
    (a footnote such as [193] is not a year; a later historian's date is)."""
    years = sorted({int(y) for y in _YEAR_RE.findall(str(text or ''))})
    if span:
        years = [y for y in years if span[0] - 20 <= y <= span[1]]
    return years


def periods_overlap(periods, start, end):
    """Same rule as the frontend validator: one year of slack at each end."""
    if not periods or (start is None and end is None):
        return True
    start, end = (start if start is not None else end), (end if end is not None else start)
    return any((a is None or a - 1 <= end) and (b is None or start <= b + 1) for a, b in periods)


def affiliations_for(catalog, window=None):
    """Affiliations that existed during the window (start, end); entries
    without periods are always offered."""
    if not window:
        return catalog['affiliations']
    return [a for a in catalog['affiliations'] if periods_overlap(a.get('periods'), *window)]


def activity_questions(catalog, evidence, window=None):
    return {
        'activity_function': {'type':'choice', 'criteria':{f['id']:f"{f['label']['en']}: {f['criteria']}" for f in catalog['functions']},
            'instructions':'Choose the defining documented activity, not citizenship, highest incidental title, victimhood or political sympathy. The defining career is the one the sources and card identify the person by, usually their opening description; do not replace it with a shorter or lesser-known earlier career. Prefer the specific field that defines the career over government, even for ministers or heads of state. Government requires defining cross-policy executive management or administrative coordination; never use it as a catch-all for officials or uncertain cases. A nominal or ceremonial head of state, or the chair of a legislature or its presidium, whose office defines the career while real power lay elsewhere belongs to legislature, not government. Choose an activity supported by ONE of the evidence excerpts.'},
        'activity_affiliation': {'type':'choice', 'criteria':{
            **{a['id']:f"{a['label']['en']} ({a['kind']}): {a['criteria']}" for a in affiliations_for(catalog, window)},
            'independent':'The chosen evidence explicitly establishes independent/unaffiliated activity.',
            'unresolved':'The evidence does not establish the organization served in the chosen activity, or it is absent from the catalogue.'},
            'instructions':'Choose the actual state, organization or force served in the career the selected excerpt documents (selected_activity_evidence). Only affiliations that existed in that period are offered. Citizenship, ethnicity, residence and research subject are NOT affiliation. For a one-party socialist system, choose the STATE for activity as part of that system during its ruling period: ruling-party leadership, cadres, government, state institutions, armed forces and security, including non-party members. Store the country, not the ruling party. This rule does NOT apply to opposition against that regime: choose its documented opposition organization, independent or unresolved; earlier state service is a separate activity. Pre-state revolutionary activity and party activity outside the ruling period retain the party or movement. Otherwise prefer a documented specific organization over a generic state. In the French Revolution choose the named club, faction, Paris Commune or the regime of the date (Bourbon monarchy, First Republic, Consulate/Empire); the generic revolutionary camp only when none is named. For scholars and artists do not infer state service from nationality or a public university. Select unresolved when evidence is insufficient.'},
        'activity_basis': {'type':'choice', 'criteria':{str(i):e['excerpt'][:2400] for i,e in enumerate(evidence)} | {'unsupported':'No excerpt documents the selected function.'},
            'instructions':'Which excerpt documents the selected function as this person\'s DEFINING activity, the career the card is about? Prefer the excerpt of that career over a later or incidental episode. Select unsupported if no excerpt documents the function.'},
    }


def activity_basis_question(catalog, evidence, function, group=None):
    """Use identical source-grounding criteria for registration and audits."""
    from commulingo.classify import FRENCH_REVOLUTION_GROUPS, FRENCH_REVOLUTION_BASIS
    question = dict(activity_questions(catalog, evidence)['activity_basis'])
    selected = next(f for f in catalog['functions'] if f['id'] == function)
    question['instructions'] += ' The function is fixed by selected_activity_function in the state.'
    question['instructions'] += ' Apply the selected function criteria: ' + selected['criteria']
    if function == 'government':
        question['instructions'] += ' A high office title alone is insufficient. Select unsupported unless an excerpt establishes cross-policy executive management or administrative coordination as the defining career; a specialized portfolio alone does not qualify.'
    if function == 'legislature':
        question['instructions'] += ' Select unsupported unless an excerpt establishes presiding over a legislature or its presidium, or a ceremonial or nominal non-monarchical head-of-state office, as the defining career; ordinary legislative membership does not qualify, and monarchs belong to monarchy.'
    if function == 'monarchy':
        question['instructions'] += ' Select unsupported unless an excerpt establishes reigning as a monarch, ruling as regent or serving as a royal consort as the defining career; royal birth or a noble title alone does not qualify.'
    if group in FRENCH_REVOLUTION_GROUPS:
        question['instructions'] += FRENCH_REVOLUTION_BASIS
    return question


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
    return {'groupId':group,'activities':[activity],
            'confidence':confidence,'low_confidence':min(confidence.values())<accept,'model':decision.model}


def activity_search_params(function_id, affiliation_id, catalog=None):
    """Validate filters and expand descendants using the same public catalogue."""
    catalog = catalog or load_catalog()
    affiliations = {a['id']: a for a in catalog['affiliations']}
    if function_id and function_id not in {f['id'] for f in catalog['functions']}:
        raise ValueError('unknown function_id; use list_activity_catalog')
    if affiliation_id and affiliation_id not in affiliations:
        raise ValueError('unknown affiliation_id; use list_activity_catalog')
    descendants = []
    for candidate in affiliations:
        current, seen = candidate, set()
        while current and current not in seen:
            if current == affiliation_id:
                descendants.append(candidate)
                break
            seen.add(current)
            current = affiliations.get(current, {}).get('parentId')
    return {'function': function_id, 'affiliation': affiliation_id,
            'descendants': descendants}
