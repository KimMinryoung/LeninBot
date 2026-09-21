"""Persistent fictional illness conditions, independent of injury healing clocks.

Rates are game constants, not clinical predictions. Time alone never diagnoses,
worsens or cures a disease; only explicit current-scene evidence changes its course.
"""
from copy import deepcopy

KINDS = {'pneumonia': '폐렴'}
STATUSES = {'active': '지속', 'worsening': '악화 중', 'recovering': '회복 중', 'resolved': '회복 완료'}
SEVERITIES = {'unknown': '미확인', 'mild': '경미', 'moderate': '보통', 'severe': '심함'}
TREATMENTS = {'unknown': '치료 여부 미확인', 'untreated': '미치료', 'treated': '치료받음'}
FEVERS = {'unknown': '발열 여부 미확인', 'yes': '발열 있음', 'no': '발열 없음'}


def validate_illnesses(items):
    if not isinstance(items, list) or len(items) > len(KINDS):
        raise ValueError('Invalid illnesses list')
    seen = set()
    for item in items:
        fields = {'kind', 'status', 'severity', 'treatment', 'fever', 'elapsed_minutes', 'source', 'scope_id'}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError('Invalid illness record')
        for key, allowed in (('kind', KINDS), ('status', STATUSES), ('severity', SEVERITIES),
                             ('treatment', TREATMENTS), ('fever', FEVERS)):
            if not isinstance(item[key], str) or item[key] not in allowed:
                raise ValueError('Invalid illness ' + key)
        if item['kind'] in seen:
            raise ValueError('Duplicate illness')
        seen.add(item['kind'])
        if type(item['elapsed_minutes']) is not int or item['elapsed_minutes'] < 0:
            raise ValueError('Invalid illness elapsed time')
        if any(not isinstance(item[k], str) or not 1 <= len(item[k]) <= limit
               for k, limit in (('source', 400), ('scope_id', 100))):
            raise ValueError('Invalid illness provenance')


def questions(state, choice):
    result = {}
    for kind, name in KINDS.items():
        key = 'disease_' + kind
        rules = (f'Judge {name} ({kind}) ONLY for the current subject in the enacted current scene. '
                 'Historical sources, old episodes, hypothetical discussion and future plans are not current illness. '
                 'Do not diagnose pneumonia from fever, cough, injury or distress alone. '
                 'A current explicit statement/diagnosis establishes it. Do not infer recovery from rest, time, or starting treatment. ')
        result[key] = choice(rules + 'Does the current scene explicitly establish or change this disease? keep preserves the stored status.', {
            'keep': 'No explicit new disease/course evidence; preserve existing record, including no record',
            'active': 'Explicitly establishes current disease, including an explicit recurrence after recovery',
            'worsening': 'Explicit worsening of established current disease',
            'recovering': 'Explicit improvement of established current disease; not merely treatment started',
            'resolved': 'Explicit current complete recovery or correction that this diagnosis was wrong',
        })
        details = (f'Update ONE attribute of the stored {name} ({kind}) record using the CURRENT scene. '
                   'This is not a new diagnosis question. Read current.illnesses for the established disease. '
                   'A keep answer to the disease-course question does not mean its symptoms or treatment stayed unchanged. '
                   'The scene need not repeat the disease name to describe the sick subject\'s current fever or medicine. '
                   'Ignore historical-only/hypothetical mentions. If there is no new evidence choose keep, not unknown. ')
        result[key + '_severity'] = choice(details + 'Explicit current severity, not the severity of injuries. Fever alone does not upgrade disease severity.',
                                          {'keep': 'No explicit update', **SEVERITIES})
        result[key + '_treatment'] = choice(details + 'Doctor-prescribed medicine actually administered or taken for the current illness is treated; '
                                           'knowing the drug name or proving it works is not required. Mere prescription or a future dose is not administration. '
                                           'Wound dressing alone is not illness treatment. Explicit cessation/no treatment is untreated.',
                                           {'keep': 'No explicit update', **TREATMENTS})
        result[key + '_fever'] = choice(details + 'Current explicit fever (열이 오른다, a doctor confirms 열이 있다) is yes. '
                                       'Track observed fever in the illness episode, not proof of its exclusive medical cause. '
                                       'Explicitly fever-free is no. Generic warmth without fever evidence is keep.',
                                       {'keep': 'No explicit update', **FEVERS})
    return result


def settle(items, labels, source, scope_id):
    """Apply accepted labels at the scene endpoint; unknown answers preserve state."""
    result = deepcopy(items)
    changes = []
    for kind in KINDS:
        key = 'disease_' + kind
        action = labels.get(key, 'keep')
        old = next((i for i in result if i['kind'] == kind), None)
        if old is None and action not in {'active', 'worsening', 'recovering'}:
            continue
        if old is not None and old['status'] == 'resolved' and action not in {'active', 'worsening', 'recovering'}:
            continue
        record = deepcopy(old) if old else {
            'kind': kind, 'status': 'active', 'severity': 'unknown', 'treatment': 'unknown',
            'fever': 'unknown', 'elapsed_minutes': 0, 'source': source[:400], 'scope_id': str(scope_id),
        }
        if old and old['status'] == 'resolved':
            record.update(elapsed_minutes=0, severity='unknown', treatment='unknown', fever='unknown')
        if action != 'keep':
            record['status'] = action
        for field in ('severity', 'treatment', 'fever'):
            value = labels.get(key + '_' + field, 'keep')
            if value != 'keep':
                record[field] = value
        if record != old:
            record.update(source=source[:400], scope_id=str(scope_id))
            if old is not None:
                result.remove(old)
            result.append(record)
            changes.append({'kind': kind, 'status': record['status']})
    validate_illnesses(result)
    return result, changes


def rates(items):
    fatigue, clarity = 0.0, 0.0
    for item in items:
        if item['status'] == 'resolved':
            continue
        factor = .5 if item['status'] == 'recovering' else 1.0
        fatigue += {'unknown': 0, 'mild': .5, 'moderate': 1, 'severe': 2}[item['severity']] * factor
        if item['fever'] == 'yes':
            clarity -= factor
    return {'fatigue': fatigue, 'clarity': clarity}


def advance_illnesses(items, minutes):
    return [{**item, 'elapsed_minutes': item['elapsed_minutes'] + (minutes if item['status'] != 'resolved' else 0)}
            for item in items]


def actor_view(items):
    return [{'name': KINDS[i['kind']], 'course': STATUSES[i['status']],
             'severity': SEVERITIES[i['severity']], 'treatment': TREATMENTS[i['treatment']],
             'fever': FEVERS[i['fever']] if i['status'] != 'resolved' else '현재 효과 없음'} for i in items]


def display(items):
    views = actor_view(items)
    return '\n'.join(' · '.join(v.values()) for v in views) if views else '등록 없음 — 질병이 없다는 확정은 아님'
