"""Target-scoped commissions, retaining the individual editorial topics."""
from collections import OrderedDict


def topics(job):
    payload = job.get('payload') or {}
    return payload.get('remaining_topics', payload.get('topics', [job['topic']]))


def work_topics(job):
    pending = topics(job)
    # A section has its own save/review contract. Process it after the card,
    # from a new research snapshot, within the same durable job.
    if job['kind'] == 'person' and len(pending) > 1:
        return [topic for topic in pending if topic != 'sections']
    return pending


def advance(job, value):
    if not (job.get('payload') or {}).get('topics'):
        return value
    remaining = [topic for topic in topics(job) if topic not in work_topics(job)]
    return {**value, 'remaining_topics': remaining} if remaining else value


def gap_ids(payload):
    return list(dict.fromkeys([*payload.get('gap_ids', []),
                              *([payload['gap_id']] if payload.get('gap_id') else [])]))


def bundle_candidates(rows):
    grouped = OrderedDict()
    for row in sorted(rows, key=lambda r: r['priority']):
        key = (row['kind'], row['target']) if row['action'] == 'update' else (
            row['kind'], row['target'], row['topic'])
        grouped.setdefault(key, []).append(row)
    result = []
    for group in grouped.values():
        first = group[0]
        if first['action'] != 'update':
            result.append(first)
            continue
        commissioned = list(dict.fromkeys(t for row in group for t in topics(row)))
        payload = {**(first.get('payload') or {}), 'topics': commissioned,
                   'remaining_topics': commissioned,
                   'gap_ids': list(dict.fromkeys(g for row in group
                                               for g in gap_ids(row.get('payload') or {}))),
                   'commissions': [{'topic':row['topic'], 'reason':row['reason'],
                                    'baseline':row['baseline'], 'payload':row.get('payload') or {}}
                                   for row in group]}
        result.append({**first, 'topic':'enrichment', 'payload':payload,
                       'baseline':next((r['baseline'] for r in group if r['baseline']), ''),
                       'reason':'Bundled enrichment: ' + ', '.join(commissioned)})
    return result
