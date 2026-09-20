"""Concrete editorial defects: empty values and explicit requests. Quotas and missing provenance never commission new prose."""
from .bundles import work_topics

PERSON = {'basics': ('years', 'epithet', 'role', 'career'), 'bio': ('bio',),
          'nationality': ('citizenship', 'nationalOrigin'), 'moment': ('moment',)}
TERM = {'definition': ('definition',), 'history': ('body',)}
FACTS = {'person': {'bio','years','moment','citizenship','nationalOrigin'},
         'term': {'definition','body','period','startYear','endYear'}}


def commission(job, current):
    """Each issue has a stable ID, affected field and an observable finish condition."""
    if job['action'] == 'create':
        return [{'id':'register', 'field':'*', 'problem':job['reason'],
                 'done_when':'A supported bilingual entry identifies the requested person or concept.'}]
    current = current or {}
    issues = []
    for topic in work_topics(job):
        fields = (PERSON if job['kind']=='person' else TERM).get(topic, ())
        for field in fields:
            value = current.get(field)
            missing = not value
            if isinstance(value, dict) and field not in {'role'}:
                text = value.get('label', value)
                missing = any(not str(text.get(lang) or '').strip() for lang in ('ko','en')) if isinstance(text,dict) else not text
            # Existing prose without recorded evidence is not an issue: attaching
            # sources turned into rewrites of reviewed text (2026-09-20).
            if missing:
                issues.append({'id':f'missing:{field}', 'field':field, 'topic':topic,
                    'problem':'Missing value or language.', 'done_when':'Supply the missing supported value/language, preserving existing facts.'})
    payload = job.get('payload') or {}
    explicit = payload.get('gap_id') or payload.get('gap_ids') or payload.get('review_feedback') or not job['reason'].startswith(('Commissioned ', 'Bundled enrichment:'))
    if explicit:
        issues.append({'id':'requested', 'field':'*', 'problem':job['reason'],
                       'done_when':'Address the explicit request with supported minimal changes, or explain why no change is warranted.'})
    return list({issue['id']: issue for issue in issues}.values())
