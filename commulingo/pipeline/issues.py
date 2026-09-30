"""Concrete editorial defects: empty values and explicit requests. Quotas and missing provenance never commission new prose."""
from .bundles import work_topics

PERSON = {'basics': ('years', 'epithet', 'activities', 'career'), 'bio': ('bio',),
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
        if job['kind']=='person' and topic=='sections':
            if not current.get('sections'):
                issues.append({'id':'missing:sections', 'field':'body', 'topic':topic,
                    'problem':'No detail section exists for this person.',
                    'done_when':('Add one distinct bilingual detail section built from documented actions, decisions, '
                        'dates, figures and outcomes the card does not already give, with body evidence. Read at '
                        'least two independent sources before writing; rest the section mainly on one source only '
                        'when that source is substantial and the others add nothing. If the material is one thin '
                        'source (a short encyclopedia entry, an obituary, a catalogue note) or only restates the '
                        'bio, finish with the no-edit tool and say why.')})
            continue
        fields = (PERSON if job['kind']=='person' else TERM).get(topic, ())
        for field in fields:
            value = current.get(field)
            missing = not value
            if isinstance(value, dict):
                text = value.get('label', value)
                missing = any(not str(text.get(lang) or '').strip() for lang in ('ko','en')) if isinstance(text,dict) else not text
            # Existing prose without recorded evidence is not an issue: attaching
            # sources turned into rewrites of reviewed text (2026-09-20).
            if missing:
                issues.append({'id':f'missing:{field}', 'field':field, 'topic':topic,
                    'problem':'Missing value or language.', 'done_when':'Supply the missing supported value/language, preserving existing facts.'})
    payload = job.get('payload') or {}
    automatic = ('Commissioned ', 'Bundled enrichment:')
    # consolidate() rewrites a bundled job's reason to "Bundled enrichment: ..." and keeps
    # each original reason under payload.commissions; an operator request bundled that way
    # was read as automatic and closed with nothing to do (job 63870, 2026-09-29).
    requests = [c['reason'] for c in payload.get('commissions', [])
                if isinstance(c, dict) and str(c.get('reason') or '') and not str(c['reason']).startswith(automatic)]
    explicit = (payload.get('gap_id') or payload.get('gap_ids') or payload.get('review_feedback')
                or requests or not job['reason'].startswith(automatic))
    if explicit:
        issues.append({'id':'requested', 'field':'*', 'problem':'\n'.join(requests) or job['reason'],
                       'done_when':'Address the explicit request with supported minimal changes, or explain why no change is warranted.'})
    return list({issue['id']: issue for issue in issues}.values())
