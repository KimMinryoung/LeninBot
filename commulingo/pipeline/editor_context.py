"""Compact author feedback; research stays within the shared stage budget."""
from .author_draft import SUBMIT_TOOL


def work_status(issues, draft, *, error='', error_kind=''):
    args = (draft or {}).get('args', {})
    fields = args.get('fields')
    if error_kind in {'citation', 'passages', 'missing_evidence'}:
        action = ('Check cached passages; correct evidence.<field>, or research the specific unsupported fact. '
                  'Resend fields only if their text also needs correction.')
        next_tool = 'commulingo_pipeline_cached_passages'
    elif draft:
        action = ('Send only corrected fields or evidence. Reuse saved sources for formatting repairs; '
                  'research only missing or conflicting facts.')
        next_tool = SUBMIT_TOOL
    else:
        action = 'Read relevant originals, then submit the supported edit or a reasoned no-edit decision.'
        next_tool = None
    return {
        'scope': [{'id':i['id'], 'field':i['field'], 'done_when':i['done_when']} for i in issues],
        'draft_saved': draft is not None,
        'saved_fields': sorted(fields) if isinstance(fields, dict) else [],
        'error_kind': error_kind or None, 'last_error': error or None,
        'next_tool': next_tool, 'next_action': action,
    }


def prose_budgets(schema):
    return {field:{lang:{'draft_target':int(part['maxLength']*.8),
                         'hard_limit':part['maxLength']}
                   for lang,part in definition.get('properties',{}).items()
                   if lang in {'ko','en'} and part.get('maxLength')}
            for field,definition in schema['properties'].items()
            if any(p.get('maxLength') for lang,p in definition.get('properties',{}).items()
                   if lang in {'ko','en'})}
