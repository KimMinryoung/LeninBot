"""Normalize historical author checkpoints at the session boundary."""
from copy import deepcopy

from .author_draft import structured_args


def restore_draft(snapshot, field_schema, *, section=False):
    """Return an isolated working copy; journal history is never rewritten."""
    draft = deepcopy(snapshot)
    # Earlier editor checkpoints used sentence arrays for bilingual prose.
    # Joining preserves their text; canonical length checks still apply.
    if draft and structured_args(draft['args']):
        saved_args = draft['args']
        # Older checkpoints asked the author for a slug. The server now
        # owns it, so retain the draft prose and regenerate at validation.
        if section:
            saved_fields = saved_args.get('fields', {})
            saved_fields.pop('slug', None)
            # Older checkpoints carried the encoded key; the author field is the year.
            saved_order = saved_fields.pop('sortOrder', None)
            if isinstance(saved_order, int) and saved_order > 0 and 'startYear' not in saved_fields:
                saved_fields['startYear'] = saved_order // 100
                if 1 <= saved_order % 100 <= 12:
                    saved_fields['startMonth'] = saved_order % 100
            saved_args['claims'] = [c for c in saved_args.get('claims', [])
                                    if c.get('field') not in {'slug', 'sortOrder'}]
        nested_notes = saved_args.get('fields', {}).pop('notes', None)
        if isinstance(nested_notes, str):
            saved_args['notes'] = '\n\n'.join(dict.fromkeys(
                n.strip() for n in (saved_args.get('notes'), nested_notes) if n and n.strip()))
        for field, value in draft['args'].get('fields', {}).items():
            properties = field_schema['properties'].get(field, {}).get('properties', {})
            if isinstance(value, dict):
                for lang in ('ko','en'):
                    if properties.get(lang, {}).get('type')=='string' and isinstance(value.get(lang), list) and all(isinstance(s,str) for s in value[lang]):
                        value[lang] = ' '.join(value[lang])
    return draft
