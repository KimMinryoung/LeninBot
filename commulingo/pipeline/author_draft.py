"""Typed author API over the durable, canonical editorial draft.

The publication contract and historical checkpoints keep fields/claims. Only
this adapter handles the author-facing field/value/evidence representation.
"""
from copy import deepcopy
import json
import re

from jsonschema import Draft202012Validator

from .draft_repair import DraftRepair, RepairProtocolError
from .evidence import MAX_PASSAGES, PASSAGE_PATTERN


def obj(properties, required=()):
    return {'type': 'object', 'additionalProperties': False,
            'properties': properties, **({'required': list(required)} if required else {})}


def intake_schema(schema):
    """Keep types visible; overlength text reaches the saved-draft validator."""
    result = deepcopy(schema)
    if isinstance(result, dict):
        limit = result.pop('maxLength', None)
        if limit is not None:
            result['description'] = result.get('description', '') + f' Final limit: {limit} characters.'
        return {key: intake_schema(value) for key, value in result.items()}
    if isinstance(result, list):
        return [intake_schema(value) for value in result]
    return result


def structured_args(args):
    """Older permissive intake could save malformed containers; never drop them."""
    return (isinstance(args.get('fields', {}), dict)
            and isinstance(args.get('claims', []), list)
            and all(isinstance(c, dict) and isinstance(c.get('field'), str) for c in args.get('claims', []))
            and isinstance(args.get('issue_results', []), list)
            and all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in args.get('issue_results', [])))


class AuthorDraft(DraftRepair):
    def configure(self, fields, issues):
        self.field_names = list(fields['properties'])
        evidence = obj({'claim': {'type': 'string', 'minLength': 1},
                        'passages': {'type': 'array', 'minItems': 1, 'maxItems': MAX_PASSAGES,
                                     'items': {'type': 'string', 'pattern': PASSAGE_PATTERN}},
                        'stance': {'type': 'string', 'enum': ['supports', 'disputes']}},
                       ['claim', 'passages'])
        changes = obj({name: obj({'value': intake_schema(schema),
                                 'evidence': {'type': 'array', 'items': evidence}}, ['value', 'evidence'])
                       for name, schema in fields['properties'].items()})
        outcome = obj({'status': {'type': 'string', 'enum': ['resolved', 'deferred']},
                       'reason': {'type': 'string', 'minLength': 10}}, ['status', 'reason'])
        outcomes = obj({issue['id']: deepcopy(outcome) for issue in issues})
        self.issue_ids = list(outcomes['properties'])
        properties = {'changes': changes, 'issues': outcomes,
                      'reason': {'type': 'string', 'minLength': 20},
                      'notes': {'type': 'string', 'maxLength': 4000}}
        submit = obj(deepcopy(properties), ['changes', 'issues', 'reason'])
        submit['properties']['changes']['minProperties'] = 1
        submit['properties']['changes']['required'] = fields.get('required', [])
        submit['properties']['issues']['required'] = self.issue_ids
        update = obj(deepcopy(properties))
        update['minProperties'] = 1
        saved = (self.draft or {}).get('args', {})
        removable = set(self.field_names)
        if structured_args(saved):
            removable.update(saved.get('fields', {}))
            removable.update(c['field'] for c in saved.get('claims', []))
        update['properties']['remove_fields'] = {'type': 'array', 'minItems': 1, 'uniqueItems': True,
            'items': {'type': 'string', 'enum': sorted(removable)},
            'description': 'Withdraw fields from this draft, including their evidence. Does not delete stored content.'}
        no_edit = obj({'status': {'type': 'string', 'enum': ['complete', 'not_applicable', 'sources_unavailable']},
                       'reason': deepcopy(properties['reason']),
                       'issues': deepcopy(outcomes)}, ['status', 'reason', 'issues'])
        no_edit['properties']['issues']['required'] = self.issue_ids
        self.submit_tool = {'name': self.name, 'description':
            'Submit a complete private edit. Each change contains its typed value and evidence. '
            'Report each commissioned issue explicitly. Use repair for later changes or no_edit to finish without edits.',
            'input_schema': submit}
        self.update_tool = {'name': 'commulingo_pipeline_repair', 'description':
            'Revise saved work with the same changes structure. Each supplied field replaces its whole draft value '
            'AND evidence; omitted fields and issue decisions remain saved. Arrays replace the whole list. '
            'Use remove_fields to withdraw a draft field. All validations run again.', 'input_schema': update}
        self.no_edit_tool = {'name': 'commulingo_pipeline_no_edit', 'description':
            'Finish without a public edit. Explain the decision and each commissioned issue. '
            'Any saved draft remains in history; do not remove its fields first.', 'input_schema': no_edit}

    def validate_call(self, value, tool):
        errors = list(Draft202012Validator(tool['input_schema']).iter_errors(value))
        if errors:
            raise RepairProtocolError('; '.join(
                f"{'/'.join(str(p) for p in error.absolute_path) or 'arguments'}: {error.message[:300]}"
                for error in errors[:4]))

    def submission(self, value, *, update=False):
        self.validate_call(value, self.update_tool if update else self.submit_tool)
        if update and not self.draft:
            raise RepairProtocolError('No saved draft. Submit a complete edit first.')
        if update and not structured_args(self.draft['args']):
            raise RepairProtocolError('Legacy draft has malformed containers. Use commulingo_pipeline_result '
                                      'to submit a complete replacement, or commulingo_pipeline_no_edit. History is retained.')
        if update and not any(value.get(key) for key in ('changes', 'issues', 'remove_fields')) and not any(
                key in value for key in ('reason', 'notes')):
            raise RepairProtocolError('Supply changed fields, issue decisions, notes or a reason.')
        if set(value.get('changes', {})) & set(value.get('remove_fields', [])):
            raise RepairProtocolError('A field cannot be changed and withdrawn in the same call.')
        result = deepcopy(self.draft['args']) if update else {'fields': {}, 'claims': [], 'issue_results': []}
        # Legacy no-edit checkpoints may contain a draft. Explicit edit submission
        # restores ready; the independent no-edit operation never mutates it.
        result['status'] = 'ready'
        replaced = set(value.get('changes', {})) | set(value.get('remove_fields', []))
        result['claims'] = [c for c in result.get('claims', []) if c.get('field') not in replaced]
        for field in value.get('remove_fields', []):
            result.setdefault('fields', {}).pop(field, None)
        for field, change in value.get('changes', {}).items():
            result.setdefault('fields', {})[field] = deepcopy(change['value'])
            result['claims'].extend({'field': field, **deepcopy(e)} for e in change['evidence'])
        outcomes = {item['id']: item for item in result.get('issue_results', [])}
        outcomes.update({key: {'id': key, **deepcopy(item)} for key, item in value.get('issues', {}).items()})
        result['issue_results'] = list(outcomes.values())
        for key in ('reason', 'notes'):
            if key in value:
                result[key] = value[key]
        return result

    def view(self):
        if not self.draft:
            return None
        args = self.draft['args']
        if not structured_args(args):
            return {'needs_full_submission': True, 'legacy_draft': deepcopy(args),
                    'instruction': 'Use commulingo_pipeline_result to replace this malformed legacy draft, '
                                   'or commulingo_pipeline_no_edit to finish without edits.'}
        claims = args.get('claims') or []
        return {'changes': {field: {'value': value, 'evidence': [
                    {key: val for key, val in c.items() if key != 'field'}
                    for c in claims if c.get('field') == field]}
                for field, value in (args.get('fields') or {}).items()},
                'issues': {item['id']: {key: val for key, val in item.items() if key != 'id'}
                           for item in args.get('issue_results', [])},
                'reason': args.get('reason', ''), 'notes': args.get('notes', '')}

    def author_error(self, message):
        args = (self.draft or {}).get('args', {})
        claims = (args.get('claims') or []) if structured_args(args) else []
        def claim_path(match):
            index = int(match[1])
            if index >= len(claims):
                return 'changes evidence'
            field = claims[index].get('field', '')
            local = sum(c.get('field') == field for c in claims[:index])
            return f'/changes/{field}/evidence/{local}'
        message = re.sub(r'/claims/(\d+)', claim_path, message)
        message = re.sub(r'/fields/([^/\s"\']+)', r'/changes/\1/value', message)
        message = re.sub(r"\bfields\.([A-Za-z]+)", r'changes.\1.value', message)
        return message.replace('/issue_results', '/issues').replace('issue_results', 'issues')

    def feedback(self, message):
        message = self.author_error(message)
        if not self.draft or '"submission_tool": "commulingo_pipeline_repair"' in message:
            return message
        errors = []
        for error in Draft202012Validator(self.canonical).iter_errors(self.draft['args']):
            path = '/' + '/'.join(str(p) for p in error.absolute_path)
            errors.append({'path': self.author_error(path), 'rule': error.validator,
                           'message': self.author_error(error.message[:300])})
        return message + '\n' + json.dumps({'errors': errors[:12],
            'submission_tool': 'commulingo_pipeline_repair',
            'instruction': 'Resend each affected change with its complete value and evidence. Omitted changes remain saved. '
                           'Use commulingo_pipeline_no_edit to finish without edits.'}, ensure_ascii=False)
