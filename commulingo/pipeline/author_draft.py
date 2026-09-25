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

SUBMIT_TOOL = 'commulingo_pipeline_submit_draft'
FIRST_SUBMISSION = ('The first submission must include changes, reason and a decision for every commissioned issue; '
                    'later submissions send only what changes.')
LEGACY_REPLACEMENT = ('The saved legacy draft is malformed and cannot be merged; send a complete replacement with '
                      'changes, reason and a decision for every commissioned issue. History is retained.')


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
        # Omitted evidence keeps the field's saved evidence: authors shortening
        # prose or fixing a date left it out and were rejected (2026-09-25 logs).
        changes = obj({name: obj({'value': intake_schema(schema),
                                 'evidence': {'type': 'array', 'items': evidence,
                                              'description': 'Omit to keep this field\'s saved evidence.'}},
                                 ['value'])
                       for name, schema in fields['properties'].items()})
        outcome = obj({'status': {'type': 'string', 'enum': ['resolved', 'deferred']},
                       'reason': {'type': 'string', 'minLength': 10}}, ['status', 'reason'])
        outcomes = obj({issue['id']: deepcopy(outcome) for issue in issues})
        self.issue_ids = list(outcomes['properties'])
        properties = {'changes': changes, 'issues': outcomes,
                      'reason': {'type': 'string', 'minLength': 20},
                      'notes': {'type': 'string', 'maxLength': 4000}}
        # The first submission must be complete; later calls to the same tool
        # replace only what they name. The server checks which case applies, so
        # the author never has to pick between a submit and a repair tool.
        full = obj(deepcopy(properties), ['changes', 'issues', 'reason'])
        full['properties']['changes']['minProperties'] = 1
        full['properties']['changes']['required'] = fields.get('required', [])
        full['properties']['issues']['required'] = self.issue_ids
        self.full_schema = full
        submit = obj(deepcopy(properties))
        submit['minProperties'] = 1
        saved = (self.draft or {}).get('args', {})
        removable = set(self.field_names)
        if structured_args(saved):
            removable.update(saved.get('fields', {}))
            removable.update(c['field'] for c in saved.get('claims', []))
        # Withdrawing a required field only produces a rejection; correct it instead.
        removable -= set(fields.get('required', []))
        if removable:
            submit['properties']['remove_fields'] = {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                'items': {'type': 'string', 'enum': sorted(removable)},
                'description': 'Withdraw optional fields from the saved draft, including their evidence. '
                               'Does not delete stored content.'}
        # Authors invented 'partial'/'blocked' and sent notes (2026-09-25 logs).
        no_edit = obj({'status': {'type': 'string', 'enum': ['complete', 'not_applicable', 'sources_unavailable'],
                                  'description': 'complete: current content already satisfies the commission; '
                                                 'not_applicable: the commission does not fit this target; '
                                                 'sources_unavailable: no accessible original supports an edit. '
                                                 'A partially supported edit is submitted, not a no-edit.'},
                       'reason': deepcopy(properties['reason']),
                       'issues': deepcopy(outcomes),
                       'notes': deepcopy(properties['notes'])}, ['status', 'reason', 'issues'])
        no_edit['properties']['issues']['required'] = self.issue_ids
        self.submit_tool = {'name': SUBMIT_TOOL, 'description':
            'Submit the edit for validation and independent review. Top-level keys are changes, issues, reason, '
            'notes and remove_fields; never put issues, reason or notes inside changes. Each change is '
            '{value, evidence}. Without a saved draft (work_status.draft_saved=false) send changes, reason and a '
            'decision for every commissioned issue. With a saved draft send only what changes: a supplied field '
            'replaces its whole value, and its evidence too when evidence is given (omit evidence to keep the saved '
            'evidence); omitted fields, issue decisions, reason and notes stay saved; arrays replace the whole list. '
            'All validations run again on the complete draft. Use commulingo_pipeline_no_edit to finish without edits.', 'input_schema': submit}
        self.no_edit_tool = {'name': 'commulingo_pipeline_no_edit', 'description':
            'Finish without a public edit. Explain the decision and each commissioned issue. '
            'Any saved draft remains in history; do not remove its fields first.', 'input_schema': no_edit}

    def validate_call(self, value, tool):
        errors = list(Draft202012Validator(tool['input_schema']).iter_errors(value))
        if errors:
            raise RepairProtocolError('; '.join(
                f"{'/'.join(str(p) for p in error.absolute_path) or 'arguments'}: {error.message[:300]}"
                for error in errors[:4]))

    def submission(self, value):
        self.validate_call(value, self.submit_tool)
        # A malformed legacy draft cannot be merged into, so it is replaced by a
        # complete submission. It stays in checkpoint history.
        update = bool(self.draft) and structured_args(self.draft['args'])
        if not update:
            required = (LEGACY_REPLACEMENT if self.draft else FIRST_SUBMISSION)
            if 'remove_fields' in value:
                raise RepairProtocolError('No mergeable saved draft to withdraw fields from. ' + required)
            try:
                self.validate_call(value, {'input_schema': self.full_schema})
            except RepairProtocolError as exc:
                raise RepairProtocolError(f'{exc}. {required}') from exc
        if update and not any(value.get(key) for key in ('changes', 'issues', 'remove_fields')) and not any(
                key in value for key in ('reason', 'notes')):
            raise RepairProtocolError('Supply changed fields, issue decisions, notes or a reason.')
        if set(value.get('changes', {})) & set(value.get('remove_fields', [])):
            raise RepairProtocolError('A field cannot be changed and withdrawn in the same call.')
        result = deepcopy(self.draft['args']) if update else {'fields': {}, 'claims': [], 'issue_results': []}
        # Legacy no-edit checkpoints may contain a draft. Explicit edit submission
        # restores ready; the independent no-edit operation never mutates it.
        result['status'] = 'ready'
        replaced = ({field for field, change in value.get('changes', {}).items() if 'evidence' in change}
                    | set(value.get('remove_fields', [])))
        result['claims'] = [c for c in result.get('claims', []) if c.get('field') not in replaced]
        for field in value.get('remove_fields', []):
            result.setdefault('fields', {}).pop(field, None)
        for field, change in value.get('changes', {}).items():
            result.setdefault('fields', {})[field] = deepcopy(change['value'])
            result['claims'].extend({'field': field, **deepcopy(e)} for e in change.get('evidence', []))
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
                    'instruction': f'Replace this malformed legacy draft with a complete {SUBMIT_TOOL} call, '
                                   'or use commulingo_pipeline_no_edit to finish without edits.'}
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
        if not self.draft or f'"submission_tool": "{SUBMIT_TOOL}"' in message:
            return message
        errors = []
        for error in Draft202012Validator(self.canonical).iter_errors(self.draft['args']):
            path = '/' + '/'.join(str(p) for p in error.absolute_path)
            errors.append({'path': self.author_error(path), 'rule': error.validator,
                           'message': self.author_error(error.message[:300])})
        return message + '\n' + json.dumps({'errors': errors[:12],
            'submission_tool': SUBMIT_TOOL,
            'instruction': 'Resend each affected change with its complete value and evidence. Omitted changes remain saved. '
                           'Use commulingo_pipeline_no_edit to finish without edits.'}, ensure_ascii=False)
