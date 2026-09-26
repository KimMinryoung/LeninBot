"""Typed author API over the durable, canonical editorial draft.

The publication contract and historical checkpoints keep fields/claims. Only
this adapter handles the author-facing field/value/evidence representation.
"""
from copy import deepcopy
import json
import re

from jsonschema import Draft202012Validator
from tool_gateway.validation import (register_argument_shape_repair,
    register_empty_arguments_hint, register_malformed_arguments_hint)

from .draft_repair import DraftRepair, RepairProtocolError
from .evidence import MAX_PASSAGES, PASSAGE_PATTERN

SUBMIT_TOOL = 'commulingo_pipeline_submit_draft'


def bilingual_field(schema):
    """A prose object whose two language values may be staged separately."""
    properties = schema.get('properties') or {}
    return (schema.get('type') == 'object'
            and {'ko', 'en'} <= set(schema.get('required') or [])
            and all((properties.get(lang) or {}).get('type') == 'string'
                    for lang in ('ko', 'en')))


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


def repair_submission_shape(args, schema):
    """Fix unambiguous nesting mistakes before schema validation.

    Editor logs (2026-09-22..25) showed issues/reason/notes sent inside changes,
    fields sent without the changes wrapper, evidence inside value and bare
    values without {value}. Each cost a rejected round although the intent was
    clear. Only moves that cannot change meaning are made; anything else is
    left for the validator to report.
    """
    props = schema.get('properties') or {}
    field_schemas = ((props.get('changes') or {}).get('properties')) or {}
    if not field_schemas:
        return args, []
    args, notes = dict(args), []
    top_keys = set(props) - {'changes'}
    loose = [key for key in args if key in field_schemas and key not in props]
    if loose and 'changes' not in args:
        args['changes'] = {key: args.pop(key) for key in loose}
        notes.append('wrapped top-level fields in changes: ' + ', '.join(loose))
    changes = args.get('changes')
    if not isinstance(changes, dict):
        return args, notes
    changes = dict(changes)
    for key in [k for k in changes if k in top_keys and k not in field_schemas]:
        if key in args:
            continue
        args[key] = changes.pop(key)
        notes.append(f'moved {key} out of changes')
    for field, change in list(changes.items()):
        value_schema = ((field_schemas.get(field) or {}).get('properties') or {}).get('value') or {}
        object_value = value_schema.get('type') == 'object' or 'properties' in value_schema
        if not isinstance(change, dict) or not ({'value', 'evidence'} & set(change)):
            changes[field] = {'value': change}
            notes.append(f'wrapped changes.{field} in value')
            continue
        if 'value' not in change and object_value and set(change) - {'evidence'}:
            changes[field] = {'value': {k: v for k, v in change.items() if k != 'evidence'},
                              **({'evidence': change['evidence']} if 'evidence' in change else {})}
            notes.append(f'moved changes.{field} content into value')
            continue
        value = change.get('value')
        if (isinstance(value, dict) and 'evidence' in value and 'evidence' not in change
                and 'evidence' not in (value_schema.get('properties') or {})):
            inner = dict(value)
            changes[field] = {**change, 'value': inner, 'evidence': inner.pop('evidence')}
            notes.append(f'moved changes.{field}.value.evidence up one level')
    args['changes'] = changes
    return args, notes


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
        self.bilingual_fields = {name for name, schema in fields['properties'].items()
                                 if bilingual_field(schema)}
        evidence = obj({'claim': {'type': 'string', 'minLength': 1},
                        'passages': {'type': 'array', 'minItems': 1, 'maxItems': MAX_PASSAGES,
                                     'items': {'type': 'string', 'pattern': PASSAGE_PATTERN}},
                        'stance': {'type': 'string', 'enum': ['supports', 'disputes']}},
                       ['claim', 'passages'])
        # Omitted evidence keeps the field's saved evidence: authors shortening
        # prose or fixing a date left it out and were rejected (2026-09-25 logs).
        def author_value(name, schema):
            value = intake_schema(schema)
            if name in self.bilingual_fields:
                # Canonical required languages remain in self.full_schema.
                # Only the author intake accepts a single language per call.
                value.pop('required', None)
                value['minProperties'] = 1
            return value
        changes = obj({name: obj({'value': author_value(name, schema),
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
                                  'description': 'complete=already satisfied; not_applicable=does not fit this '
                                                 'target; sources_unavailable=no accessible original. Submit a '
                                                 'partial edit instead.'},
                       'reason': deepcopy(properties['reason']),
                       'issues': deepcopy(outcomes),
                       'notes': deepcopy(properties['notes'])}, ['status', 'reason', 'issues'])
        no_edit['properties']['issues']['required'] = self.issue_ids
        self.submit_tool = {'name': SUBMIT_TOOL, 'description':
            'Submit the edit for review. Each change is {value, evidence}. Top-level keys only: changes, issues, '
            'reason, notes, remove_fields. Calls merge into one saved draft, so a large edit may be sent a few '
            'fields per call; bilingual prose may send ko and en in separate calls. It is validated once it '
            'holds both languages of each changed prose field, required fields, reason and every issue. Omit '
            'evidence to keep saved evidence. Arrays replace the whole list. No edit: commulingo_pipeline_no_edit.', 'input_schema': submit}
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
        if not update and 'remove_fields' in value:
            raise RepairProtocolError('No saved draft to withdraw fields from.')
        if not any(value.get(key) for key in ('changes', 'issues', 'remove_fields')) and not any(
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
            previous = result.setdefault('fields', {}).get(field)
            incoming = deepcopy(change['value'])
            if field in self.bilingual_fields and isinstance(previous, dict) and isinstance(incoming, dict):
                result['fields'][field] = {**previous, **incoming}
            else:
                result['fields'][field] = incoming
            result['claims'].extend({'field': field, **deepcopy(e)} for e in change.get('evidence', []))
        outcomes = {item['id']: item for item in result.get('issue_results', [])}
        outcomes.update({key: {'id': key, **deepcopy(item)} for key, item in value.get('issues', {}).items()})
        result['issue_results'] = list(outcomes.values())
        for key in ('reason', 'notes'):
            if key in value:
                result[key] = value[key]
        return result

    def missing(self, result):
        """What a merged draft still lacks before full validation can run.

        Large edits may arrive over several calls; a draft is validated only
        once it holds the required fields, every issue decision and a reason.
        """
        fields = result.get('fields') or {}
        required = self.full_schema['properties']['changes'].get('required', [])
        decided = {item.get('id') for item in result.get('issue_results') or []}
        missing = [f'changes.{field}' for field in required if field not in fields]
        for field in sorted(self.bilingual_fields & set(fields)):
            if isinstance(fields[field], dict):
                missing.extend(f'changes.{field}.value.{lang}' for lang in ('ko', 'en')
                               if lang not in fields[field])
        if not fields and not required:
            missing.append('changes (at least one field)')
        missing += [f'issues.{issue}' for issue in self.issue_ids if issue not in decided]
        if not result.get('reason'):
            missing.append('reason')
        return missing

    def view(self):
        if not self.draft:
            return None
        args = self.draft['args']
        if not structured_args(args):
            return {'needs_full_submission': True, 'legacy_draft': deepcopy(args),
                    'instruction': f'Replace this malformed legacy draft through {SUBMIT_TOOL} '
                                   '(it may take several calls), or use commulingo_pipeline_no_edit.'}
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


register_argument_shape_repair(SUBMIT_TOOL, repair_submission_shape)
# DeepSeek drops long tool arguments and delivers {} (2,600-2,900 output
# tokens each, 2026-09-25); resending the same call fails the same way.
register_empty_arguments_hint(SUBMIT_TOOL, (
    'Arguments arrived empty: the provider dropped this call because its arguments were too long. '
    'Do not resend the same call. Send the draft in parts; for bilingual prose send ko and en '
    'in separate calls. Send evidence, issues and reason in small calls. Parts are saved and merged.'))
register_malformed_arguments_hint(SUBMIT_TOOL, (
    'Do not resend the same long JSON. Send one language of a bilingual prose field per call; '
    'send its evidence, issue decisions and reason in small separate calls. Valid parts are saved and merged.'))
