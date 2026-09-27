"""Flat author input with partial saves over the durable fields/claims contract."""
from copy import deepcopy
import json
import re

from jsonschema import Draft202012Validator
from tool_gateway.validation import (register_argument_shape_repair,
    register_empty_arguments_hint, register_malformed_arguments_hint,
    validate_tool_arguments, ToolArgumentValidationError)

from .draft_repair import RepairProtocolError
from .evidence import MAX_PASSAGES, PASSAGE_PATTERN

SUBMIT_TOOL = 'commulingo_pipeline_submit_draft'


def bilingual_field(schema):
    properties = schema.get('properties') or {}
    return (schema.get('type') == 'object'
            and {'ko', 'en'} <= set(schema.get('required') or [])
            and all((properties.get(lang) or {}).get('type') == 'string'
                    for lang in ('ko', 'en')))


# English renderings of Korean prose run 1.79-2.68x the Korean character count
# across 2,764 published sections (1st-99th percentile, 2026-09-27). Far
# outside that band one language is untranslated or truncated: sections went
# live with en "", "placeholder" or "REPLACE_EN" beside a full Korean body,
# and others dropped whole Korean paragraphs from the English.
PARITY_MIN_CHARS = 300
PARITY_RATIO = (1.4, 3.4)


def bilingual_gaps(field, value):
    """Missing-list entries for a bilingual value: absent or blank languages,
    then a long text whose languages are out of proportion."""
    texts = {lang: value.get(lang) for lang in ('ko', 'en')}
    gaps = [f'fields.{field}.{lang}' for lang, text in texts.items()
            if not (isinstance(text, str) and text.strip())]
    if gaps:
        return gaps
    ko, en = len(texts['ko'].strip()), len(texts['en'].strip())
    if max(ko, en) < PARITY_MIN_CHARS:
        return []
    low, high = PARITY_RATIO
    if en < ko * low:
        return [f'fields.{field}.en (only {en} characters beside {ko} Korean; English normally runs '
                f'about 2x the Korean length, so render every Korean claim in English)']
    if en > ko * high:
        return [f'fields.{field}.ko (only {ko} characters beside {en} English; Korean normally runs '
                f'about half the English length, so write every English claim in Korean)']
    return []


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


def unwrap_arguments(args, schema):
    """Accept the provider's redundant arguments envelope, never guess field placement."""
    if set(args) == {'arguments'}:
        wrapped = args['arguments']
        if isinstance(wrapped, str):
            try:
                wrapped = json.loads(wrapped)
            except ValueError:
                return args, []
        if isinstance(wrapped, dict):
            return wrapped, ['unwrapped arguments object']
    return args, []


def structured_args(args):
    """Older permissive intake could save malformed containers; never drop them."""
    return (isinstance(args.get('fields', {}), dict)
            and isinstance(args.get('claims', []), list)
            and all(isinstance(c, dict) and isinstance(c.get('field'), str) for c in args.get('claims', []))
            and isinstance(args.get('issue_results', []), list)
            and all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in args.get('issue_results', [])))


class AuthorDraft:
    """Own the author contract and saved draft, independently of direct-write repairs."""

    name = 'commulingo_pipeline_result'  # Durable checkpoint identity.

    def __init__(self, fields, issues, *, editable_fields=None, factual_fields=(), draft=None):
        self.draft = deepcopy(draft)
        self.canonical_fields = deepcopy(fields)
        self._configure(fields if editable_fields is None else editable_fields, issues,
                       factual_fields=factual_fields)

    def save(self, value):
        self.draft = {'tool': self.name, 'args': deepcopy(value)}

    def prepare(self, value):
        """Retain rejected prose for corrections; only return canonical valid drafts."""
        if not value:
            raise RepairProtocolError('Empty arguments do not replace the saved draft.')
        candidate = deepcopy(value)
        try:
            prepared = validate_tool_arguments(self.name, candidate,
                                               schema=self.canonical, risk_class='write')
        except ToolArgumentValidationError as exc:
            self.save(candidate)
            raise ValueError(self.feedback(str(exc))) from exc
        self.save(prepared)
        return prepared

    def _configure(self, fields, issues, *, factual_fields=()):
        self.field_names = list(fields['properties'])
        self.required_fields = list(fields.get('required', []))
        self.factual_fields = set(factual_fields)
        self.bilingual_fields = {name for name, schema in fields['properties'].items()
                                 if bilingual_field(schema)}
        values = obj({name: intake_schema(schema) for name, schema in fields['properties'].items()})
        for name in self.bilingual_fields:
            values['properties'][name].pop('required', None)
            values['properties'][name]['minProperties'] = 1
        claim = obj({'claim': {'type': 'string', 'minLength': 1},
                     'passages': {'type': 'array', 'minItems': 1, 'maxItems': MAX_PASSAGES,
                                  'items': {'type': 'string', 'pattern': PASSAGE_PATTERN}},
                     'stance': {'type': 'string', 'enum': ['supports', 'disputes']}},
                    ['claim', 'passages'])
        evidence = obj({name: {'type': 'array', 'items': deepcopy(claim)} for name in self.field_names})
        evidence['description'] = 'Evidence by field. Each supplied list replaces that field\'s evidence; omitted fields keep theirs.'
        outcome = obj({'status': {'type': 'string', 'enum': ['resolved', 'deferred']},
                       'reason': {'type': 'string', 'minLength': 10}}, ['status', 'reason'])
        self.issue_ids = [issue['id'] for issue in issues]
        outcomes = obj({key: deepcopy(outcome) for key in self.issue_ids})
        properties = {'fields': values, 'evidence': evidence, 'issues': outcomes,
                      'reason': {'type': 'string', 'minLength': 20},
                      'notes': {'type': 'string', 'maxLength': 4000}}
        # One definition for claim, outcome and metadata constraints; the
        # public intake allows partial prose while the durable form validates it.
        durable_claim = obj({'field': {'type': 'string', 'enum': list(self.canonical_fields['properties'])},
                             **deepcopy(claim['properties'])}, ['field', *claim['required']])
        durable_outcome = obj({'id': {'type': 'string', 'enum': self.issue_ids},
                               **deepcopy(outcome['properties'])}, ['id', *outcome['required']])
        self.canonical = obj({
            'status': {'type': 'string', 'enum': ['ready', 'complete', 'not_applicable', 'sources_unavailable']},
            'reason': deepcopy(properties['reason']), 'fields': deepcopy(self.canonical_fields),
            'claims': {'type': 'array', 'items': durable_claim},
            'issue_results': {'type': 'array', 'items': durable_outcome},
            'notes': deepcopy(properties['notes'])}, ['status', 'reason'])
        submit = obj(properties)
        submit['minProperties'] = 1
        saved = (self.draft or {}).get('args', {})
        removable = set(self.field_names)
        if structured_args(saved):
            removable.update(saved.get('fields', {}))
            removable.update(c['field'] for c in saved.get('claims', []))
        removable -= set(self.required_fields)
        if removable:
            submit['properties']['remove_fields'] = {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                'items': {'type': 'string', 'enum': sorted(removable)},
                'description': 'Withdraw optional fields and their evidence from this draft only.'}
        no_edit = obj({'status': {'type': 'string', 'enum': ['complete', 'not_applicable', 'sources_unavailable'],
                                  'description': 'complete=already satisfied; not_applicable=does not fit; '
                                                 'sources_unavailable=no accessible original.'},
                       'reason': deepcopy(properties['reason']), 'issues': deepcopy(outcomes),
                       'notes': deepcopy(properties['notes'])}, ['status', 'reason', 'issues'])
        no_edit['properties']['issues']['required'] = self.issue_ids
        self.submit_tool = {'name': SUBMIT_TOOL, 'description':
            'Submit fields, evidence, issue decisions and a reason for independent review. '
            'Send a complete small edit in one call. Partial calls are also saved: omitted fields, '
            'languages and evidence are kept. To correct evidence, send only evidence.<field>. '
            'Arrays replace their whole list. Validation starts when required values, factual evidence '
            'and all issue decisions are present. Use commulingo_pipeline_no_edit if no edit is warranted.',
            'input_schema': submit}
        self.no_edit_tool = {'name': 'commulingo_pipeline_no_edit', 'description':
            'Finish without a public edit. Explain the decision and each commissioned issue. '
            'Any saved draft remains in history.', 'input_schema': no_edit}

    def validate_call(self, value, tool):
        errors = list(Draft202012Validator(tool['input_schema']).iter_errors(value))
        if errors:
            raise RepairProtocolError('; '.join(
                f"{'/'.join(str(p) for p in error.absolute_path) or 'arguments'}: {error.message[:300]}"
                for error in errors[:4]))

    def submission(self, value):
        self.validate_call(value, self.submit_tool)
        update = bool(self.draft) and structured_args(self.draft['args'])
        if not update and 'remove_fields' in value:
            raise RepairProtocolError('No saved draft to withdraw fields from.')
        if not any(value.get(key) for key in ('fields', 'evidence', 'issues', 'remove_fields')) and not any(
                key in value for key in ('reason', 'notes')):
            raise RepairProtocolError('Supply fields, evidence, issue decisions, notes or a reason.')
        removed = set(value.get('remove_fields', []))
        if (set(value.get('fields', {})) | set(value.get('evidence', {}))) & removed:
            raise RepairProtocolError('A field cannot be changed and withdrawn in the same call.')
        result = deepcopy(self.draft['args']) if update else {'fields': {}, 'claims': [], 'issue_results': []}
        result['status'] = 'ready'
        saved_fields = result.setdefault('fields', {})
        for field in removed:
            saved_fields.pop(field, None)
        for field, incoming in deepcopy(value.get('fields', {})).items():
            previous = saved_fields.get(field)
            saved_fields[field] = ({**previous, **incoming}
                if field in self.bilingual_fields and isinstance(previous, dict) and isinstance(incoming, dict)
                else incoming)
        replaced = set(value.get('evidence', {})) | removed
        result['claims'] = [c for c in result.get('claims', []) if c.get('field') not in replaced]
        for field, claims in value.get('evidence', {}).items():
            if field not in saved_fields:
                raise RepairProtocolError(f'evidence.{field} needs a saved or supplied field value.')
            result['claims'].extend({'field': field, **deepcopy(claim)} for claim in claims)
        outcomes = {item['id']: item for item in result.get('issue_results', [])}
        outcomes.update({key: {'id': key, **deepcopy(item)} for key, item in value.get('issues', {}).items()})
        result['issue_results'] = list(outcomes.values())
        for key in ('reason', 'notes'):
            if key in value:
                result[key] = value[key]
        return result

    def missing(self, result, *, factual_fields=None):
        fields = result.get('fields') or {}
        decided = {item['id'] for item in result.get('issue_results') or []}
        missing = [f'fields.{field}' for field in self.required_fields if field not in fields]
        for field in sorted(self.bilingual_fields & set(fields)):
            if isinstance(fields[field], dict):
                missing.extend(bilingual_gaps(field, fields[field]))
        if not fields and not self.required_fields:
            missing.append('fields (at least one field)')
        evidenced = {claim['field'] for claim in result.get('claims') or []}
        factual = self.factual_fields if factual_fields is None else set(factual_fields)
        missing += [f'evidence.{field}' for field in sorted((set(fields) & factual) - evidenced)]
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
                    'instruction': f'Replace this malformed legacy draft through {SUBMIT_TOOL}, '
                                   'or use commulingo_pipeline_no_edit. History is retained.'}
        claims = args.get('claims') or []
        return {'fields': deepcopy(args.get('fields') or {}),
                'evidence': {field: [{k: v for k, v in c.items() if k != 'field'}
                                    for c in claims if c['field'] == field]
                             for field in dict.fromkeys(c['field'] for c in claims)},
                'issues': {item['id']: {key: val for key, val in item.items() if key != 'id'}
                           for item in args.get('issue_results', [])},
                'reason': args.get('reason', ''), 'notes': args.get('notes', '')}

    def author_error(self, message):
        args = (self.draft or {}).get('args', {})
        claims = (args.get('claims') or []) if structured_args(args) else []
        def claim_path(match):
            index = int(match[1])
            if index >= len(claims):
                return '/evidence'
            field = claims[index]['field']
            local = sum(c['field'] == field for c in claims[:index])
            return f'/evidence/{field}/{local}'
        message = re.sub(r'/claims/(\d+)', claim_path, message)
        return message.replace('/issue_results', '/issues').replace('issue_results', 'issues')

    def feedback(self, message):
        message = self.author_error(message)
        if not self.draft or 'Saved draft; send only' in message:
            return message
        return message + ('\nSaved draft; send only the rejected fields or evidence through ' + SUBMIT_TOOL
                          + '. Omitted values, languages and evidence remain saved.')


register_argument_shape_repair(SUBMIT_TOOL, unwrap_arguments)
register_empty_arguments_hint(SUBMIT_TOOL, (
    'Arguments arrived empty. Do not resend the same call. Send the draft in parts: '
    'fields for values, evidence for supporting passages. Parts are saved and merged.'))
register_malformed_arguments_hint(SUBMIT_TOOL, (
    'Use valid JSON: escape double quotes and newlines inside strings; close every string, array and object. '
    'Do not resend the same long JSON. Send one language in fields per call; send evidence separately. '
    'Valid parts are saved and merged.'))
