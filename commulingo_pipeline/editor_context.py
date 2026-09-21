"""Author-visible working state and bounded research controls."""
import json
from tool_gateway.results import ToolRejection


def work_status(issues, draft, reads, *, error='', error_kind=''):
    """Describe actual runtime state without treating saved claims as approved."""
    args = (draft or {}).get('args', {})
    fields = args.get('fields')
    claims = args.get('claims')
    if error_kind == 'citation':
        action = ('Read the cited passages for the rejected claims with commulingo_pipeline_cached_passages. '
                  'Correct the claim to match the original, or research only its missing/conflicting fact. '
                  'Then repair the affected /claims/N and any dependent field text.')
        next_tool = 'commulingo_pipeline_cached_passages'
    elif error_kind == 'passages' or reads.missing_fields:
        action = ('Inspect current cached pages and labels; repair invalid references or add support for missing fields. '
                  'Fetch an original only if cached text is insufficient. Preserve other draft content.')
        next_tool = 'commulingo_pipeline_cached_passages'
    elif draft:
        action = ('Repair the saved draft at the reported JSON pointers using only repairs. '
                  'Do not resend unchanged fields or claims. If a fact needs investigation, follow research_access.')
        next_tool = 'commulingo_pipeline_repair'
    else:
        action = ('Read the target context and available originals for the commissioned issues. '
                  'Fetch relevant originals if the cache is empty or insufficient, then submit a full draft. '
                  'If no supported edit is warranted, submit a substantive no-edit decision.')
        next_tool = None
    return {
        'scope': [{'id':i['id'], 'field':i['field'], 'done_when':i['done_when']} for i in issues],
        'draft_saved': draft is not None,
        'saved_fields': sorted(fields) if isinstance(fields, dict) else [],
        'saved_claim_count': len(claims) if isinstance(claims, list) else 0,
        'saved_content_status': 'Retained for editing; saving does not establish validation or approval.',
        'mode': 'format_repair' if reads.repair_only else 'research_allowed',
        'research_access': ('Search/fetch are blocked. For a missing or conflicting fact, call '
                            'commulingo_pipeline_research with fields and reason first.'
                            if reads.repair_only else 'Search/fetch are allowed for specific missing or conflicting facts.'),
        'registry_lookups_remaining': max(0, 3-reads.lookups) if reads.repair_only else None,
        'missing_evidence_fields': list(reads.missing_fields),
        'error_kind': error_kind or None, 'last_error': error or None,
        'next_tool': next_tool, 'next_action': action,
        'submission_tool': 'commulingo_pipeline_repair' if draft else 'commulingo_pipeline_result',
    }


def prose_budgets(schema):
    return {field:{lang:{'draft_target':int(part['maxLength']*.8),
                         'hard_limit':part['maxLength']}
                   for lang,part in definition.get('properties',{}).items()
                   if lang in {'ko','en'} and part.get('maxLength')}
            for field,definition in schema['properties'].items()
            if any(p.get('maxLength') for lang,p in definition.get('properties',{}).items()
                   if lang in {'ko','en'})}


class RepairReads:
    """Keep format repair on its saved evidence; explicitly reopen factual research."""
    def __init__(self, sources, repair_only=False):
        self.sources = sources
        self.repair_only = repair_only
        self.lookups = 0
        self.missing_fields = []
        self.status = None

    def with_status(self, text):
        return text + ('\n' + json.dumps({'work_status': self.status()}, ensure_ascii=False)
                       if self.status else '')

    def wrap(self, name, call):
        fetched = self.sources.wrap(name,call)
        async def read(**args):
            if self.repair_only:
                allowed = {'get_person','get_term','get_sections','get_office','get_event'}
                if name!='commulingo_people' or args.get('action') not in allowed or self.lookups>=3:
                    raise ToolRejection(self.with_status('Format repair uses saved claims and cached passages. Repair the draft now. '
                        'If a fact is missing or conflicting, first call commulingo_pipeline_research with its fields and reason.'))
                self.lookups += 1
            return await fetched(**args)
        return read

    def tool(self, fields, usage, on_reopen=None):
        async def reopen(fields, reason):
            self.repair_only = False
            self.missing_fields = fields
            usage.tracker['targeted_research'] = fields
            if on_reopen:
                await on_reopen()
            return self.with_status('Research only these missing or conflicting facts: '+', '.join(fields)+
                    '. Preserve the saved draft and all other claims. Reason: '+reason)
        return ({'name':'commulingo_pipeline_research',
            'description':'Reopen research during format repair only for a specific missing or conflicting fact. No need to resubmit supported claims.',
            'input_schema':{'type':'object','additionalProperties':False,
                'properties':{'fields':{'type':'array','minItems':1,'maxItems':10,
                    'items':{'type':'string','enum':list(fields)}},
                    'reason':{'type':'string','minLength':20}},'required':['fields','reason']}},reopen,False)
