"""Small writing aids retained from the legacy editor."""
from tool_gateway.results import ToolRejection


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

    def wrap(self, name, call):
        fetched = self.sources.wrap(name,call)
        async def read(**args):
            if self.repair_only:
                allowed = {'get_person','get_term','get_sections','get_office','get_event'}
                if name!='commulingo_people' or args.get('action') not in allowed or self.lookups>=3:
                    raise ToolRejection('Format repair uses saved claims and cached passages. Repair the draft now. '
                        'If a fact is missing or conflicting, first call commulingo_pipeline_research with its fields and reason.')
                self.lookups += 1
            return await fetched(**args)
        return read

    def tool(self, fields, usage):
        async def reopen(fields, reason):
            self.repair_only = False
            self.missing_fields = fields
            usage.tracker['targeted_research'] = fields
            return ('Research only these missing or conflicting facts: '+', '.join(fields)+
                    '. Preserve the saved draft and all other claims. Reason: '+reason)
        return ({'name':'commulingo_pipeline_research',
            'description':'Reopen research during format repair only for a specific missing or conflicting fact. No need to resubmit supported claims.',
            'input_schema':{'type':'object','additionalProperties':False,
                'properties':{'fields':{'type':'array','minItems':1,'maxItems':10,
                    'items':{'type':'string','enum':list(fields)}},
                    'reason':{'type':'string','minLength':20}},'required':['fields','reason']}},reopen,False)
