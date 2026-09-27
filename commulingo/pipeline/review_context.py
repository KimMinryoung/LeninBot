"""Lossless patch input with old values once and on-demand unchanged context."""
from copy import deepcopy
from tool_gateway.validation import register_argument_shape_repair
from .patches import changes


def repair_context_fields(args, schema):
    """Map section leaf requests to the current sections container."""
    fields = args.get('fields')
    allowed = set(schema.get('properties', {}).get('fields', {}).get('items', {}).get('enum', []))
    if not isinstance(fields, list) or 'sections' not in allowed:
        return args, []
    repaired = []
    changed = False
    for field in fields:
        if field in {'body', 'heading', 'slug', 'sortOrder'} and field not in allowed:
            field = 'sections'
            changed = True
        if field not in repaired:
            repaired.append(field)
    if not changed:
        return args, []
    return {**args, 'fields': repaired}, ['mapped section leaf fields to sections']


def context(proposal, current, previous_patch=None):
    current = current or {}
    fields = proposal['patch_json']
    before = current
    if proposal['target_type'] == 'person_section':
        before = next((s for s in current.get('sections', []) if s.get('slug') == fields.get('slug')), {})
    delta = changes(before, fields)
    identity = {k:current[k] for k in ('id','name','term','revision','notes') if k in current}
    # Every proposed value and all evidence stay in suggestion.patch_json.
    # Changed old leaves occur once here; unchanged fields can be requested.
    result = {'suggestion':proposal, 'identity':identity,
              'changes':[{'path':'/fields'+r['path'],'before':r['before']} for r in delta],
              'available_current_fields': sorted(current)}
    if previous_patch is not None:
        result['changes_since_previous_patch'] = [
            {'path':'/fields'+r['path'],'before':r['before']}
            for r in changes(previous_patch, fields)]
    return result


def context_tool(current):
    current = current or {}
    available = sorted(current)
    async def read(fields):
        if not 1 <= len(fields) <= len(available) or len(set(fields)) != len(fields) or any(f not in current for f in fields):
            raise ValueError('Request distinct fields from available_current_fields: '
                             + ', '.join(available))
        from .stages import stage_evidence
        return stage_evidence({f:deepcopy(current[f]) for f in fields})
    return ({'name':'commulingo_pipeline_review_context',
             'description':'Read unchanged current entry fields, including notes/sections, when needed to check contradictions or duplicate sections. This is current entry data, not independent source evidence.',
             'input_schema':{'type':'object','additionalProperties':False,
                 'properties':{'fields':{'type':'array','minItems':1,'maxItems':len(available),
                     'uniqueItems':True,'items':{'type':'string','enum':available}}},
                 'required':['fields']}}, read, False)


register_argument_shape_repair('commulingo_pipeline_review_context', repair_context_fields)
