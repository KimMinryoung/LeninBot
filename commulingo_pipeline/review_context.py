"""Lossless patch input with old values once and on-demand unchanged context."""
from copy import deepcopy
from .patches import changes


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
              'changes':[{'path':r['path'],'before':r['before']} for r in delta],
              'available_current_fields': sorted(current)}
    if previous_patch is not None:
        result['changes_since_previous_patch'] = [
            {'path':r['path'],'before':r['before']}
            for r in changes(previous_patch, fields)]
    return result


def context_tool(current):
    current = current or {}
    async def read(fields):
        if not 1 <= len(fields) <= 12 or any(f not in current for f in fields):
            raise ValueError('Request 1..12 fields from available_current_fields')
        from .stages import stage_evidence
        return stage_evidence({f:deepcopy(current[f]) for f in fields})
    return ({'name':'commulingo_pipeline_review_context',
             'description':'Read unchanged current entry fields, including notes/sections, when needed to check contradictions or duplicate sections. This is current entry data, not independent source evidence.',
             'input_schema':{'type':'object','additionalProperties':False,
                 'properties':{'fields':{'type':'array','minItems':1,'maxItems':12,
                     'items':{'type':'string'}}}, 'required':['fields']}}, read, False)
