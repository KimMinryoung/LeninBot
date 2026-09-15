"""Attempt-local prose repair; every completed patch passes the original schema."""
from copy import deepcopy

from scripts.commulingo_write_session import draft_id, prepare_write, repair_schema
from tool_gateway.results import ToolRejection


class DraftRepair:
    def __init__(self, tool):
        self.name = tool['name']
        self.canonical = deepcopy(tool['input_schema'])
        self.draft = None
        self.tool = deepcopy(tool)
        self.tool['input_schema'] = repair_schema(self.canonical)
        # Let overlength prose reach the local scratchpad so a rejected full
        # patch can be repaired by ID. The canonical limits remain mandatory.
        def intake(node):
            if isinstance(node,dict):
                if 'maxLength' in node:
                    limit = node['maxLength']
                    node['description'] = node.get('description','') + f' Final hard limit: {limit} characters.'
                    del node['maxLength']
                for child in node.values():
                    intake(child)
            elif isinstance(node,list):
                for child in node:
                    intake(child)
        intake(self.tool['input_schema'])

    def prepare(self, value):
        try:
            return prepare_write(self.name,value,self.draft,schema=self.canonical)
        except ToolRejection as exc:
            candidate = getattr(exc,'canonical_args',None)
            if candidate is not None:
                self.draft = {'tool':self.name,'args':deepcopy(candidate)}
            message = str(exc)
            if self.draft:
                message += (f'\nSaved draft_id={draft_id(self.draft)}. Call the same tool with only draft_id and '
                    'repairs, e.g. [{"op":"set","path":"/fields/bio/en","value":"shorter text"}]. '
                    'Replace only the rejected fields; do not resend unchanged content. '
                    'Cut an optional clause or sentence and leave margin below the hard limit.')
            raise ValueError(message) from exc
