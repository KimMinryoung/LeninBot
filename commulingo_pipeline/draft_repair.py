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
            prepared = prepare_write(self.name,value,self.draft,schema=self.canonical)
            self.draft = {'tool':self.name, 'args':deepcopy(prepared)}
            return prepared
        except ToolRejection as exc:
            candidate = getattr(exc,'canonical_args',None)
            if candidate is not None:
                self.draft = {'tool':self.name,'args':deepcopy(candidate)}
            raise ValueError(self.feedback(str(exc))) from exc

    def feedback(self, message):
        if self.draft and 'Saved draft_id=' not in message:
            message += (f'\nSaved draft_id={draft_id(self.draft)}. Use this current ID and repairs only. '
                'Replace rejected fields using JSON pointers; unchanged fields remain saved. '
                'A stale ID was not applied. Retain supported claims and obey final field limits.')
        return message
