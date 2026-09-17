"""Attempt-local prose repair; every completed patch passes the original schema."""
from copy import deepcopy
import re

from scripts.commulingo_write_session import draft_id, prepare_write, repair_schema
from tool_gateway.results import ToolRejection


class DraftRepair:
    def __init__(self, tool):
        self.name = tool['name']
        self.canonical = deepcopy(tool['input_schema'])
        self.draft = None
        self.overlength = {}
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
            guidance = self.length_guidance(message)
            if guidance:
                message += '\n' + guidance
        return message

    def length_guidance(self, message):
        """Give the author paragraph sizes so an over-length field is cut in one step.

        The model cannot count characters; without sizes it shaves a sentence per
        round and burns the whole stage (observed 12-round loops, 2026-09-17).
        """
        notes = []
        for excess, path in re.findall(r"(\d+) over the \d+-character limit at '([^']+)'", message):
            node = self.draft['args']
            try:
                for part in path.split('.'):
                    node = node[int(part)] if isinstance(node, list) else node[part]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if not isinstance(node, str):
                continue
            count = self.overlength[path] = self.overlength.get(path, 0) + 1
            paragraphs = [p for p in node.split('\n') if p.strip()]
            sizes = ', '.join(f'P{i} {len(p)}' for i, p in enumerate(paragraphs, 1))
            note = (f'{path}: remove at least {excess} characters in ONE repair. Paragraph sizes: {sizes}. '
                    'Drop or merge whole paragraphs or sentences of secondary detail and keep the supported core claims; '
                    'trimming a few words per attempt will not converge.')
            if count >= 3:
                note += f' This is over-length rejection #{count} for this field: cut well below the limit now.'
            notes.append(note)
        return '\n'.join(notes)
