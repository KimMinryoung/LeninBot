"""Attempt-local prose repair; every completed patch passes the original schema."""
from copy import deepcopy
import re

from commulingo.pipeline.write_session import draft_id, prepare_write, repair_schema
from tool_gateway.results import ToolRejection
from jsonschema import Draft202012Validator
import json


class RepairProtocolError(ValueError):
    """A malformed edit request is not evidence of editorial stagnation."""


class DraftRepair:
    def __init__(self, tool, *, capture_invalid=False):
        self.name = tool['name']
        self.canonical = deepcopy(tool['input_schema'])
        self.draft = None
        self.overlength = {}
        self.separate_tools = False
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
        self.capture_invalid = capture_invalid
        if capture_invalid:
            # This tool only records a private draft. Let malformed nested data
            # reach prepare_write so it can be retained and repaired; publication
            # still requires the unchanged canonical schema and store validation.
            self.tool['input_schema'] = {
                'type':'object', 'additionalProperties':False,
                'properties':{key: {} for key in self.canonical['properties']},
            }
            self.tool['input_schema']['properties'].update({
                'draft_id':{'type':'string'},
                'repairs':repair_schema(self.canonical)['properties']['repairs'],
            })
            self.tool['description'] = (self.tool.get('description','') +
                ' Saves a private draft before validation. Follow the draft_contract in the prompt. '
                'After rejection send only repairs, never repeat the entire draft.')

    def prepare(self, value):
        if self.capture_invalid and 'repairs' not in value:
            # A full draft sometimes echoes its prior ID. It is not a repair.
            value = {k:v for k,v in value.items() if k!='draft_id'}
        if not value:
            # An empty/undecodable call is not a replacement editorial draft.
            # Preserve the scratchpad and keep it out of stagnation counting.
            instruction = ('Use commulingo_pipeline_repair with only repairs.'
                           if self.separate_tools and self.draft else
                           'Resend the intended arguments as valid JSON.')
            raise RepairProtocolError(self.feedback(
                'Empty arguments do not replace the saved draft. ' + instruction))
        try:
            prepared = prepare_write(self.name,value,self.draft,schema=self.canonical)
            self.draft = {'tool':self.name, 'args':deepcopy(prepared)}
            return prepared
        except ToolRejection as exc:
            candidate = getattr(exc,'canonical_args',None)
            if candidate is not None:
                self.draft = {'tool':self.name,'args':deepcopy(candidate)}
            error = ValueError if candidate is not None else RepairProtocolError
            raise error(self.feedback(str(exc))) from exc

    def feedback(self, message):
        if self.draft and 'Saved draft_id=' not in message and '"submission_tool": "commulingo_pipeline_repair"' not in message:
            if self.separate_tools:
                errors = []
                for error in Draft202012Validator(self.canonical).iter_errors(self.draft['args']):
                    path = '/' + '/'.join(str(p).replace('~','~0').replace('/','~1') for p in error.absolute_path)
                    if error.validator=='additionalProperties' and isinstance(error.instance,dict):
                        for key in error.instance.keys()-error.schema.get('properties',{}).keys():
                            errors.append({'path':path+'/'+key.replace('~','~0').replace('/','~1'),
                                           'rule':'additionalProperties','repair':'remove'})
                        continue
                    errors.append({'path':path,'rule':error.validator,'expected':error.validator_value,
                                   'current':str(error.instance)[:240],'message':error.message[:350]})
                return message + '\n' + json.dumps({'draft_id':draft_id(self.draft),
                    'errors':errors[:12], 'submission_tool':'commulingo_pipeline_repair',
                    'instruction':'Send only repairs. The draft and all untouched claims are retained.'},ensure_ascii=False)
            example = '/fields/bio/ko' if self.capture_invalid else '/fields/bio/ko/2'
            message += (f'\nSaved draft_id={draft_id(self.draft)}. Send only repairs (JSON pointers such as '
                f'{example}) to replace or remove the rejected parts; unchanged fields remain saved. '
                'Retain supported claims and obey final field limits.')
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
