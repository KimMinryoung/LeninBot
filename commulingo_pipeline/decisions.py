"""Jev owns closed-set editorial decisions; authors supply supported prose."""
import asyncio
from copy import deepcopy
from .patches import canonical


class Decisions:
    def __init__(self, job, current, catalogs, usage, cache=None):
        self.job, self.current, self.catalogs, self.usage = job,current or {},catalogs,usage
        self.cache = cache if cache is not None else {}
        self.citations = {}

    def detailed(self, *args, **kwargs):
        from llm.call_registry import decide_detailed
        result = decide_detailed(*args, **kwargs)
        decision = result.decision
        if decision is not None:
            cost = decision.cost_usd or 0
            tracker = self.usage.tracker
            tracker['jev_cost_usd'] = tracker.get('jev_cost_usd',0)+cost
            tracker['jev_calls'] = tracker.get('jev_calls',0)+1
        return result

    async def decide(self, *args, **kwargs):
        return (await asyncio.to_thread(self.detailed,*args,**kwargs)).decision

    def author_schema(self, schema):
        out = deepcopy(schema)
        assigned = {'group','groupId','role'} if self.job['kind']=='person' else {'category'}
        for field in assigned:
            out['properties'].pop(field,None)
        out['required'] = [f for f in out.get('required',[]) if f not in assigned]
        for field, code in (('citizenship','code'),('nationalOrigin','code'),('fate','kind')):
            obj = out['properties'].get(field)
            if obj:
                obj.get('properties',{}).pop(code,None)
                obj['required'] = [k for k in obj.get('required',[]) if k!=code]
        return out

    def strip_assigned(self, fields):
        for key in ('group','groupId','role') if self.job['kind']=='person' else ('category',):
            fields.pop(key,None)
        for field,key in (('citizenship','code'),('nationalOrigin','code'),('fate','kind')):
            if isinstance(fields.get(field),dict):
                fields[field].pop(key,None)

    async def classify(self, fields, claims, sources):
        from runtime_tools import commulingo_classify as c
        excerpts = {}
        for claim in claims:
            page = sources[claim['source_id']]['body']
            excerpts.setdefault(claim['field'],[]).append({'claim':claim['claim'],
                'excerpt':page[claim['start']:claim['end']][:1500]})
        merged = {**self.current, **fields}
        kind = self.job['kind']
        person = kind=='person' and (self.job['action']=='create' or
            not self.current.get('groupId') or not self.current.get('role'))
        codes = kind=='person' and any(k in fields for k in ('citizenship','nationalOrigin','fate'))
        term = kind=='term' and self.job['action']=='create'
        if not (person or codes or term):
            return fields, {}
        state = c.term_state(merged) if term else c.person_card_state(merged,excerpts)
        key = canonical({'state':state,'person':person,'codes':codes,'term':term})
        verdict = self.cache.get(key)
        if verdict is None:
            if term:
                verdict = await asyncio.to_thread(c.classify_term,merged,decide=self.detailed)
            elif person:
                verdict = await asyncio.to_thread(c.classify_person_card,merged,catalogs=self.catalogs,
                                                 claims=excerpts,decide=self.detailed)
            else:
                verdict = await asyncio.to_thread(c.classify_person_codes,merged,claims=excerpts,decide=self.detailed)
            if verdict is None or (person and verdict.get('person') is None):
                raise RuntimeError('Jev classification unavailable; saved draft retained for retry, no LLM fallback')
            self.cache[key] = verdict
        out = deepcopy(fields)
        if term:
            out = c.fill_term_category(out,verdict)
        else:
            out = c.fill_person_codes(out,verdict.get('codes') if person else verdict)
            if c.missing_person_codes(out):
                raise RuntimeError('Jev returned incomplete codes; saved draft retained for retry')
            if person:
                classified = c.fill_classification(out,verdict['person'])
                for key in ('groupId','role'):
                    if self.job['action']=='create' or not self.current.get(key):
                        out[key] = classified[key]
        return out, verdict
