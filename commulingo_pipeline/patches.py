"""Canonical patch identity, actual changes, and deterministic write schemas."""
import hashlib
import json
from copy import deepcopy


def canonical(value):
    # ASCII JSON is deliberately shared with the JS publication boundary.
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def patch_hash(request):
    bound = {k: request[k] for k in ('target','action','id','fields','sources')}
    return hashlib.sha256(canonical(bound).encode()).hexdigest()


def changes(current, fields):
    """Only changed leaves, including new evidence; no whole unchanged biography."""
    result = []
    def walk(before, after, path):
        if isinstance(before,dict) and isinstance(after,dict):
            for key, value in after.items():
                walk(before.get(key), value, path+'/'+key.replace('~','~0').replace('/','~1'))
        elif before != after:
            result.append({'path':path, 'before':before, 'after':after})
    for key, value in fields.items():
        if key not in {'expectedRevision','evidence','sources','reviewFlags'}:
            walk((current or {}).get(key), value, '/'+key)
    return result


def schema_for(job, current, catalogs=None):
    from runtime_tools.commulingo_people import (
        _COMMULINGO_FIELD_SCHEMA, COMMULINGO_PERSON_CREATE_TOOL, COMMULINGO_PERSON_UPDATE_TOOL,
        COMMULINGO_TERM_CREATE_TOOL, COMMULINGO_TERM_UPDATE_TOOL, COMMULINGO_SECTION_SAVE_TOOL,
    )
    from .bundles import work_topics
    source = {('person','create'):COMMULINGO_PERSON_CREATE_TOOL, ('person','update'):COMMULINGO_PERSON_UPDATE_TOOL,
              ('term','create'):COMMULINGO_TERM_CREATE_TOOL, ('term','update'):COMMULINGO_TERM_UPDATE_TOOL}[job['kind'],job['action']]
    schema = deepcopy(source['input_schema']['properties']['fields'])
    section = job['kind']=='person' and job['action']=='update' and work_topics(job)==['sections']
    if section:
        props = COMMULINGO_SECTION_SAVE_TOOL['input_schema']['properties']
        return {'type':'object','additionalProperties':False,
                'properties':{**{k:deepcopy(props[k]) for k in ('slug','heading','body')},
                              'sortOrder':{'type':'integer'}}, 'required':['slug','heading','body']}
    canonical_fields = _COMMULINGO_FIELD_SCHEMA['properties']
    # The author chooses from real closed sets; no hidden classifier rewrites a
    # reviewed fact or prevents a format repair when a second provider is down.
    if job['kind']=='person':
        for field in ('citizenship','nationalOrigin','fate','groupId','role','activities'):
            schema['properties'][field] = deepcopy(canonical_fields[field])
        groups, offices, roles = catalogs or ([], [], [])
        if not groups or not roles:
            raise ValueError('classification catalogs unavailable')
        schema['properties']['groupId']['enum'] = [row['id'] for row in groups]
        for key in ('category','categoryId'):
            schema['properties']['role']['properties'][key]['enum'] = [row['id'] for row in roles]
        if offices:
            schema['properties']['role']['properties']['officeId']['enum'] = [row['id'] for row in offices]
        if job['action']=='create':
            schema['required'] = [*schema.get('required', []), 'groupId', 'role']
            schema.setdefault('allOf', []).append({'anyOf':[{'required':['givenName']},{'required':['familyName']}]})
        else:
            if (current or {}).get('groupId'):
                schema['properties'].pop('groupId',None)
            if (current or {}).get('role'):
                schema['properties'].pop('role',None)
    elif job['action']=='create':
        schema['properties']['category'] = deepcopy(canonical_fields['category'])
        schema['required'] = [*schema.get('required', []), 'category']
    for key in ('evidence','sources','expectedRevision','confidence','reviewFlags'):
        schema['properties'].pop(key,None)
    schema['required'] = [k for k in schema.get('required',[]) if k in schema['properties']]
    if 'sortOrder' in schema['properties']:
        schema['properties']['sortOrder']['type'] = 'integer'
    for full, edits in (('aliases','aliasEdits'),('career','careerEdits'),('scenes','sceneEdits')):
        if full in schema['properties'] and edits in schema['properties']:
            schema.setdefault('allOf', []).append({'not':{'required':[full,edits]}})
    return schema
