"""Strict wire schema for author calls; durable partial-update semantics stay local."""
from copy import deepcopy

from jsonschema import Draft202012Validator
from tool_gateway.results import ToolRejection


def _nullable(schema):
    return Draft202012Validator(schema).is_valid(None)


def wire_schema(schema):
    result = deepcopy(schema)
    # These constraints are checked against the original schema after decoding.
    for key in ('minProperties', 'maxProperties', 'uniqueItems', 'default'):
        result.pop(key, None)
    if 'properties' in schema:
        required = set(schema.get('required', []))
        properties = {}
        for key, child in schema['properties'].items():
            converted = wire_schema(child)
            if key not in required:
                # A native null is a real value, so wrap that optional property
                # to distinguish {value:null} from null (= omitted update).
                if _nullable(child):
                    converted = {'type': 'object', 'properties': {'value': converted},
                                 'required': ['value'], 'additionalProperties': False}
                converted = {'anyOf': [converted, {'type': 'null'}],
                             'description': 'null means omit this update; keep the saved value.'}
            properties[key] = converted
        result.update(properties=properties, required=list(properties), additionalProperties=False)
    if 'items' in schema:
        result['items'] = wire_schema(schema['items'])
    for key in ('anyOf', 'oneOf', 'allOf'):
        if key in schema:
            result[key] = [wire_schema(child) for child in schema[key]]
    return result


def decode(value, schema):
    if isinstance(value, dict) and 'properties' in schema:
        required = set(schema.get('required', []))
        result = {}
        for key, item in value.items():
            child = schema.get('properties', {}).get(key, {})
            if key not in required and key in schema['properties']:
                if item is None:
                    continue
                if _nullable(child):
                    item = item['value']
            result[key] = decode(item, child)
        return result
    if isinstance(value, list):
        return [decode(item, schema.get('items', {})) for item in value]
    return value


def strict_tool(tool, handler):
    original = deepcopy(tool['input_schema'])
    wire = {**tool, 'input_schema': wire_schema(original), 'strict': True}

    wire_validator = Draft202012Validator(wire['input_schema'])
    original_validator = Draft202012Validator(original)

    def validate(validator, value):
        error = next(validator.iter_errors(value), None)
        if error is not None:
            path = '/' + '/'.join(str(p) for p in error.absolute_path)
            raise ToolRejection(f'{path}: {error.message[:300]}')

    async def accept(**value):
        # Validate both contracts even when invoked outside the tool gateway.
        validate(wire_validator, value)
        decoded = decode(value, original)
        validate(original_validator, decoded)
        return await handler(**decoded)

    return wire, accept
