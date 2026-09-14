"""Explicit, reviewed name/type decisions; never infer identity from name alone."""
import json
from functools import lru_cache
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'kg_identity_review.json'


@lru_cache(maxsize=1)
def reviewed_targets():
    from kg_runtime.identity import normalize_alias_key
    data = json.loads(CONFIG.read_text())
    if data.get('version') != 1:
        raise ValueError('Unsupported KG identity review version')
    targets = {}
    for entry in data['entries']:
        for name in {entry['original_name'], entry['name']}:
            for label in entry['types']:
                key = (normalize_alias_key(name), label)
                if key in targets and targets[key]['target_uuid'] != entry['target_uuid']:
                    raise ValueError(f'Conflicting KG identity review: {key}')
                targets[key] = entry
    return targets


def reviewed_target(name, entity_type, *, external_id=None):
    # A source record's external ID remains authoritative, even for a namesake.
    if external_id:
        return None
    from kg_runtime.identity import normalize_alias_key
    return reviewed_targets().get((normalize_alias_key(name), entity_type))


CYPHER_REVIEWED_TARGET = '''MATCH (n:Entity {uuid: $uuid})
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels'''


def checked_result(record, entry):
    from kg_runtime.identity import ResolveResult
    if not record or not set(entry['target_types']).issubset(record['labels']):
        raise RuntimeError(f"Reviewed KG target missing or changed: {entry['target_uuid']}")
    return ResolveResult(record['uuid'], 'reviewed', record['name'], list(record['labels']))
