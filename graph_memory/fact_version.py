"""Canonical relation content shared by mirror comparison and version IDs."""
import json
from datetime import datetime, timezone

# Provenance and bookkeeping do not change the assertion's meaning.
OPERATIONAL_ATTRIBUTES = frozenset({
    'sync_key', 'doc_ref', 'source_url', 'doc_kind', 'extraction',
    'created_at', 'updated_at', 'extracted_at', 'content_sha256',
    'version_hash', 'source_active', 'verification_status', 'withdrawn',
})


def canonical_date(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def relation_content(predicate, fact, valid_at=None, invalid_at=None, attributes=None):
    from graph_memory.graphiti_patches import normalize_entity_names_in_text
    attrs = {k: v for k, v in (attributes or {}).items()
             if k not in OPERATIONAL_ATTRIBUTES and v is not None}
    return json.dumps([predicate, normalize_entity_names_in_text(fact or ''),
                       canonical_date(valid_at), canonical_date(invalid_at), attrs],
                      sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)
