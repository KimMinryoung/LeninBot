"""Content-addressed source snapshots and exact, bounded claim citations."""
import hashlib
from datetime import datetime, timedelta, timezone

SOURCE_CHUNK_CHARS = 240


def resolve_claim_chunks(claims, sources):
    """Translate displayed chunk IDs to exact offsets; never ask a model to count."""
    resolved = []
    for claim in claims:
        source = sources.get(claim.get('source_id'))
        if not source or not source.get('body'):
            raise ValueError('unknown source_id; use an ID from a retrieved source')
        chunks = claim.get('chunks', [claim['chunk']] if 'chunk' in claim else [])
        count = (len(source['body']) + SOURCE_CHUNK_CHARS - 1) // SOURCE_CHUNK_CHARS
        if not chunks or any(type(n) is not int or n < 0 or n >= count for n in chunks):
            raise ValueError(f'Use displayed chunk IDs in 0..{count-1} for source {source["id"]}')
        groups = []
        for chunk in sorted(set(chunks)):
            if groups and chunk == groups[-1][-1] + 1:
                groups[-1].append(chunk)
            else:
                groups.append([chunk])
        for group in groups:
            value = {k:v for k,v in claim.items() if k not in {'chunks','chunk'}}
            value.update(start=group[0]*SOURCE_CHUNK_CHARS,
                         end=min(len(source['body']), (group[-1]+1)*SOURCE_CHUNK_CHARS))
            resolved.append(value)
    if len(resolved) > 50:
        raise ValueError(f'Expanded evidence has {len(resolved)} claims; at most 50 allowed. Select fewer contiguous ranges while retaining support for every required field.')
    return resolved


def snapshot(url, body, now=None):
    if not isinstance(body, str) or not body.strip():
        raise ValueError('source body is empty')
    # PostgreSQL text cannot store NUL. Normalize before hashing and assigning
    # chunk offsets so the saved text and the displayed evidence stay identical.
    body = body.replace('\x00', '\ufffd')
    now = now or datetime.now(timezone.utc)
    digest = hashlib.sha256(body.encode()).hexdigest()
    source_id = hashlib.sha256((url + '\0' + digest).encode()).hexdigest()
    return {'id': source_id, 'url': url, 'content_hash': digest, 'body': body,
            'fetched_at': now, 'expires_at': now + timedelta(days=14)}


def compile_evidence(claims, sources, changed_fields):
    """The model selects offsets; it cannot supply a fabricated excerpt."""
    result = []
    now = datetime.now(timezone.utc)
    for claim in claims:
        source = sources.get(claim.get('source_id'))
        if not source or not source.get('body') or source['expires_at'] <= now:
            raise ValueError('missing or expired source; retrieve it again')
        start, end = claim.get('start'), claim.get('end')
        body = source['body']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(body):
            raise ValueError('invalid source character range')
        if not 20 <= end-start <= 6000:
            raise ValueError('source range must contain 20..6000 characters')
        if claim.get('field') not in changed_fields or not str(claim.get('claim', '')).strip():
            raise ValueError('claim must name a changed field and explain its support')
        stance = claim.get('stance', 'supports')
        if stance not in {'supports', 'disputes'}:
            raise ValueError('invalid evidence stance')
        result.append({'field': claim['field'], 'claim': claim['claim'],
            'source': source['url'], 'locator': f'characters {start}:{end}; sha256 {source["content_hash"]}',
            'excerpt': body[start:end], 'stance': stance})
    return result


class SourceHandles:
    """Attempt-local short names; persisted artifacts always use content IDs."""
    def __init__(self, sources):
        self.ids = {}
        for source_id in sorted(sources):
            self.handle(source_id)

    def handle(self, source_id):
        if source_id not in self.ids.values():
            self.ids[f'S{len(self.ids)+1}'] = source_id
        return next(k for k,v in self.ids.items() if v == source_id)

    def resolve(self, claims, sources):
        result = []
        for claim in claims:
            source_id = self.ids.get(claim.get('source_id'), claim.get('source_id'))
            source = sources.get(source_id)
            if not source or not source.get('body'):
                available = ', '.join(f'{self.handle(k)}: chunks 0..{(len(v["body"])-1)//SOURCE_CHUNK_CHARS}'
                    for k,v in sources.items() if v.get('body'))
                raise ValueError('unknown source_id; retrieve or use an available source: ' + available)
            result.append({**claim, 'source_id':source_id})
        return result
