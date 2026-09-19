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
    """Attempt-local short names, one per URL.

    A page fetched in several offsets used to be several snapshots with
    separate handles and chunk numbering that restarted at 0 on each; the
    model then cited page 2's chunk numbers under page 1's handle. 68 of the
    71 jobs with chunk-ID rejections in the week to 2026-09-19 had such a
    URL. Now a URL has one handle whose current snapshot is the merged text
    of every page fetched (see SourcePages); older snapshots stay in the
    job's source map so carried-over claims still compile, and any
    persistent ID remains accepted.
    """
    def __init__(self, sources):
        self.ids = {}      # handle -> current source id
        self.by_url = {}   # url -> handle
        for source in sorted(sources.values(), key=lambda s: (s['url'], str(s.get('fetched_at') or ''))):
            self.register(source)

    def register(self, source):
        handle = self.by_url.setdefault(source['url'], f'S{len(self.by_url)+1}')
        self.ids[handle] = source['id']
        return handle

    def handle(self, source):
        return self.register(source)

    def resolve(self, claims, sources):
        result = []
        for claim in claims:
            source_id = self.ids.get(claim.get('source_id'), claim.get('source_id'))
            source = sources.get(source_id)
            if not source or not source.get('body'):
                available = ', '.join(f'{handle}: chunks 0..{(len(sources[sid]["body"])-1)//SOURCE_CHUNK_CHARS}'
                    for handle,sid in self.ids.items() if sources.get(sid,{}).get('body'))
                raise ValueError('unknown source_id; retrieve or use an available source: ' + available)
            result.append({**claim, 'source_id':source_id})
        return result


class SourcePages:
    """One growing snapshot per URL within a research attempt.

    absorb(url, page) returns the merged snapshot and the character span the
    page occupies in it, so a paginated fetch is displayed as chunks a..b of
    one continuous numbering. Pages are identified by content hash, never by
    text search, so re-fetching the same offset changes nothing.
    """
    def __init__(self):
        self.current = {}   # url -> merged snapshot
        self.spans = {}     # url -> {page hash: (start, end)}

    def seed(self, sources):
        """Merge the snapshots a job already holds for a URL, oldest first."""
        merged = []
        by_url = {}
        for source in sorted((s for s in sources.values() if s.get('body')),
                             key=lambda s: str(s.get('fetched_at') or '')):
            by_url.setdefault(source['url'], []).append(source)
        for url, pages in by_url.items():
            if len(pages) == 1:
                self.current[url] = pages[0]
                self.spans[url] = {pages[0]['content_hash']: (0, len(pages[0]['body']))}
                continue
            for page in pages:
                snapshot_, span, created = self.absorb(url, page['body'])
            merged.append(self.current[url])
        return merged

    def absorb(self, url, body):
        body = body.replace('\x00', '\ufffd')
        digest = hashlib.sha256(body.encode()).hexdigest()
        current = self.current.get(url)
        spans = self.spans.setdefault(url, {})
        if current is None:
            merged = snapshot(url, body)
            spans[digest] = (0, len(body))
        elif digest in spans:
            return current, spans[digest], False
        else:
            offset = len(current['body']) + 1
            merged = snapshot(url, current['body'] + '\n' + body)
            spans[digest] = (offset, offset + len(body))
        self.current[url] = merged
        return merged, spans[digest], True
