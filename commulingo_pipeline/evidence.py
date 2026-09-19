"""Content-addressed source snapshots and exact, bounded claim citations."""
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

# A cited passage is stored with the sentences around it so the excerpt a
# reviewer reads is prose, not a tile cut mid-word. Bounded so one quote
# cannot pull in a whole page.
logger = logging.getLogger(__name__)

EXCERPT_CONTEXT = 300
_SENTENCE_END = re.compile(r'[.!?。]["\')\]]?\s|\n')


def excerpt_window(body, start, end):
    """Widen (start, end) to the sentence boundaries around it, within EXCERPT_CONTEXT."""
    lo = max(0, start - EXCERPT_CONTEXT)
    before = list(_SENTENCE_END.finditer(body, lo, start))
    if before:
        start = before[-1].end()
    # From end-1 so a quote that ends on its own full stop closes there.
    after = _SENTENCE_END.search(body, max(start, end - 1), min(len(body), end + EXCERPT_CONTEXT))
    if after:
        end = after.start() + 1 if body[after.start()] != '\n' else after.start()
    return start, min(end, len(body))


def locate_claim_quotes(claims, sources):
    """Find each claim's verbatim quote and replace it with exact offsets.

    The model copies a passage; the runner finds it (typographic quotes,
    dashes, spacing and case folded) in the named source, or failing that in
    any retrieved source of the job, and never asks the model to count.
    """
    from runtime_tools.commulingo_review_policy import locate
    resolved = []
    for claim in claims:
        source = sources.get(claim.get('source_id'))
        if not source or not source.get('body'):
            raise ValueError('unknown source_id; use an ID from a retrieved source')
        quote = str(claim.get('quote') or '').strip()
        if len(quote) < 20:
            raise ValueError('quote must copy at least 20 characters verbatim from the displayed source text')
        found = locate(source['body'], quote, min_prefix=40)
        if found is None:
            for other in sources.values():
                if other is not source and other.get('body'):
                    found = locate(other['body'], quote, min_prefix=40)
                    if found:
                        source = other
                        break
        if found is None:
            logger.info('research quote not located in %s (%s): %r', claim.get('source_id'), source.get('url'), quote[:300])
            raise ValueError(f'quote not found in {claim.get("source_id")} or any retrieved source: copy 20..2000 '
                             'characters exactly as displayed (no ellipsis, no paraphrase)')
        start, end = excerpt_window(source['body'], *found)
        value = {k: v for k, v in claim.items() if k != 'quote'}
        value.update(source_id=source['id'], start=start, end=end)
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

    A URL has one handle whose current snapshot is the merged text of every
    page fetched (see SourcePages); older snapshots stay in the job's source
    map so carried-over claims still compile, and a persistent ID is still
    accepted and mapped to the URL's current snapshot.
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
                available = ', '.join(f'{handle} ({sources[sid]["url"]})'
                    for handle,sid in self.ids.items() if sources.get(sid,{}).get('body'))
                raise ValueError('unknown source_id; retrieve or use an available source: ' + available)
            # A persistent ID names one snapshot; a model that kept it from an
            # earlier fetch means the URL's current merged text, of which the
            # earlier snapshot is a prefix.
            current = sources.get(self.ids.get(self.by_url.get(source['url'])))
            if current and current.get('body') and current['body'].startswith(source['body']):
                source_id = current['id']
            result.append({**claim, 'source_id':source_id})
        return result


# A URL's merged snapshot may not exceed this many characters. A whole
# Wikipedia article is under 600k; anything larger is a runaway merge, not a
# source. Job 2523 (kosygin) reached a 453 MB snapshot on 2026-09-19 and the
# research process was OOM-killed at 9 GB anon-rss, taking the host down once.
MAX_SNAPSHOT_CHARS = 2_000_000


class SourcePages:
    """One growing snapshot per URL within a research attempt.

    absorb(url, page) returns the merged snapshot and the character span the
    page occupies in it. Pages are identified by content hash, so re-fetching
    the same offset changes nothing, and a quote from any page is found in
    the one snapshot. A page that already lies inside the snapshot, or that
    is itself an earlier merge of it, never gets appended again: seeding a
    retry from the job's stored snapshots (each attempt's merge is stored
    beside the pages it was built from) used to concatenate every earlier
    merge onto the next, so the snapshot grew geometrically per attempt.
    """
    def __init__(self):
        self.current = {}   # url -> merged snapshot
        self.spans = {}     # url -> {page hash: (start, end)}

    def seed(self, sources):
        """Merge the snapshots a job already holds for a URL, oldest first.

        Returns only snapshots that this seeding actually changed, so an
        unchanged job stores nothing new.
        """
        merged = []
        by_url = {}
        for source in sorted((s for s in sources.values() if s.get('body')),
                             key=lambda s: str(s.get('fetched_at') or '')):
            if len(source['body']) > MAX_SNAPSHOT_CHARS:
                logger.warning('skipping oversize stored snapshot %s (%d chars) for %s',
                               source.get('id', '?')[:16], len(source['body']), source['url'])
                continue
            by_url.setdefault(source['url'], []).append(source)
        for url, pages in by_url.items():
            if len(pages) == 1:
                self.current[url] = pages[0]
                self.spans[url] = {pages[0]['content_hash']: (0, len(pages[0]['body']))}
                continue
            known = {page['id'] for page in pages}
            for page in pages:
                self.absorb(url, page['body'])
            if self.current[url]['id'] not in known:
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
        elif body.startswith(current['body']):
            # An earlier merge that already extends the current text: adopt
            # it whole. Existing spans stay valid because the prefix is kept.
            merged = snapshot(url, body)
            spans[digest] = (0, len(body))
        elif (at := current['body'].find(body)) >= 0:
            # Already inside the snapshot (an earlier partial merge, or a page
            # re-fetched with different surrounding whitespace).
            spans[digest] = (at, at + len(body))
            return current, spans[digest], False
        elif len(current['body']) + 1 + len(body) > MAX_SNAPSHOT_CHARS:
            logger.warning('snapshot for %s would exceed %d chars; restarting from the new page',
                           url, MAX_SNAPSHOT_CHARS)
            merged = snapshot(url, body)
            spans.clear()
            spans[digest] = (0, len(body))
        else:
            offset = len(current['body']) + 1
            merged = snapshot(url, current['body'] + '\n' + body)
            spans[digest] = (offset, offset + len(body))
        self.current[url] = merged
        return merged, spans[digest], True
