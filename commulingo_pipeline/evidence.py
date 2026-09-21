"""Content-addressed source snapshots and exact, bounded claim citations.

Displayed paragraphs have attempt-local labels such as ``[P1]``. Each label
binds an immutable snapshot and a canonical paragraph range, never a mutable
URL handle. Persisted evidence still uses snapshot IDs and character ranges.
"""
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

PASSAGE_PATTERN = r'^P[1-9][0-9]*$'
MAX_PARAGRAPH_CHARS = 3000   # a longer paragraph is shown as several labelled pieces
MAX_PASSAGE_CHARS = 6000     # the most one claim may cite (compile_evidence bound)
MAX_PASSAGES = 8
_SENTENCE_END = re.compile(r'[.!?。]["\')\]]?\s')


def paragraph_spans(body):
    """Canonical (start, end) ranges computed from the complete snapshot.

    A line longer than MAX_PARAGRAPH_CHARS is split at sentence ends so every
    piece stays citable within the passage bound.
    """
    spans = []
    for match in re.finditer(r'[^\n]+', body):
        start, end = match.start(), match.end()
        while start < end and body[start].isspace():
            start += 1
        while end > start and body[end - 1].isspace():
            end -= 1
        if end == start:
            continue
        while end - start > MAX_PARAGRAPH_CHARS:
            cut = _SENTENCE_END.search(body, start + MAX_PARAGRAPH_CHARS // 2, min(end, start + MAX_PARAGRAPH_CHARS))
            split = cut.end() if cut else start + MAX_PARAGRAPH_CHARS
            spans.append((start, split))
            start = split
            while start < end and body[start].isspace():
                start += 1
        if end > start:
            spans.append((start, end))
    return spans


class Passages:
    """Immutable passage references shared by research and independent review.

    Viewports select whole canonical paragraphs; they never cut a paragraph
    into a new citation. Repeated displays reuse labels. New snapshots get
    new labels while previously displayed snapshots remain citable.
    """
    def __init__(self):
        self.shown = {}   # label -> (snapshot id, start, end)
        self._labels = {}  # (snapshot id, start, end) -> label
        self._snapshots = {}  # snapshot id -> (digest, canonical ranges)

    def restore(self, shown, sources):
        """Restore runner-owned checkpoint references without reassigning IDs.

        Expired/missing snapshots reserve their old labels but cannot resolve;
        a refetch must not accidentally make an old label mean a new passage.
        """
        if self.shown:
            raise ValueError('restore requires an empty passage registry')
        for label, entry in shown.items():
            if not re.fullmatch(PASSAGE_PATTERN, label) or len(entry) != 3:
                raise ValueError('invalid passage checkpoint')
            source_id, start, end = entry
            source = sources.get(source_id) or {}
            body = source.get('body')
            if body:
                if source_id not in self._snapshots:
                    self._snapshots[source_id] = (_digest(body), paragraph_spans(body))
                if (start, end) not in self._snapshots[source_id][1]:
                    raise ValueError('checkpoint passage is not a canonical source paragraph')
            self.shown[label] = (source_id, start, end)
            self._labels[(source_id, start, end)] = label
        if set(self.shown) != {f'P{i}' for i in range(1, len(self.shown)+1)}:
            raise ValueError('checkpoint passage IDs must be contiguous')

    def show(self, source_id, body, first=0, last=None):
        last = len(body) if last is None else last
        if not 0 <= first <= last <= len(body):
            raise ValueError('invalid source display range')
        digest = _digest(body)
        if source_id not in self._snapshots:
            self._snapshots[source_id] = (digest, paragraph_spans(body))
        stored_digest, spans = self._snapshots[source_id]
        if digest != stored_digest:
            raise ValueError('source snapshot changed; register the retrieved text as a new snapshot')
        lines = []
        for start, end in spans:
            if first == last or end <= first or start >= last:
                continue
            key = (source_id, start, end)
            if key not in self._labels:
                label = f'P{len(self.shown) + 1}'
                self._labels[key] = label
                self.shown[label] = key
            lines.append(f'[{self._labels[key]}] {body[start:end]}')
        return '\n'.join(lines)

    def resolve(self, labels, body_of, limit=MAX_PASSAGE_CHARS):
        """Resolve every label or reject; never silently discard evidence."""
        by_source, unknown = {}, []
        for label in (str(label) for label in labels):
            entry = self.shown.get(label)
            if entry is None:
                unknown.append(label)
            else:
                by_source.setdefault(entry[0], []).append(entry[1:])
        if unknown:
            raise ValueError(f'passage labels not displayed: {", ".join(unknown)}; '
                             'correct only this item using labels already shown, preserving the other items. '
                             'Retrieve the relevant source again only if its labels are unavailable.')
        ranges = []
        for source_id, spans in by_source.items():
            body = body_of(source_id)
            if body is None:
                raise ValueError(f'source for these passages is not available; retrieve it again')
            spans = sorted(set(spans))
            if _digest(body) != self._snapshots[source_id][0]:
                raise ValueError('source snapshot changed since these passages were shown; retrieve it again '
                                 'and cite the new snapshot labels')
            for cluster in _clusters(spans, limit):
                ranges.append((source_id, cluster[0][0], cluster[-1][1]))
        return ranges


def _digest(text):
    return hashlib.blake2b(text.encode(), digest_size=8).digest()


def _clusters(spans, limit):
    """Consecutive (start, end, ...) spans grouped so each group's range stays within ``limit``."""
    groups = []
    for span in spans:
        if groups and span[1] - groups[-1][0][0] <= limit:
            groups[-1].append(span)
        else:
            groups.append([span])
    return groups


def resolve_passages(claims, passages, sources, *, draft_paths=False):
    """Replace each claim's passage labels with source snapshots and character ranges.

    One resolved claim per range ``passages.resolve`` returns; invalid labels
    are refused by claim number even beside valid ones. Whether a cited
    paragraph, however short, supports the claim is the citation gate's
    judgement, not a length rule.
    """
    def body_of(source_id):
        source = sources.get(source_id) or {}
        return source.get('body') or None
    resolved = []
    for index, claim in enumerate(claims, 1):
        labels = claim.get('passages') or []
        if not labels:
            raise ValueError(f'claim {index}: passages must list the labels shown in brackets before the paragraphs '
                             'that state it (for example P12)')
        try:
            ranges = passages.resolve(labels, body_of)
        except ValueError as exc:
            raise ValueError(f'claim {index}: {exc}') from exc
        for source_id, start, end in ranges:
            value = {k: v for k, v in claim.items() if k != 'passages'}
            value.update(source_id=source_id, start=start, end=end)
            if draft_paths:
                value['draft_path'] = f'/claims/{index - 1}'
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
    """The model selects displayed passages; it cannot supply a fabricated excerpt."""
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
        if end - start > MAX_PASSAGE_CHARS:
            raise ValueError(f'source range may contain at most {MAX_PASSAGE_CHARS} characters')
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
    map so carried-over claims still compile.
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

    def seed(self, sources, *, now=None):
        """Merge the snapshots a job already holds for a URL, oldest first.

        Returns only snapshots that this seeding actually changed, so an
        unchanged job stores nothing new.
        """
        merged = []
        by_url = {}
        now = now or datetime.now(timezone.utc)
        for source in sorted((s for s in sources.values() if s.get('body')),
                             key=lambda s: str(s.get('fetched_at') or '')):
            if source['expires_at'] <= now:
                continue
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
                self.absorb(url, page['body'], fetched_at=page['fetched_at'],
                            expires_at=page['expires_at'])
            if self.current[url]['id'] not in known:
                merged.append(self.current[url])
        return merged

    def absorb(self, url, body, *, fetched_at=None, expires_at=None):
        if not isinstance(body, str) or not body.strip():
            raise ValueError('source body is empty')
        if len(body) > MAX_SNAPSHOT_CHARS:
            raise ValueError(f'source page exceeds {MAX_SNAPSHOT_CHARS} characters; retrieve a smaller page')
        body = body.replace('\x00', '\ufffd')
        fetched_at = fetched_at or datetime.now(timezone.utc)
        expires_at = expires_at if expires_at is not None else fetched_at + timedelta(days=14)
        digest = hashlib.sha256(body.encode()).hexdigest()
        current = self.current.get(url)
        spans = self.spans.setdefault(url, {})
        if current is None:
            merged = snapshot(url, body, now=fetched_at)
            spans[digest] = (0, len(body))
        elif digest in spans:
            return current, spans[digest], False
        elif body.startswith(current['body']):
            # An earlier merge that already extends the current text: adopt
            # it whole. Existing spans stay valid because the prefix is kept.
            merged = snapshot(url, body, now=fetched_at)
            spans[digest] = (0, len(body))
        elif (at := current['body'].find(body)) >= 0:
            # Already inside the snapshot (an earlier partial merge, or a page
            # re-fetched with different surrounding whitespace).
            spans[digest] = (at, at + len(body))
            return current, spans[digest], False
        elif len(current['body']) + 1 + len(body) > MAX_SNAPSHOT_CHARS:
            logger.warning('snapshot for %s would exceed %d chars; restarting from the new page',
                           url, MAX_SNAPSHOT_CHARS)
            merged = snapshot(url, body, now=fetched_at)
            spans.clear()
            spans[digest] = (0, len(body))
        else:
            offset = len(current['body']) + 1
            # Appending a page does not re-fetch the older portion. The whole
            # snapshot is usable only while every constituent is still fresh.
            fetched_at = min(fetched_at, current['fetched_at'])
            expires_at = min(expires_at, current['expires_at'])
            merged = snapshot(url, current['body'] + '\n' + body, now=fetched_at)
            spans[digest] = (offset, offset + len(body))
        merged['expires_at'] = expires_at
        self.current[url] = merged
        return merged, spans[digest], True
