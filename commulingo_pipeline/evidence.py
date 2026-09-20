"""Content-addressed source snapshots and exact, bounded claim citations.

A displayed source shows every paragraph behind a label ``[S2@12303]`` (the
source handle and the paragraph's character offset). A claim cites labels;
the runner already knows what text each label showed, so nothing is copied,
matched or counted. Copied quotes located by folded substring matching
(2026-09-19, one day) bounced whole results when one copy drifted, and the
numbered 240-character tiles before that drifted from their snapshots.
"""
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

LABEL = re.compile(r'^(S[0-9]+|R[0-9a-f]{16})@([0-9]+)$')
MAX_PARAGRAPH_CHARS = 3000   # a longer paragraph is shown as several labelled pieces
MIN_PASSAGE_CHARS = 20       # a heading-sized fragment is not evidence
MAX_PASSAGE_CHARS = 6000     # the most one claim may cite (compile_evidence bound)
MAX_PASSAGES = 8
_SENTENCE_END = re.compile(r'[.!?。]["\')\]]?\s')


def paragraph_spans(body, first=0, last=None):
    """(start, end) of each non-blank line of body[first:last], in body offsets.

    A line longer than MAX_PARAGRAPH_CHARS is split at sentence ends so every
    piece stays citable within the passage bound.
    """
    last = len(body) if last is None else last
    spans = []
    for match in re.finditer(r'[^\n]+', body[first:last]):
        start, end = first + match.start(), first + match.end()
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


def label_passages(handle, body, first=0, last=None, base=0):
    """Render body[first:last] with a passage label before each paragraph.

    Returns (text, {label: (start, end)}) with spans in ``body`` offsets; the
    label's number is the offset plus ``base`` (a page's position in its
    source when ``body`` is one fetched slice).
    """
    lines, shown = [], {}
    for start, end in paragraph_spans(body, first, last):
        label = f'{handle}@{base + start}'
        shown[label] = (start, end)
        lines.append(f'[{label}] {body[start:end]}')
    return '\n'.join(lines), shown


class Passages:
    """The paragraphs shown in one run, by label: the one registry both the
    research lane (S-handles over merged snapshots) and the review lane
    (R-ids over fetched slices) cite from and resolve against.

    ``show`` renders a text with labels and remembers each paragraph's span
    and a digest of its text; ``resolve`` turns cited labels back into
    evidence ranges. The rules are the same in both lanes: labels of several
    sources give one range per source, paragraphs too far apart for one
    MAX_PASSAGE_CHARS range give one range per contiguous cluster, a label
    never shown is ignored beside shown ones, and a paragraph whose text
    changed since it was shown (a restarted snapshot) is refused. What to do
    when nothing resolves — refuse the claim, drop the check — is the lane's
    policy.
    """
    def __init__(self):
        self.shown = {}   # label -> (handle, start, end, digest)

    def show(self, handle, body, first=0, last=None, base=0):
        text, labels = label_passages(handle, body, first, last, base)
        for label, (start, end) in labels.items():
            self.shown[label] = (handle, start, end, _digest(body[start:end]))
        return text

    def resolve(self, labels, body_of, limit=MAX_PASSAGE_CHARS):
        """([(handle, start, end)], unknown labels). ``body_of(handle)`` returns
        that source's current text, or None when it is no longer available."""
        by_handle, unknown = {}, []
        for label in (str(label) for label in labels):
            entry = self.shown.get(label)
            if entry is None:
                unknown.append(label)
            else:
                by_handle.setdefault(entry[0], []).append(entry[1:])
        ranges = []
        for handle, spans in by_handle.items():
            body = body_of(handle)
            if body is None:
                raise ValueError(f'source {handle} is not available; retrieve it again')
            spans = sorted(set(spans))
            if any(_digest(body[s:e]) != d for s, e, d in spans):
                raise ValueError(f'the text of {handle} changed since those passages were shown; retrieve it again '
                                 'and cite the new labels')
            for cluster in _clusters(spans, limit):
                ranges.append((handle, cluster[0][0], cluster[-1][1]))
        return ranges, unknown


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


def resolve_passages(claims, passages, handles, sources):
    """Replace each claim's passage labels with source snapshots and character ranges.

    One resolved claim per range ``passages.resolve`` returns; a claim none of
    whose labels was shown, or whose range is a heading-sized fragment, is
    refused by claim number.
    """
    def body_of(handle):
        source = sources.get(handles.ids.get(handle)) or {}
        return source.get('body') or None
    resolved = []
    for index, claim in enumerate(claims, 1):
        labels = claim.get('passages') or []
        if not labels:
            raise ValueError(f'claim {index}: passages must list the labels shown in brackets before the paragraphs '
                             'that state it (for example S2@12303)')
        try:
            ranges, unknown = passages.resolve(labels, body_of)
        except ValueError as exc:
            raise ValueError(f'claim {index}: {exc}') from exc
        if not ranges:
            raise ValueError(f'claim {index}: passage label {unknown[0]!r} was not displayed in this research; copy a '
                             'label exactly as shown in brackets at the start of a paragraph')
        if unknown:
            logger.info('claim %d: ignoring passage labels never displayed: %s', index, unknown)
        for handle, start, end in ranges:
            if end - start < MIN_PASSAGE_CHARS:
                raise ValueError(f'claim {index}: the cited passage is a heading or fragment of {end - start} '
                                 'characters; cite the paragraph that states the fact')
            value = {k: v for k, v in claim.items() if k != 'passages'}
            value.update(source_id=sources[handles.ids[handle]]['id'], start=start, end=end)
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
        if not MIN_PASSAGE_CHARS <= end-start <= MAX_PASSAGE_CHARS:
            raise ValueError(f'source range must contain {MIN_PASSAGE_CHARS}..{MAX_PASSAGE_CHARS} characters')
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
