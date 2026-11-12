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


# A URL's merged snapshot may not exceed this many characters. A whole
# Wikipedia article is under 600k; anything larger is a runaway merge, not a
# source. Job 2523 (kosygin) reached a 453 MB snapshot on 2026-09-19 and the
# research process was OOM-killed at 9 GB anon-rss, taking the host down once.
MAX_SNAPSHOT_CHARS = 2_000_000
