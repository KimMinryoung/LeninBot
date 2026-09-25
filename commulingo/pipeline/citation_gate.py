"""Citation-support gate: does each located excerpt actually support its claim?

resolve_passages proves a cited passage was displayed from a source; nothing
proved it says what the claim asserts. On 2026-09-19, 5 of 30 sampled research claims that
had passed research, draft and independent review cited passages unrelated to
the claim (a Britannica bot-check page, an interview paragraph for an author
list, an aesthetics passage for a 1933 appointment). The reviewer verifies
facts against quotes of its own, so a wrong citation stored as evidence was
never caught.

Each (claim, excerpt) pair is judged by a System One model through the call
registry (feature ``commulingo_citation_support``): a choice supports /
contradicts / unrelated with confidence, plus a noul for access-check or
boilerplate text. A confident non-support rejects the result call so the
model quotes a passage that states the fact, picks another source, or drops
the claim. Every check is recorded on the research artifact for calibration;
an unavailable model never blocks research (the gate then only records).

The same question is asked of the independent reviewer's own checks
(``check_review_checks``, feature ``commulingo_review_citation_support``):
each check pairs a located quote with a Korean finding of what it verifies.
In a 40-check sample from stored reviews, 2 quotes said nothing about their
finding (one under a revise, one under an approve); no false positives, so
it enforces from the start.

dev_docs/jev_system_one_adoption.md §4.5 and §4.8 have the measured baselines.
"""
import asyncio
import hashlib
import json
import logging

from llm.call_registry import fan_out, resolve

logger = logging.getLogger(__name__)

FEATURE = 'commulingo_citation_support'
REVIEW_FEATURE = 'commulingo_review_citation_support'
DEFAULT_THRESHOLDS = {'reject': 0.85, 'boilerplate': 0.9}
MAX_EXCERPT_CHARS = 3000
CONCURRENCY = 8

# 'partially_supports' (2026-09-19) is a passing outcome: on real data most
# confident-looking 'unrelated' verdicts were long compound claims whose
# excerpt covered only part, and a Choice is a relative distribution, so
# without the option that probability leaked into 'unrelated'. Naming it
# keeps 'unrelated' for excerpts that truly cover none of the claim.
QUESTIONS = {
    'support': {
        'type': 'choice',
        'instructions': 'Does the quoted source excerpt support the claim made for the field?',
        'criteria': {
            'supports': 'The excerpt states the facts the claim asserts (dates, names, events, roles may be paraphrased).',
            'partially_supports': 'The excerpt states some of the facts the claim asserts but is silent on others '
                                  '(a compound claim only part of which the excerpt covers).',
            'contradicts': 'The excerpt states something incompatible with the claim.',
            'unrelated': 'The excerpt does not speak to any of the facts the claim asserts.',
        },
    },
    'specific': {
        'type': 'noul',
        'instructions': 'The excerpt contains the specific facts (names, dates, places, figures) the claim relies on, '
                        'not only the general topic.',
    },
    'boilerplate': {
        'type': 'noul',
        'instructions': 'The excerpt is site boilerplate rather than article content: an access or bot check, '
                        'a cookie or consent notice, a login wall, navigation, or an error page.',
    },
}

# The reviewer's check names what its quote verifies (a Korean "finding")
# rather than a field claim; the questions differ only in those words.
REVIEW_QUESTIONS = {
    'support': {
        'type': 'choice',
        'instructions': 'Does the quoted source passage support what the finding says it verifies?',
        'criteria': {
            'supports': 'The passage states the facts the finding says it verifies (names, dates, events, roles may be '
                        'paraphrased or translated).',
            'partially_supports': 'The passage states some of the facts the finding says it verifies but is silent on '
                                  'others (a compound finding only part of which the passage covers).',
            'contradicts': 'The passage states something incompatible with the finding.',
            'unrelated': 'The passage does not speak to any of the facts the finding says it verifies.',
        },
    },
    'specific': {
        'type': 'noul',
        'instructions': 'The passage contains the specific facts (names, dates, places, figures) the finding relies on, '
                        'not only the general topic.',
    },
    'boilerplate': QUESTIONS['boilerplate'],
}


def settings(feature=FEATURE):
    """Hot-reloadable gate settings from the registry entry."""
    profile = resolve(feature)
    extra = profile.extra or {}
    thresholds = {**DEFAULT_THRESHOLDS, **(extra.get('thresholds') or {})}
    return {'enabled': bool(extra.get('enabled', True)), 'enforce': bool(extra.get('enforce', True)),
            'thresholds': thresholds, 'model': (profile.provider, profile.model)}


def claim_state(claim, source):
    excerpt = source['body'][claim['start']:claim['end']]
    return {'field': claim.get('field'), 'claim': claim.get('claim'), 'source_url': source.get('url'),
            'excerpt': excerpt[:MAX_EXCERPT_CHARS]}


def verdict(decision, thresholds, stance='supports'):
    """One claim's recorded check, with the rejection reason when confident.

    A claim filed with stance "disputes" cites a source that contradicts it
    on purpose (research collects conflicting sources too), so for it only
    an unrelated excerpt is a defect.
    """
    support, confidence = decision.choice('support'), decision.confidence('support') or 0.0
    failing = ('unrelated',) if stance == 'disputes' else ('unrelated', 'contradicts')
    boilerplate = decision.noul('boilerplate') or 0.0
    check = {'support': support, 'confidence': round(confidence, 3),
             'specific': round(decision.noul('specific') or 0.0, 3), 'boilerplate': round(boilerplate, 3),
             'model': decision.model}
    if boilerplate >= thresholds['boilerplate']:
        check['reject'] = ('the quoted text is a site access check, consent notice or other boilerplate, not '
                           'article content; that source could not be read here — fetch it another way or cite '
                           'a different source')
    elif support in failing and confidence >= thresholds['reject']:
        check['reject'] = (f'the quoted passage is judged {support} to the claim (confidence {confidence:.2f}); '
                           'quote a passage that states the claimed facts (names, dates, figures), cite another '
                           'source, or drop the claim')
    return check


RECORDED = ('support', 'confidence', 'specific', 'boilerplate')


def claim_key(claim):
    return (claim.get('field'), claim.get('claim'), claim.get('source_id'), claim.get('start'), claim.get('end'),
            claim.get('stance') or 'supports')


def review_check_state(check):
    return {'finding': check.get('finding'), 'quote': str(check.get('quote') or '')[:MAX_EXCERPT_CHARS],
            'source_url': check.get('source')}


def review_check_key(check):
    return (check.get('finding'), check.get('quote'), check.get('source'))


def annotate(claims, checks):
    """Copy of ``claims`` with each judged claim carrying its compact check.

    The numbers ride on the claim itself so they stay aligned with it through
    merge_claims and into the stored artifact; the rejection text and model
    name stay out, because the research artifact is forwarded into the draft
    prompt and a stale "drop the claim" there would steer the drafter.
    """
    out = []
    for i, claim in enumerate(claims):
        # A disabled gate returns no checks at all; a claim it could not judge
        # returns support=None. Either way the claim is stored as it came.
        check = checks[i] if i < len(checks) else None
        if not check or check.get('support') is None:
            out.append(claim)
            continue
        out.append({**claim, 'citation_check': {k: check[k] for k in RECORDED}})
    return out


BATCH_CHARS = 24_000   # state characters per request; 60k of Cyrillic exceeded the model's ceiling
BATCH_ITEMS = 12


def _batches(entries, state):
    """Group (key, item) pairs into requests: each item's state under its own id,
    bounded by BATCH_CHARS/BATCH_ITEMS so one request stays under the ceiling."""
    batches, batch, size = [], [], 0
    for k, item in entries:
        st = state(item)
        chars = len(json.dumps(st, ensure_ascii=False))
        if batch and (size + chars > BATCH_CHARS or len(batch) >= BATCH_ITEMS):
            batches.append(batch); batch, size = [], 0
        batch.append((k, item, st)); size += chars
    if batch:
        batches.append(batch)
    return batches


async def _judge(feature, questions, items, *, key, state, skip, describe, usage=None, decide=None, cache=None,
                 tracker_prefix='', noun='claim'):
    """Shared core: one check per item, index-aligned; raises when enforce is on
    and a verdict is a confident rejection. ``skip(item)`` names why an item
    cannot be judged (None to judge it), ``describe(i, item)`` labels it in
    the rejection message.

    Items are judged in batches: one request carries several items' states
    under ``items.c1``, ``items.c2``, ... and every question for each of them
    (TypeSafe evaluates the questions of one request in parallel; requests,
    not questions, cost latency). One request per claim took 8 concurrent
    round trips of ~600 ms for a 28-claim result call.
    """
    conf = settings(feature)
    if not conf['enabled'] or not items:
        return []
    if decide is None:
        from llm.call_registry import decide as registry_decide
        decide = registry_decide
    cache = cache if cache is not None else {}
    gate = asyncio.Semaphore(CONCURRENCY)
    # Source ids/offsets alone do not identify the evaluated text after a
    # refetch. Include the actual bounded state and policy in the cache key.
    states = [state(item) if not skip(item) else None for item in items]
    keys = [hashlib.sha256(json.dumps(
        [feature, conf.get('model'), questions, conf['thresholds'], key(item), st],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        for item, st in zip(items, states)]
    cache_hits = sum(k in cache for k in keys)
    requests = 0

    # One paid judgement per distinct item: duplicates within the batch share
    # the first item's slot, and anything already judged comes from the cache.
    results, pending = {}, {}
    for k, item, st in zip(keys, items, states):
        if k in cache or k in results or k in pending:
            continue
        reason = skip(item)
        if reason:
            results[k] = {'support': None, 'error': reason}
        else:
            pending[k] = (item, st)

    async def run(batch):
        nonlocal requests
        ids = {f'c{i + 1}': entry for i, entry in enumerate(batch)}
        request_state, request_questions = fan_out({cid: st for cid, (_, _, st) in ids.items()}, questions)
        async with gate:
            requests += 1
            decision = await decide(feature, request_state, request_questions)
        if decision is None:
            results.update({k: {'support': None, 'error': 'decision unavailable'} for k, _, _ in batch})
            return
        for cid, (k, item, _) in ids.items():
            results[k] = verdict(decision.item(cid, questions), conf['thresholds'], item.get('stance') or 'supports')

    prepared = _batches(list(pending.items()), lambda entry: entry[1])
    await asyncio.gather(*(run([(k, entry[0], st) for k, entry, st in batch]) for batch in prepared))
    for k, result in results.items():
        if result.get('support') is not None:
            cache[k] = result
    checks = [cache.get(k) or results[k] for k in keys]
    if usage is not None:
        tracker = usage.tracker
        for metric, count in {'citation_requests': requests, 'citation_cache_hits': cache_hits,
                              'citation_unique_items': len(pending)}.items():
            name = f'{tracker_prefix}{metric}'
            tracker[name] = tracker.get(name, 0) + count
        tracker[f'{tracker_prefix}citation_checks'] = tracker.get(f'{tracker_prefix}citation_checks', 0) + len(checks)
        tracker[f'{tracker_prefix}citation_unavailable'] = (tracker.get(f'{tracker_prefix}citation_unavailable', 0)
                                                            + sum(1 for c in checks if c.get('support') is None))
    rejected = [(i, c) for i, c in enumerate(checks) if c.get('reject')]
    if rejected:
        logger.info('%s: %d/%d rejected (enforce=%s)', feature, len(rejected), len(checks), conf['enforce'])
        if usage is not None:
            usage.tracker[f'{tracker_prefix}citation_rejections'] = (
                usage.tracker.get(f'{tracker_prefix}citation_rejections', 0) + len(rejected))
        if conf['enforce']:
            lines = [f'{describe(i, items[i])}: {c["reject"]}' for i, c in rejected]
            raise ValueError('citation check failed for ' + (f'one {noun}' if len(lines) == 1 else f'{len(lines)} {noun}s')
                             + f'; the other {noun}s are fine and may be resubmitted unchanged:\n' + '\n'.join(lines))
    return checks


async def check_claims(claims, sources, *, usage=None, decide=None, cache=None):
    """Judge every located claim; return the checks, raising ValueError on confident failures.

    ``claims`` carry start/end offsets from locate_claim_quotes. Returns one
    check per claim (index-aligned with the input); a claim the model could
    not judge gets {'support': None}. Raises only when enforce is on and at
    least one claim is confidently unsupported — the message names each such
    claim so the model can fix exactly those. ``cache`` (one dict per result
    handler) makes that promise hold: a claim resubmitted unchanged keeps the
    verdict it already received instead of being judged again.
    """
    def skip(claim):
        return None if (sources.get(claim.get('source_id')) or {}).get('body') else 'source body unavailable'

    return await _judge(FEATURE, QUESTIONS, claims, key=claim_key, skip=skip,
                        state=lambda claim: claim_state(claim, sources.get(claim.get('source_id')) or {}),
                        usage=usage, decide=decide, cache=cache,
                        describe=lambda i, c: (f'claim {i + 1} ({c.get("field")}, {str(c.get("claim"))[:120]!r})'
                                              + (f' [repair {c["draft_path"]}]' if c.get('draft_path') else '')))


def review_gate(usage=None, cache=None):
    """The decision hook for make_handlers(gate=): judge the decision's checks
    and return it with each verdict attached. One per review run so resubmitted
    checks reuse their verdicts. ``usage`` needs a ``tracker`` dict."""
    cache = cache if cache is not None else {}

    async def gate(value):
        checks = value.get('checks', [])
        return {**value, 'checks': annotate(checks, await check_review_checks(checks, usage=usage, cache=cache))}
    return gate


async def check_review_checks(checks, *, usage=None, decide=None, cache=None):
    """Judge the independent reviewer's checks (finding ↔ located quote).

    ``checks`` are the decision's checks after resolve_review_checks, so each
    carries the exact quote text and its source URL. Same contract as
    check_claims: index-aligned results, counters under ``review_`` in the
    usage tracker, ValueError naming the failing checks only when the review
    gate's entry says enforce.
    """
    def skip(check):
        return None if check.get('quote') and check.get('finding') else 'check without quote or finding'

    return await _judge(REVIEW_FEATURE, REVIEW_QUESTIONS, checks, key=review_check_key, skip=skip,
                        state=review_check_state, usage=usage, decide=decide, cache=cache, tracker_prefix='review_',
                        describe=lambda i, c: f'check {i + 1} ({str(c.get("finding"))[:120]!r})', noun='check')
