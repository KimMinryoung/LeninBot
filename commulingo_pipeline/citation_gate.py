"""Citation-support gate: does each located excerpt actually support its claim?

locate_claim_quotes proves a quote exists in a source; nothing proved it says
what the claim asserts. On 2026-09-19, 5 of 30 sampled research claims that
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

dev_docs/jev_system_one_adoption.md §4.5 has the measured baseline.
"""
import asyncio
import logging

from llm.call_registry import resolve

logger = logging.getLogger(__name__)

FEATURE = 'commulingo_citation_support'
DEFAULT_THRESHOLDS = {'reject': 0.85, 'boilerplate': 0.9}
MAX_EXCERPT_CHARS = 3000
CONCURRENCY = 8

QUESTIONS = {
    'support': {
        'type': 'choice',
        'instructions': 'Does the quoted source excerpt support the claim made for the field?',
        'criteria': {
            'supports': 'The excerpt states the facts the claim asserts (dates, names, events, roles may be paraphrased).',
            'contradicts': 'The excerpt states something incompatible with the claim.',
            'unrelated': 'The excerpt does not speak to what the claim asserts.',
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


def settings():
    """Hot-reloadable gate settings from the registry entry."""
    profile = resolve(FEATURE)
    extra = profile.extra or {}
    thresholds = {**DEFAULT_THRESHOLDS, **(extra.get('thresholds') or {})}
    return {'enabled': bool(extra.get('enabled', True)), 'enforce': bool(extra.get('enforce', True)),
            'thresholds': thresholds}


def claim_state(claim, source):
    excerpt = source['body'][claim['start']:claim['end']]
    return {'field': claim.get('field'), 'claim': claim.get('claim'), 'source_url': source.get('url'),
            'excerpt': excerpt[:MAX_EXCERPT_CHARS]}


def verdict(decision, thresholds):
    """One claim's recorded check, with the rejection reason when confident."""
    support, confidence = decision.choice('support'), decision.confidence('support') or 0.0
    boilerplate = decision.noul('boilerplate') or 0.0
    check = {'support': support, 'confidence': round(confidence, 3),
             'specific': round(decision.noul('specific') or 0.0, 3), 'boilerplate': round(boilerplate, 3),
             'model': decision.model}
    if boilerplate >= thresholds['boilerplate']:
        check['reject'] = ('the quoted text is a site access check, consent notice or other boilerplate, not '
                           'article content; that source could not be read here — fetch it another way or cite '
                           'a different source')
    elif support in ('unrelated', 'contradicts') and confidence >= thresholds['reject']:
        check['reject'] = (f'the quoted passage is judged {support} to the claim (confidence {confidence:.2f}); '
                           'quote a passage that states the claimed facts (names, dates, figures), cite another '
                           'source, or drop the claim')
    return check


async def check_claims(claims, sources, *, usage=None, decide=None):
    """Judge every located claim; return the checks, raising ValueError on confident failures.

    ``claims`` carry start/end offsets from locate_claim_quotes. Returns one
    check per claim (index-aligned); a claim the model could not judge gets
    {'support': None}. Raises only when enforce is on and at least one claim
    is confidently unsupported — the message names each such claim so the
    model can fix exactly those.
    """
    conf = settings()
    if not conf['enabled'] or not claims:
        return []
    if decide is None:
        from llm.call_registry import decide as registry_decide
        decide = registry_decide
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(claim):
        source = sources.get(claim.get('source_id')) or {}
        if not source.get('body'):
            return {'support': None, 'error': 'source body unavailable'}
        async with gate:
            decision = await decide(FEATURE, claim_state(claim, source), QUESTIONS)
        if decision is None:
            return {'support': None, 'error': 'decision unavailable'}
        return verdict(decision, conf['thresholds'])

    checks = list(await asyncio.gather(*(one(c) for c in claims)))
    if usage is not None:
        tracker = usage.tracker
        tracker['citation_checks'] = tracker.get('citation_checks', 0) + len(checks)
        tracker['citation_unavailable'] = tracker.get('citation_unavailable', 0) + sum(1 for c in checks if c.get('support') is None)
    rejected = [(i, c) for i, c in enumerate(checks) if c.get('reject')]
    if rejected:
        logger.info('citation gate: %d/%d claims rejected (enforce=%s)', len(rejected), len(checks), conf['enforce'])
        if usage is not None:
            usage.tracker['citation_rejections'] = usage.tracker.get('citation_rejections', 0) + len(rejected)
        if conf['enforce']:
            lines = [f'claim {i + 1} ({claims[i].get("field")}, {str(claims[i].get("claim"))[:120]!r}): {c["reject"]}'
                     for i, c in rejected]
            raise ValueError('citation check failed for ' + ('one claim' if len(lines) == 1 else f'{len(lines)} claims')
                             + '; the other claims are fine and may be resubmitted unchanged:\n' + '\n'.join(lines))
    return checks
