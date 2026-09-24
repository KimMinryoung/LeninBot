"""Server-owned slug and chronological key for a new CommuLingo person section."""
import json
import logging
import re

from llm.call_registry import generate_detailed, resolve
from llm.gateway import estimate_cost_usd
from runtime_tools.commulingo_people import _dedup_key, _fold_slug

logger = logging.getLogger(__name__)

FEATURE = 'commulingo_section_slug'
_SLUG = re.compile(r'^[a-z0-9][a-z0-9-]{1,60}$')
_FALLBACK_CHARS = 48
_SYSTEM = ('Return exactly one lowercase English kebab-case slug for the distinct section topic. '
           'Use 2 to 61 characters: letters, digits and hyphens only. '
           'Do not use the person ID, an existing section slug, a generic word such as biography, '
           'a quote, JSON, an explanation, or a suffix to disguise a duplicate topic.')


def section_sort_order(start_year, start_month=None, existing=()):
    """Stored sort key YYYYMM from the author's period start; append when undated.

    Sections render by sort_order. Authors give the year (and month when known)
    as data; the encoding is the server's. The shared store writes 0 for a
    missing key, which put undated sections at the top of the life story.
    """
    if isinstance(start_year, int) and not isinstance(start_year, bool):
        month = start_month if isinstance(start_month, int) and 1 <= start_month <= 12 else 0
        return start_year * 100 + month
    orders = [row.get('sortOrder') for row in existing or () if isinstance(row, dict)]
    orders = [order for order in orders if isinstance(order, int)]
    return max(orders) + 1 if orders else 0


def _available(slug, person_id, taken):
    return bool(slug) and _SLUG.fullmatch(slug) and slug != person_id and slug not in taken


def _fallback_slug(person_id, heading, taken):
    """Deterministic slug from the English heading when the model gives none.

    The slug is an identifier, not reviewed content: a heading that already
    passed the duplicate check must not lose its save to a flaky one-shot.
    """
    words = _fold_slug(str(heading.get('en') or '')).split('-')
    base = ''
    for word in words:
        if base and len(base) + 1 + len(word) > _FALLBACK_CHARS:
            break
        base = f'{base}-{word}' if base else word
    base = base[:_FALLBACK_CHARS].strip('-') or 'section'
    if len(base) < 2:
        base = f'{base}-section'
    slug, n = base, 2
    while not _available(slug, person_id, taken):
        slug, n = f'{base}-{n}', n + 1
    return slug


def generate_section_slug(person_id, heading, body, existing=(), *, usage=None):
    """Generate once at the API boundary. Only a duplicate heading blocks the save."""
    if not isinstance(heading, dict) or not any(str(heading.get(lang) or '').strip() for lang in ('ko', 'en')):
        raise ValueError('section heading is required before its slug can be generated')
    existing = [row for row in (existing or ()) if isinstance(row, dict)]
    # Same key as the store's create validation, checked here before paying for a call.
    wanted = {lang: _dedup_key(str(heading.get(lang) or '')) for lang in ('ko', 'en')}
    for row in existing:
        prior = row.get('heading') if isinstance(row.get('heading'), dict) else {}
        if any(wanted[lang] and wanted[lang] == _dedup_key(str(prior.get(lang) or '')) for lang in wanted):
            raise ValueError(f"section '{row.get('slug')}' already covers this heading; "
                             'update that section or choose a different topic')
    taken = {row.get('slug') for row in existing}
    prompt = json.dumps({
        'person_id': person_id,
        'heading': {lang: str(heading.get(lang) or '')[:180] for lang in ('ko', 'en')},
        'body_excerpt': {lang: str((body or {}).get(lang) or '')[:300] for lang in ('ko', 'en')},
        'existing_sections': [{'slug': row.get('slug'), 'heading': row.get('heading')} for row in existing][:30],
    }, ensure_ascii=False)
    profile = resolve(FEATURE)
    result = generate_detailed(FEATURE, prompt, system=_SYSTEM, profile=profile)
    if usage is not None:
        semantics = 'gemini' if profile.provider == 'gemini' else (
            'anthropic' if profile.provider in {'claude', 'deepseek_anthropic'} else 'openai')
        cost = estimate_cost_usd(profile.model, token_semantics=semantics, **result.usage) or 0
        usage.tracker['auxiliary_cost_usd'] = usage.tracker.get('auxiliary_cost_usd', 0) + cost
        usage.tracker['section_slug_calls'] = usage.tracker.get('section_slug_calls', 0) + 1
    # Models wrap the answer in quotes/backticks or keep capitals; fold before judging.
    slug = _fold_slug((result.text or '').strip().strip('`"\''))
    if _available(slug, person_id, taken):
        return slug
    logger.warning('section slug fallback for %s: %s', person_id,
                   result.error or f'unusable model slug {result.text!r}')
    if usage is not None:
        usage.tracker['section_slug_fallbacks'] = usage.tracker.get('section_slug_fallbacks', 0) + 1
    return _fallback_slug(person_id, heading, taken)
