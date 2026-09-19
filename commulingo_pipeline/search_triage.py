"""Search-hit triage, shadow: does each web_search hit cover the research target?

Reviewers fetch 5.8 URLs per review and cite 3.1 of them; research fetches
6 URLs and 4.7 wiki pages for 5.6 cited sources (tool_audit_log, 14 days to
2026-09-19). Search results reach the model in provider order with no
ranking. This module asks a System One model, per hit and from the title,
URL and snippet alone, whether the page directly covers the target, and
records the verdicts (``search_triage`` on the stage's usage tracker, hence
the artifact metrics) without changing what the model sees. After a week
the verdicts are joined with which URLs were fetched and cited to decide
whether showing them would steer the model. Registry feature
``commulingo_search_triage``; ``enabled=false`` stops the calls.
"""
import logging
import re

from llm.call_registry import resolve

logger = logging.getLogger(__name__)

FEATURE = 'commulingo_search_triage'
MAX_HITS = 10
SNIPPET_CHARS = 600

QUESTION = {
    'type': 'choice',
    'instructions': 'Judged from its title, URL and snippet alone, does search hit {key} cover the research target?',
    'criteria': {
        'directly': 'The page is about the target itself (the person, the term, the event) or is a document by or '
                    'about it: a biography, an encyclopedia entry, an archive record, an article on it.',
        'possibly': 'The page mentions the target or a closely related matter but is mainly about something else.',
        'unrelated': 'The page does not concern the target: a namesake, another topic, a shop, index or listing page.',
    },
}

# One hit as web_gateway.search._format_results renders it: a "### title" line,
# the source_kind line, the URL line, then the snippet until the next hit.
_BLOCK = re.compile(r'^### ', re.M)
_HIT = re.compile(r'^(?P<title>[^\n]*)\n\[source_kind=[^\]]*\]\n(?P<url>https?://\S+)\n?(?P<snippet>.*)$', re.S)


def settings():
    extra = resolve(FEATURE).extra or {}
    return {'enabled': bool(extra.get('enabled', True))}


def parse_hits(text):
    """Title, URL and snippet of each hit in a rendered web_search result."""
    body = re.sub(r'\n</external>\s*$', '', str(text))
    hits = []
    for block in _BLOCK.split(body)[1:]:
        match = _HIT.match(block.strip())
        if not match:
            continue
        title = re.sub(r' \([^()]*\)$', '', match['title']).strip()
        hits.append({'title': title, 'url': match['url'].strip(), 'snippet': (match['snippet'] or '').strip()[:SNIPPET_CHARS]})
    return hits[:MAX_HITS]


def _labels(record):
    out = []
    for key in ('name', 'term', 'label', 'title'):
        value = (record or {}).get(key)
        if isinstance(value, dict):
            out.extend(v for v in (value.get('ko'), value.get('en')) if v)
        elif isinstance(value, str) and value:
            out.append(value)
    return out


def search_target(kind, target, current, topic=None):
    """What the searches are for: the dictionary entry under research."""
    return {'kind': kind, 'id': target, 'labels': _labels(current), 'topic': topic}


async def triage_hits(target, hits, *, usage=None, decide=None):
    """One decision for a search's hits; returns [{url, verdict, confidence}] and records them."""
    if not hits or not settings()['enabled']:
        return []
    if decide is None:
        from llm.call_registry import decide as registry_decide
        decide = registry_decide
    keys = [f'h{i}' for i in range(1, len(hits) + 1)]
    state = {'target': target, 'hits': dict(zip(keys, hits))}
    questions = {k: {**QUESTION, 'instructions': QUESTION['instructions'].format(key=k)} for k in keys}
    decision = await decide(FEATURE, state, questions)
    tracker = usage.tracker if usage is not None else None
    if decision is None:
        if tracker is not None:
            tracker['search_triage_unavailable'] = tracker.get('search_triage_unavailable', 0) + 1
        return []
    rows = [{'url': hit['url'], 'verdict': decision.choice(k), 'confidence': round(decision.confidence(k) or 0.0, 3)}
            for k, hit in zip(keys, hits)]
    if tracker is not None:
        tracker.setdefault('search_triage', []).extend(rows)
        tracker['search_triage_calls'] = tracker.get('search_triage_calls', 0) + 1
    logger.info('search triage (shadow) for %s/%s: %s', target.get('kind'), target.get('id'),
                ', '.join(f"{r['verdict']}:{r['confidence']:.2f} {r['url'][:60]}" for r in rows))
    return rows


def shadow(target, usage=None, decide=None):
    """Hook for a web_search wrapper: judge the rendered result, never alter or fail it."""
    async def hook(text):
        try:
            await triage_hits(target, parse_hits(text), usage=usage, decide=decide)
        except Exception as exc:   # shadow: a triage failure must not cost the search
            logger.warning('search triage failed: %s', exc)
    return hook
