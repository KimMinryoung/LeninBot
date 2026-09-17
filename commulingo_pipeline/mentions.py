"""Report mentions per glossary term, read from the frontend's report render cache.

The frontend links glossary terms inside public research reports at render
time (alias registry, first-mention policy) and persists those renders in
data/cache/report-renders.json. Counting the reports whose links include a
term gives the "linked documents" measure the operator chose for ordering
body-less terms (2026-09-17). No name scanning is done here: only the links
the site actually renders count. A missing or unreadable cache yields zero
mentions for every term, which leaves the tie-break order unchanged.
"""
import json
import logging
import os
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = '/home/grass/frontend/data/cache/report-renders.json'


def report_mentions_by_term(path=None):
    path = Path(path or os.environ.get('COMMULINGO_REPORT_RENDER_CACHE', DEFAULT_PATH))
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning('report render cache unavailable (%s); term mentions default to 0', exc)
        return {}
    per_language = {}
    for generation in data.get('generations') or []:
        lang = generation.get('lang') or ''
        counts = per_language.setdefault(lang, Counter())
        for entry in generation.get('entries') or []:
            record = entry[1] if isinstance(entry, list) and len(entry) == 2 else entry
            links = ((record or {}).get('result') or {}).get('links') or []
            for term_id in {link.get('id') for link in links if isinstance(link, dict) and link.get('kind') == 'term'}:
                if term_id:
                    counts[term_id] += 1
    # A report counts once per term, whichever language names it more often.
    merged = {}
    for counts in per_language.values():
        for term_id, count in counts.items():
            merged[term_id] = max(merged.get(term_id, 0), count)
    return merged
