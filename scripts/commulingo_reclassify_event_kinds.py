#!/usr/bin/env python3
"""Re-judge the relation kind of existing person-to-event links with Jev.

The kinds were redefined on 2026-09-29 (commulingo/relation_kinds.py). This
asks Jev, event by event and up to BATCH links per request, which kind each
link's relation and caption describe. It only writes a JSONL report; applying
changes is a separate reviewed step through the frontend store.

  venv/bin/python scripts/commulingo_reclassify_event_kinds.py LINKS.json EVENTS.json OUT.jsonl [--events a,b]

LINKS.json / EVENTS.json are json_agg exports (scripts/query-db). Rows already
in OUT.jsonl are skipped, so an interrupted run resumes.
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commulingo.relation_kinds import HISTORY_RELATION_KIND_DEFINITIONS  # noqa: E402
from llm.call_registry import decide, fan_out  # noqa: E402

FEATURE = 'commulingo_event_relation_kind'
BATCH = 12
CONCURRENCY = 6
QUESTIONS = {'kind': {
    'type': 'choice',
    'instructions': ("How does this person relate to the event in `event`, judged from the person's "
                     "relation and caption? `event.focus` is the side the event centres on: opponents "
                     "are the people on the side against it, and there are none when it says none."),
    'criteria': HISTORY_RELATION_KIND_DEFINITIONS,
}}


def item_state(link):
    return {'person': f"{link['name_en']} ({link.get('years') or ''})",
            'relation': {'en': link['relation_en'], 'ko': link['relation_ko']},
            'caption': {'en': link['note_en'], 'ko': link['note_ko']}}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('links'); parser.add_argument('events'); parser.add_argument('out')
    parser.add_argument('--events', dest='only', default='')
    args = parser.parse_args()
    events = {e['id']: e for e in json.loads(Path(args.events).read_text())}
    done = set()
    out = Path(args.out)
    if out.exists():
        done = {(r['event_id'], r['person_id']) for r in map(json.loads, out.read_text().splitlines())}
    only = set(filter(None, args.only.split(',')))
    by_event = defaultdict(list)
    for link in json.loads(Path(args.links).read_text()):
        if (link['event_id'], link['person_id']) not in done and (not only or link['event_id'] in only):
            by_event[link['event_id']].append(link)
    batches = [(eid, links[i:i + BATCH]) for eid, links in by_event.items() for i in range(0, len(links), BATCH)]
    gate, cost = asyncio.Semaphore(CONCURRENCY), {'usd': 0.0, 'requests': 0, 'failed': 0}

    async def run(event_id, links):
        event = events[event_id]
        ids = {f'c{i + 1}': link for i, link in enumerate(links)}
        state, questions = fan_out({cid: item_state(l) for cid, l in ids.items()}, QUESTIONS,
                                   event={'title': event['title_en'], 'period': event['period'],
                                          'focus': (event.get('focus') or {}).get('en')
                                          or 'none: no single focus, so no one is an opponent',
                                          'summary': event.get('summary_en') or ''})
        async with gate:
            decision = await decide(FEATURE, state, questions)
        cost['requests'] += 1
        if decision is None:
            cost['failed'] += 1
            return
        cost['usd'] += getattr(decision, 'cost_usd', 0) or 0
        with out.open('a') as fh:
            for cid, link in ids.items():
                item = decision.item(cid, QUESTIONS)
                fh.write(json.dumps({'event_id': event_id, 'person_id': link['person_id'], 'current': link['kind'],
                                     'jev': item.choice('kind'), 'confidence': round(item.confidence('kind') or 0, 3),
                                     'relation_ko': link['relation_ko'], 'note_ko': link['note_ko']},
                                    ensure_ascii=False) + '\n')

    await asyncio.gather(*(run(eid, links) for eid, links in batches))
    print(json.dumps({'batches': len(batches), **cost}))


if __name__ == '__main__':
    asyncio.run(main())
