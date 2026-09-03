#!/usr/bin/env python3
"""Fill empty Entity.summary deterministically from the node's best facts.

No LLM: summary = up to N highest-tier, active facts touching the node,
joined with " / ". Entities the sync jobs own (external_ids) already carry a
curated summary and are skipped. Dry-run by default.

    NEO4J_PASSWORD=... venv/bin/python scripts/kg_backfill_summaries.py [--execute] [--facts 3] [--limit N]
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

TIER_ORDER = {"anchor": 4, "corroborated": 3, "single": 2, "unverified": 1}

CANDIDATES = """
MATCH (n:Entity)
WHERE coalesce(n.summary, '') = '' AND size(coalesce(n.external_ids, [])) = 0
  AND (n)-[:RELATES_TO]-()
RETURN n.uuid AS uuid, n.name AS name
LIMIT $limit
"""

FACTS = """
MATCH (n:Entity {uuid: $uuid})-[r:RELATES_TO]-()
WHERE r.expired_at IS NULL AND coalesce(r.fact, '') <> ''
OPTIONAL MATCH (ep:Episodic) WHERE ep.uuid IN coalesce(r.episodes, [])
WITH r, collect(ep.name) AS ep_names
RETURN r.fact AS fact, ep_names, toString(r.created_at) AS created_at
"""


def _tier(names) -> int:
    from kg_runtime.search import _tier_from_names
    t = _tier_from_names(names)
    return TIER_ORDER.get(t, 0)


def main() -> int:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--facts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--max-chars", type=int, default=600)
    args = ap.parse_args()

    from kg_runtime.search import _get_neo4j_sync_driver

    updated = 0
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            candidates = [dict(r) for r in s.run(CANDIDATES, limit=args.limit)]
            print(f"candidates (empty summary, has facts, not synced): {len(candidates)}")
            for i, c in enumerate(candidates):
                facts = [dict(r) for r in s.run(FACTS, uuid=c["uuid"])]
                facts.sort(key=lambda f: (-_tier(f["ep_names"]), f["created_at"] or ""), reverse=False)
                facts.sort(key=lambda f: -_tier(f["ep_names"]))
                chosen = []
                for f in facts:
                    if f["fact"] in chosen:
                        continue
                    chosen.append(f["fact"])
                    if len(chosen) >= args.facts:
                        break
                if not chosen:
                    continue
                summary = " / ".join(chosen)
                if len(summary) > args.max_chars:
                    summary = summary[: args.max_chars - 1].rstrip() + "…"
                if i < 5:
                    print(f"  {c['name']}: {summary[:120]}")
                if args.execute:
                    s.run("MATCH (n:Entity {uuid: $uuid}) SET n.summary = $summary", uuid=c["uuid"], summary=summary).consume()
                updated += 1
    print(f"{'updated' if args.execute else 'would update'}: {updated}")
    if not args.execute:
        print("[DRY RUN] re-run with --execute to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
