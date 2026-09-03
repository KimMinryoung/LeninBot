#!/usr/bin/env python3
"""Fold legacy nodes into their identity-bearing twins by alias.

After the CommuLingo mirror, a pre-existing news/analysis node such as
"Stalin" (no external id) may sit next to "Joseph Stalin"
(commulingo:person:stalin) — the resolver only runs on writes, so existing
nodes never got the chance to converge. This pass resolves every node that
has no external_ids through the same identity rules (strong key or unique
weak key, same label) and merges it into the identity-bearing node.

    NEO4J_PASSWORD=... venv/bin/python scripts/kg_merge_by_alias.py [--execute] [--limit N]

Dry-run by default. Merges go through kg_runtime.identity.merge_entity_nodes_sync
(the legacy name is kept as an alias on the canonical node).
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

CANDIDATES = """
MATCH (n:Entity)
WHERE size(coalesce(n.external_ids, [])) = 0 AND n.name IS NOT NULL
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
       size([(n)-[r:RELATES_TO]-() | r]) AS deg
ORDER BY deg DESC
LIMIT $limit
"""


def main() -> int:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    from kg_runtime.identity import resolve_entity_sync, merge_entity_nodes_sync
    from kg_runtime.search import _get_neo4j_sync_driver

    plan = []
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            cands = [dict(r) for r in s.run(CANDIDATES, limit=args.limit)]
            print(f"candidates without external ids: {len(cands)}")
            for c in cands:
                label = next((l for l in c["labels"] if l != "Entity" and not l.startswith("Entity_")), "Entity")
                if label == "Entity":
                    continue
                hit = resolve_entity_sync(s, name=c["name"], entity_type=label, exclude_uuid=c["uuid"])
                if not hit.found:
                    continue
                # Only fold into nodes that actually carry an identity (mirror twins).
                rec = s.run("MATCH (n:Entity {uuid: $u}) RETURN size(coalesce(n.external_ids, [])) AS ids, n.name AS name",
                            u=hit.uuid).single()
                if not rec or not rec["ids"]:
                    continue
                plan.append({"dup": c, "canonical_uuid": hit.uuid, "canonical_name": rec["name"], "method": hit.method})

            print(f"merge plan: {len(plan)} node(s)")
            for p in plan[:60]:
                print(f"  {p['dup']['name']!r} (deg {p['dup']['deg']}) → {p['canonical_name']!r} [{p['method']}]")
            if len(plan) > 60:
                print(f"  … and {len(plan) - 60} more")
            if not args.execute:
                print("[DRY RUN] re-run with --execute to apply")
                return 0

            stats = []
            for p in plan:
                st = merge_entity_nodes_sync(s, p["canonical_uuid"], [p["dup"]["uuid"]])
                stats.append({**p, **{k: st[k] for k in ("edges_moved", "mentions_moved")}})
    log_dir = ROOT / "data" / "kg_backups"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"merge_by_alias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"merged {len(stats)} node(s); edges moved {sum(s['edges_moved'] for s in stats)}; log {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
