#!/usr/bin/env python3
"""Rebuild the graph nodes of specific CommuLingo persons.

Use when a person node was polluted (e.g. namesakes folded together before
the external-id namespace guard existed): the nodes carrying the given ids
are deleted (their mirror edges with them; agent facts on them are reported
first) and the persons are re-mirrored from Postgres so each gets its own
node.

    NEO4J_PASSWORD=... DB_PASSWORD=... LENINBOT_ALLOW_WRITE=1 \\
    venv/bin/python scripts/kg_resync_persons.py --execute id1 id2 ...
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


def main() -> int:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="+", help="CommuLingo person ids (slugs)")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    from jobs import kg_sync_commulingo as sync
    from kg_runtime.search import _get_neo4j_sync_driver

    ext_ids = [sync.ext_id("person", i) for i in args.ids]
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            rows = [dict(r) for r in s.run(
                "MATCH (n:Entity) WHERE any(x IN coalesce(n.external_ids, []) WHERE x IN $ids) "
                "OPTIONAL MATCH (n)-[r:RELATES_TO]-() WHERE r.sync_key IS NULL "
                "RETURN n.uuid AS uuid, n.name AS name, n.external_ids AS ids, count(r) AS agent_edges", ids=ext_ids)]
            print(f"nodes carrying the ids: {len(rows)}")
            for r in rows:
                print(f"  {r['name']!r} ids={r['ids']} agent_edges={r['agent_edges']}")
            if not args.execute:
                print("[DRY RUN] --execute deletes these nodes and re-mirrors the persons")
                return 0
            deleted = s.run("MATCH (n:Entity) WHERE any(x IN coalesce(n.external_ids, []) WHERE x IN $ids) "
                            "DETACH DELETE n RETURN count(*) AS c", ids=ext_ids).single()["c"]
            print(f"deleted {deleted} node(s)")

    src = sync.load_source()
    facts = sync.build_facts(src, changed={"person": set(args.ids), "event": set(), "term": set(), "office": set()})
    print(f"re-mirroring {len(args.ids)} person(s): {len(facts)} facts")
    res = sync.write_facts(facts)
    print({k: v for k, v in res.items() if k != "errors"}, res["errors"][:2])
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
