#!/usr/bin/env python3
"""Fold label twins into their canonical node (dry-run by default).

A *label twin* is a node the extractor created because strict label matching
refused an existing node with the same name or strong alias key (미국
[Location] next to United States [Organization], 트루먼 독트린 [Policy] next to
the Concept term, TSMC [Asset] next to the company). Since 2026-09-04 only
Person is a strict label (kg_runtime.identity.STRICT_LABELS), so these now
resolve to one node; this script cleans up the ones already created.

For every non-Person node without external ids the script resolves the node's
own name against the graph (excluding itself, name only). When the hit is
label-compatible the pair is merged; the survivor is, in order: the node with
external ids, a Document node, the older node, the higher-degree node.
Merging goes through identity.merge_entity_nodes_sync (edges moved with
same-predicate dedupe, MENTIONS moved, identity unioned, twin deleted).

    ./venv/bin/python skills/kg-maintenance/scripts/merge_label_twins.py            # plan only
    ./venv/bin/python skills/kg-maintenance/scripts/merge_label_twins.py --execute  # backup + merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "skills", "kg-maintenance", "scripts"))

from kg_runtime.identity import (  # noqa: E402
    STRICT_LABELS, _labels_compatible, merge_entity_nodes_sync, resolve_entity_sync,
)
from kg_runtime.search import _get_neo4j_sync_driver  # noqa: E402

# (twin name, survivor name) pairs the name/alias match gets wrong: a
# CommuLingo alias that collides with an abbreviation (WTO = Warsaw Treaty
# Organization, DiaMat = dialectical materialism), a Korean party name on the
# wrong party, an archive vs the crisis it archives, a summit vs the group …
EXCLUDE = {
    ("World Trade Organization", "바르샤바조약기구"), ("디아마트 (DiaMat)", "dialectical materialism"),
    ("노동당", "UK Labour Party"), ("Agriculture", "농업"), ("1997년 외환위기", "1997 외환위기아카이브"),
    ("오스트리아 마르크스주의", "오스트리아 사회민주주의"), ("Headline CPI", "United States Consumer Price Index"),
    ("G7 에비앙 정상회의", "G7"), ("숨가이트", "숨가이트"), ("Bay of Pigs", "피그만 침공"),
    ("포스펠로프 위원회", "포스펠로프 위원회 보고서 (1956)"), ("파시즘 대평의회", "파시즘 대평의회 문서 세 건 (1928~1943)"),
    ("삼성그룹 초기업노조 삼성전자지부", "초기업노조"), ("자영업", "자영업자"), ("파리 협정", "파리 협정 (1954)"),
    ("재정경제부", "기획재정부"), ("Military Revolutionary Committee", "군사혁명위원회"), ("민주당", "Democratic Party"),
    ("Alphabet", "Google"),
}
# aliases that caused a wrong match and should not stay on the node
STRIP_ALIASES = {("UK Labour Party", "노동당")}

CYPHER_CANDIDATES = """
MATCH (n:Entity)
WHERE size(coalesce(n.external_ids, [])) = 0
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, n.group_id AS gid,
       toString(n.created_at) AS created_at, count(r) AS deg
"""
CYPHER_NODE = """
MATCH (n:Entity {uuid: $uuid})
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, n.group_id AS gid,
       toString(n.created_at) AS created_at, count(r) AS deg, coalesce(n.external_ids, []) AS ext
"""


def _label(labels) -> str:
    ls = [l for l in (labels or []) if l != "Entity"]
    return ls[0] if ls else "Entity"



def choose_survivor(a: dict, b: dict) -> tuple[dict, dict]:
    """(survivor, twin)."""
    for key in (lambda n: bool(n.get("ext")), lambda n: "Document" in (n.get("labels") or [])):
        ka, kb = key(a), key(b)
        if ka != kb:
            return (a, b) if ka else (b, a)
    da, db = a.get("deg") or 0, b.get("deg") or 0
    if da != db:
        return (a, b) if da > db else (b, a)
    ca, cb = a.get("created_at") or "", b.get("created_at") or ""
    return (a, b) if ca <= cb else (b, a)


def plan(session) -> list[dict]:
    cands = [dict(r) for r in session.run(CYPHER_CANDIDATES)]
    out, taken = [], set()
    for c in cands:
        if c["uuid"] in taken:
            continue
        etype = _label(c["labels"])
        if etype in STRICT_LABELS:
            continue
        hit = resolve_entity_sync(session, name=c["name"], entity_type=etype, exclude_uuid=c["uuid"], trusted=False)
        if not hit.found or hit.uuid in taken:
            continue
        if not _labels_compatible(etype, hit.labels):
            continue
        other = session.run(CYPHER_NODE, uuid=hit.uuid).single()
        if not other:
            continue
        other = dict(other)
        c["ext"] = []
        survivor, twin = choose_survivor(other, c)
        if (twin["name"], survivor["name"]) in EXCLUDE or (survivor["name"], twin["name"]) in EXCLUDE:
            continue
        taken.add(twin["uuid"])
        out.append({
            "twin": {k: twin.get(k) for k in ("uuid", "name", "labels", "gid", "deg")},
            "into": {k: survivor.get(k) for k in ("uuid", "name", "labels", "gid", "deg", "ext")},
            "method": hit.method,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"== merge_label_twins [{mode}] {datetime.now():%Y-%m-%d %H:%M:%S}")
    if args.execute:
        from secrets_loader import get_secret
        os.environ.setdefault("NEO4J_PASSWORD", get_secret("NEO4J_PASSWORD") or "")
        from backup_kg import backup
        print(f"backup: {backup(include_embeddings=False)}")
    all_merges = []
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            for node_name, alias in STRIP_ALIASES:
                print(f"strip alias {alias!r} from {node_name!r}")
                if args.execute:
                    s.run("""MATCH (n:Entity {name: $name})
                             SET n.aliases = [a IN coalesce(n.aliases, []) WHERE a <> $alias],
                                 n.alias_keys = [k IN coalesce(n.alias_keys, []) WHERE k <> toLower($alias)],
                                 n.weak_keys = [k IN coalesce(n.weak_keys, []) WHERE k <> toLower($alias)]""",
                          name=node_name, alias=alias).consume()
            # chained twins (A→B, B→C) need another pass once the first is applied
            for pass_no in range(1, 4):
                merges = plan(s)
                print(f"pass {pass_no}: twins to merge: {len(merges)}")
                for m in merges:
                    t, i = m["twin"], m["into"]
                    print(f"  - {t['name']} [{_label(t['labels'])}] deg={t['deg']} → {i['name']} [{_label(i['labels'])}] deg={i['deg']}{' ext' if i['ext'] else ''}")
                all_merges.extend(merges)
                if not merges or not args.execute:
                    break
                stats = {"merged": 0, "edges_moved": 0}
                for m in merges:
                    st = merge_entity_nodes_sync(s, m["into"]["uuid"], [m["twin"]["uuid"]])
                    stats["merged"] += len(st["merged"])
                    stats["edges_moved"] += st["edges_moved"]
                print(f"  applied: {stats}")
    merges = all_merges
    out = os.path.join(ROOT, "data", "kg_backups", f"merge_label_twins_{mode.lower()}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"mode": mode, "merges": merges}, fh, ensure_ascii=False, indent=1, default=str)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
