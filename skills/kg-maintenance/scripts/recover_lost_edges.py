#!/usr/bin/env python3
"""
Recover RELATES_TO edges that were silently dropped by a buggy merge run.

Background: `merge_one_group` in `merge_entities.py` uses a `CALL { ... }`
subquery without an explicit scope clause. On the current Neo4j driver this
fires a deprecation warning AND returns count(*)=0 from the move query —
the edges aren't actually moved. The subsequent `DETACH DELETE dup` then
takes the duplicate node away along with every RELATES_TO edge that was
supposed to migrate. MENTIONS happened to survive because that query path
is structurally different.

This script reads:
  - the pre-merge backup `edges_<TS>.json`
  - the merge log `merge_exact_log_<TS>.json`
and recreates every backup edge whose source or target was a duplicate,
remapped to the surviving canonical node, skipping self-loops and edges
that already exist in the current graph.

Usage:
    python recover_lost_edges.py --backup-edges <path> --merge-log <path>            # dry-run
    python recover_lost_edges.py --backup-edges <path> --merge-log <path> --execute  # apply
"""
import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv("/home/grass/leninbot/.env")

from neo4j import GraphDatabase

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]
NEO4J_DB = os.environ.get("NEO4J_DATABASE", "neo4j")


def build_canonical_map(merge_log: list[dict]) -> dict[str, str]:
    """dup_uuid → canonical_uuid"""
    mapping = {}
    for entry in merge_log:
        canon = entry["canonical_uuid"]
        for d in entry["merged"]:
            mapping[d["uuid"]] = canon
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--backup-edges", required=True, help="Path to edges_<TS>.json")
    parser.add_argument("--merge-log", required=True, help="Path to merge_exact_log_<TS>.json")
    parser.add_argument("--execute", action="store_true", help="Actually create edges")
    args = parser.parse_args()

    print(f"Loading backup edges: {args.backup_edges}")
    with open(args.backup_edges) as f:
        backup_edges = json.load(f)
    print(f"  {len(backup_edges)} backup edges")

    print(f"Loading merge log: {args.merge_log}")
    with open(args.merge_log) as f:
        merge_log = json.load(f)
    canonical_map = build_canonical_map(merge_log)
    print(f"  {len(canonical_map)} dup→canonical mappings")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # ── Step 1: classify backup edges ──
    affected = []  # edges that touch a dup → must be recovered (or skipped)
    untouched = []
    for e in backup_edges:
        src = e["source_uuid"]
        tgt = e["target_uuid"]
        if src in canonical_map or tgt in canonical_map:
            affected.append(e)
        else:
            untouched.append(e)

    print(f"\nBackup edge classification:")
    print(f"  affected (touch a dup): {len(affected)}")
    print(f"  untouched:              {len(untouched)}")

    # ── Step 2: remap and dedupe affected edges ──
    # For each affected edge, compute (new_src, new_tgt, name) tuple.
    # Skip self-loops. Dedupe so each (new_src, name, new_tgt) appears once.
    seen = set()
    plan = []
    skipped_self_loops = 0
    skipped_dupe_in_plan = 0
    for e in affected:
        new_src = canonical_map.get(e["source_uuid"], e["source_uuid"])
        new_tgt = canonical_map.get(e["target_uuid"], e["target_uuid"])
        name = e.get("name") or ""
        if new_src == new_tgt:
            skipped_self_loops += 1
            continue
        key = (new_src, name, new_tgt)
        if key in seen:
            skipped_dupe_in_plan += 1
            continue
        seen.add(key)
        plan.append({**e, "new_source_uuid": new_src, "new_target_uuid": new_tgt})

    print(f"\nAfter remap+dedupe:")
    print(f"  candidate edges to recreate: {len(plan)}")
    print(f"  skipped self-loops:          {skipped_self_loops}")
    print(f"  skipped duplicates-in-plan:  {skipped_dupe_in_plan}")

    # ── Step 3: filter out edges that already exist in current DB ──
    print("\nChecking which candidate edges already exist in current DB...")
    actually_missing = []
    with driver.session(database=NEO4J_DB) as s:
        for p in plan:
            exists = s.run("""
                MATCH (a:Entity {uuid: $src})-[r:RELATES_TO]->(b:Entity {uuid: $tgt})
                WHERE coalesce(r.name, '') = $name
                RETURN count(r) AS c
            """, src=p["new_source_uuid"], tgt=p["new_target_uuid"], name=p.get("name") or "").single()["c"]
            if exists == 0:
                actually_missing.append(p)
    print(f"  edges that need recreation: {len(actually_missing)}")
    print(f"  edges that already exist:   {len(plan) - len(actually_missing)}")

    # ── Step 4: also verify both endpoints still exist ──
    print("\nValidating endpoint existence...")
    valid = []
    invalid_endpoints = []
    with driver.session(database=NEO4J_DB) as s:
        for p in actually_missing:
            row = s.run("""
                OPTIONAL MATCH (a:Entity {uuid: $src})
                OPTIONAL MATCH (b:Entity {uuid: $tgt})
                RETURN a IS NOT NULL AS has_src, b IS NOT NULL AS has_tgt
            """, src=p["new_source_uuid"], tgt=p["new_target_uuid"]).single()
            if row["has_src"] and row["has_tgt"]:
                valid.append(p)
            else:
                invalid_endpoints.append(p)
    print(f"  recreatable: {len(valid)}")
    print(f"  invalid endpoints (skipping): {len(invalid_endpoints)}")
    if invalid_endpoints:
        print("  examples:")
        for p in invalid_endpoints[:5]:
            print(f"    {p['source_name']} -> {p['target_name']}  name={p.get('name')}")

    # ── Step 5: edge name distribution preview ──
    name_dist = {}
    for p in valid:
        name_dist[p.get("name") or "NULL"] = name_dist.get(p.get("name") or "NULL", 0) + 1
    print("\nRecreation plan by edge name:")
    for n, c in sorted(name_dist.items(), key=lambda x: -x[1]):
        print(f"  {n:25s}: {c}")

    if not args.execute:
        print(f"\n[DRY RUN] Would recreate {len(valid)} edges. Re-run with --execute.")
        driver.close()
        return

    # ── Step 6: execute ──
    print(f"\nRecreating {len(valid)} edges...")
    created = 0
    failed = 0
    with driver.session(database=NEO4J_DB) as s:
        for i, p in enumerate(valid, 1):
            props = {
                "uuid": p.get("uuid") or "",
                "name": p.get("name") or "",
                "fact": p.get("fact") or "",
                "group_id": p.get("group_id") or "",
                "created_at": p.get("created_at"),
                "valid_at": p.get("valid_at"),
                "invalid_at": p.get("invalid_at"),
                "expired_at": p.get("expired_at"),
                # episodes MUST be a list (not None) — graphiti's Pydantic
                # EntityEdge model rejects None on reads. Default to empty.
                "episodes": p.get("episodes") if p.get("episodes") is not None else [],
            }
            # Drop nulls so we don't write them as Cypher properties
            # (but keep episodes=[] explicitly — empty list is required)
            props = {k: v for k, v in props.items()
                     if v is not None and (v != "" or k == "episodes")}
            try:
                s.run("""
                    MATCH (a:Entity {uuid: $src})
                    MATCH (b:Entity {uuid: $tgt})
                    CREATE (a)-[r:RELATES_TO]->(b)
                    SET r = $props
                """, src=p["new_source_uuid"], tgt=p["new_target_uuid"], props=props)
                created += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  [ERROR] {p['source_name']} -> {p['target_name']}: {e}")
            if i % 50 == 0:
                print(f"  ... {i}/{len(valid)} ({created} ok, {failed} failed)")

    print(f"\nDone: {created} edges recreated, {failed} failed")
    driver.close()


if __name__ == "__main__":
    main()
