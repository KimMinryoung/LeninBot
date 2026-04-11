#!/usr/bin/env python3
"""
Merge entities with the EXACT same name — no LLM, deterministic.

Reuses the merge machinery from `merge_entities.py` but replaces the
Gemini batch grouping step with a pure name-equality grouping. Useful
when graphiti's entity resolution split a single real entity into
multiple identically-named nodes (e.g. United States ×5).

Usage:
    python merge_exact_name_dupes.py            # dry-run (default)
    python merge_exact_name_dupes.py --execute  # apply

The script highlights groups where the duplicate nodes carry
*different* type labels (e.g. one Location + one Organization). Those
are still merged, since identical names with different LLM-assigned
types are almost always the same real-world entity that the entity
resolver mis-typed once. The label of the highest-degree node wins.
"""
import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv("/home/grass/leninbot/.env")

from merge_entities import (
    NEO4J_DB,
    LOG_DIR,
    get_driver,
    fix_null_uuids,
    fetch_all_entities,
    select_canonical,
    merge_one_group,
)


def fetch_entities_with_labels(driver):
    """Like fetch_all_entities but also returns the type labels per node."""
    with driver.session(database=NEO4J_DB) as s:
        result = s.run("""
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS rel_count
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                   labels(n) AS labels, rel_count
            ORDER BY n.name
        """)
        return [dict(r) for r in result]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the merges (default: dry-run)")
    args = parser.parse_args()

    driver = get_driver()

    print("Checking NULL UUIDs...")
    fix_null_uuids(driver)

    print("\nFetching all entities...")
    entities = fetch_entities_with_labels(driver)
    print(f"  {len(entities)} entities found")

    # Group by exact name (case-sensitive — graphiti is case-sensitive too)
    by_name: dict[str, list[dict]] = {}
    for e in entities:
        if e["name"]:
            by_name.setdefault(e["name"], []).append(e)

    dup_groups = {n: ents for n, ents in by_name.items() if len(ents) > 1}
    print(f"  {len(dup_groups)} duplicate groups (exact name match)")

    if not dup_groups:
        print("\nNo exact-name duplicates. Nothing to do.")
        driver.close()
        return

    # Build merge plan, surface label mismatches
    merge_plan = []
    label_mismatch_groups = []
    for name, ents in sorted(dup_groups.items(), key=lambda kv: -sum(e["rel_count"] for e in kv[1])):
        canonical, duplicates = select_canonical(ents)
        # Note label diversity (excluding the always-present 'Entity' base label)
        label_sets = []
        for e in ents:
            non_base = sorted([l for l in e["labels"] if l != "Entity"])
            label_sets.append(tuple(non_base))
        label_diversity = len(set(label_sets))

        plan = {
            "canonical": canonical,
            "duplicates": duplicates,
            "canonical_display": name,
            "label_sets": label_sets,
            "label_mismatch": label_diversity > 1,
            "total_rels": sum(e["rel_count"] for e in ents),
        }
        merge_plan.append(plan)
        if plan["label_mismatch"]:
            label_mismatch_groups.append(plan)

    print(f"\n{'='*70}")
    print(f"MERGE PLAN — {len(merge_plan)} groups")
    print(f"{'='*70}\n")

    for i, plan in enumerate(merge_plan, 1):
        c = plan["canonical"]
        dups = plan["duplicates"]
        flag = " [LABEL MISMATCH]" if plan["label_mismatch"] else ""
        print(f"  [{i}] {plan['canonical_display']}{flag}  (total rels: {plan['total_rels']})")
        canon_labels = sorted([l for l in c["labels"] if l != "Entity"]) or ["Entity"]
        print(f"      canonical: uuid={c['uuid'][:12]}… rels={c['rel_count']:>4} "
              f"summary={'yes' if c['summary'] else 'no '} labels={canon_labels}")
        for d in dups:
            d_labels = sorted([l for l in d["labels"] if l != "Entity"]) or ["Entity"]
            print(f"      ← merge:   uuid={d['uuid'][:12]}… rels={d['rel_count']:>4} "
                  f"summary={'yes' if d['summary'] else 'no '} labels={d_labels}")
        print()

    total_dups = sum(len(p["duplicates"]) for p in merge_plan)
    print(f"Summary: {len(merge_plan)} groups, {total_dups} nodes to be merged away")
    print(f"  Label-mismatch groups: {len(label_mismatch_groups)}")
    if label_mismatch_groups:
        print("  Label-mismatch group names:")
        for p in label_mismatch_groups[:30]:
            print(f"    - {p['canonical_display']}: {set(map(tuple, p['label_sets']))}")
        if len(label_mismatch_groups) > 30:
            print(f"    ... and {len(label_mismatch_groups) - 30} more")

    if not args.execute:
        print("\n[DRY RUN] No changes made. Re-run with --execute to apply.")
        driver.close()
        return

    print("\nExecuting merges...")
    all_stats = []
    for i, plan in enumerate(merge_plan, 1):
        print(f"  [{i}/{len(merge_plan)}] {plan['canonical_display']:<40s}", end=" ")
        try:
            stats = merge_one_group(driver, plan["canonical"], plan["duplicates"], execute=True)
            all_stats.append(stats)
            print(f"edges={stats['edges_moved']:>4} mentions={stats['mentions_moved']:>4}")
        except Exception as e:
            print(f"ERROR: {e}")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"merge_exact_log_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    total_edges = sum(s["edges_moved"] for s in all_stats)
    total_mentions = sum(s["mentions_moved"] for s in all_stats)
    print(f"\nDone: {total_dups} nodes merged, {total_edges} edges moved, "
          f"{total_mentions} mentions moved")
    print(f"Log: {log_path}")
    driver.close()


if __name__ == "__main__":
    main()
