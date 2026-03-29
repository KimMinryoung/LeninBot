#!/usr/bin/env python3
"""
KG 전체 정리 파이프라인 — 백업 → 병합 → 정규화 → 고아 삭제 → 타입 부여.

각 단계는 이전 단계 성공 시에만 실행.
--dry-run(기본): 모든 단계를 분석만 수행.
--execute: 실제 변경 적용.

Usage:
    python run_cleanup.py              # 전체 dry-run
    python run_cleanup.py --execute    # 전체 실행
    python run_cleanup.py --step merge # 특정 단계만 실행
"""
import argparse
import sys
import time

STEPS = ["backup", "merge", "normalize", "orphans", "classify"]


def run_backup():
    """Step 1: JSON 백업."""
    from backup_kg import backup
    print("\n" + "=" * 60)
    print("STEP 1/5: KG 백업")
    print("=" * 60)
    backup()
    return True


def run_merge(execute: bool, batch_size: int = 50):
    """Step 2: 중복 엔티티 병합."""
    from merge_entities import (
        get_driver, fix_null_uuids, fetch_all_entities,
        group_names_batch, select_canonical, merge_one_group,
    )
    import os
    import json
    from datetime import datetime

    print("\n" + "=" * 60)
    print("STEP 2/5: 중복 엔티티 병합" + (" [DRY RUN]" if not execute else ""))
    print("=" * 60)

    driver = get_driver()

    # Fix NULL UUIDs (always execute — non-destructive)
    print("\nChecking NULL UUIDs...")
    fix_null_uuids(driver)

    print("\nFetching all entities...")
    entities = fetch_all_entities(driver)
    print(f"  {len(entities)} entities found")

    name_to_entities = {}
    for e in entities:
        if e["name"]:
            name_to_entities.setdefault(e["name"], []).append(e)

    unique_names = sorted(name_to_entities.keys())
    print(f"  {len(unique_names)} unique names")

    # LLM grouping
    print(f"\nGrouping via Gemini (batch_size={batch_size})...")
    all_groups = []
    for i in range(0, len(unique_names), batch_size):
        batch = unique_names[i:i + batch_size]
        batch_num = i // batch_size + 1
        total = (len(unique_names) + batch_size - 1) // batch_size
        print(f"  [Batch {batch_num}/{total}] {len(batch)} names...", end=" ")
        try:
            groups = group_names_batch(batch)
            all_groups.extend(groups)
            print(f"→ {len(groups)} groups")
        except Exception as e:
            print(f"ERROR: {e}")
        if i + batch_size < len(unique_names):
            time.sleep(2)

    # Exact-name duplicates
    for name, ents in name_to_entities.items():
        if len(ents) > 1:
            already = any(
                name in g.get("variants", []) or name == g.get("canonical")
                for g in all_groups
            )
            if not already:
                all_groups.append({"canonical": name, "variants": [name]})

    # Build merge plan
    merge_plan = []
    for g in all_groups:
        canonical_name = g["canonical"]
        variant_names = set(g.get("variants", []))
        variant_names.add(canonical_name)
        group_entities = []
        for vname in variant_names:
            group_entities.extend(name_to_entities.get(vname, []))
        if len(group_entities) < 2:
            continue
        canonical, duplicates = select_canonical(group_entities)
        merge_plan.append({
            "canonical": canonical,
            "duplicates": duplicates,
            "canonical_display": canonical_name,
        })

    total_dups = sum(len(p["duplicates"]) for p in merge_plan)
    print(f"\n{len(merge_plan)} merge groups, {total_dups} duplicate nodes")

    for i, plan in enumerate(merge_plan, 1):
        c = plan["canonical"]
        dups = plan["duplicates"]
        print(f"  [{i}] {plan['canonical_display']}")
        print(f"      canonical: '{c['name']}' (uuid={c['uuid']}, rels={c['rel_count']})")
        for d in dups:
            print(f"      ← merge: '{d['name']}' (uuid={d['uuid']}, rels={d['rel_count']})")

    if not execute:
        print(f"\n[DRY RUN] {total_dups} nodes would be merged.")
        driver.close()
        return True

    # Execute
    print(f"\nExecuting {len(merge_plan)} merges...")
    all_stats = []
    for i, plan in enumerate(merge_plan, 1):
        print(f"  [{i}/{len(merge_plan)}] → {plan['canonical_display']}...", end=" ")
        try:
            stats = merge_one_group(driver, plan["canonical"], plan["duplicates"], execute=True)
            all_stats.append(stats)
            print(f"edges={stats['edges_moved']}, mentions={stats['mentions_moved']}")
        except Exception as e:
            print(f"ERROR: {e}")

    LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "kg_backups")
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"merge_log_{ts}.json")
    import json as _json
    with open(log_path, "w", encoding="utf-8") as f:
        _json.dump(all_stats, f, ensure_ascii=False, indent=2)

    total_edges = sum(s["edges_moved"] for s in all_stats)
    total_mentions = sum(s["mentions_moved"] for s in all_stats)
    print(f"\nMerge complete: {total_dups} nodes merged, {total_edges} edges moved, "
          f"{total_mentions} mentions moved")
    print(f"Log: {log_path}")
    driver.close()
    return True


def run_normalize(execute: bool, batch_size: int = 15):
    """Step 3: 비표준 관계명 정규화."""
    from normalize_edge_names import (
        get_driver, fetch_nonstandard_edges, classify_batch, STANDARD_TYPES, NEO4J_DB,
    )

    print("\n" + "=" * 60)
    print("STEP 3/5: 관계명 정규화" + (" [DRY RUN]" if not execute else ""))
    print("=" * 60)

    driver = get_driver()

    print("\nFetching non-standard edges...")
    edges = fetch_nonstandard_edges(driver)
    print(f"  {len(edges)} edges to normalize")

    if not edges:
        print("  All edge names are standard!")
        driver.close()
        return True

    # Distribution
    dist = {}
    for e in edges:
        name = e["current_name"] or "NULL"
        dist[name] = dist.get(name, 0) + 1
    print("\nCurrent non-standard distribution:")
    for name, cnt in sorted(dist.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name:25s} {cnt}")

    # Classify
    all_results = []
    for i in range(0, len(edges), batch_size):
        batch = edges[i:i + batch_size]
        batch_num = i // batch_size + 1
        total = (len(edges) + batch_size - 1) // batch_size
        print(f"\n[Batch {batch_num}/{total}] Classifying {len(batch)} edges...", end=" ")
        try:
            results = classify_batch(batch)
            all_results.extend(results)
            print(f"→ {len(results)} classified")
        except Exception as e:
            print(f"ERROR: {e}")
        if i + batch_size < len(edges):
            time.sleep(2)

    new_dist = {}
    for r in all_results:
        new_dist[r["new_name"]] = new_dist.get(r["new_name"], 0) + 1
    print(f"\nClassification results ({len(all_results)}/{len(edges)}):")
    for t in STANDARD_TYPES:
        if t in new_dist:
            print(f"  {t:20s}: {new_dist[t]}")

    if not execute:
        print(f"\n[DRY RUN] {len(all_results)} edges would be updated.")
        driver.close()
        return True

    # Apply
    print(f"\nApplying {len(all_results)} updates...")
    applied = 0
    with driver.session(database=NEO4J_DB) as s:
        for r in all_results:
            try:
                s.run("MATCH ()-[r:RELATES_TO]->() WHERE r.uuid = $uuid SET r.name = $new_name",
                      uuid=r["id"], new_name=r["new_name"])
                applied += 1
            except Exception as e:
                print(f"  [ERROR] uuid={r['id']}: {e}")
    print(f"Done: {applied}/{len(all_results)} edges updated.")
    driver.close()
    return True


def run_orphans(execute: bool):
    """Step 4: 고아 엔티티 삭제."""
    from cleanup_orphans import cleanup_orphans

    print("\n" + "=" * 60)
    print("STEP 4/5: 고아 엔티티 삭제" + (" [DRY RUN]" if not execute else ""))
    print("=" * 60)

    cleanup_orphans(dry_run=not execute)
    return True


def run_classify(execute: bool):
    """Step 5: 미분류 엔티티 타입 부여."""
    import os
    import sys

    # classify_untyped_entities.py는 scripts/ 디렉토리에 있음
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
    sys.path.insert(0, scripts_dir)

    from classify_untyped_entities import get_driver, fetch_untyped_entities, classify_batch, apply_label

    print("\n" + "=" * 60)
    print("STEP 5/5: 미분류 엔티티 타입 부여" + (" [DRY RUN]" if not execute else ""))
    print("=" * 60)

    driver, db = get_driver()
    entities = fetch_untyped_entities(driver, db)
    print(f"\n  {len(entities)} untyped entities")

    if not entities:
        print("  Nothing to do!")
        driver.close()
        return True

    all_results = []
    batch_size = 30
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        batch_num = i // batch_size + 1
        total = (len(entities) + batch_size - 1) // batch_size
        print(f"  [Batch {batch_num}/{total}] Classifying {len(batch)}...", end=" ")
        try:
            results = classify_batch(batch)
            all_results.extend(results)
            print(f"→ {len(results)}")
        except Exception as e:
            print(f"ERROR: {e}")
        if i + batch_size < len(entities):
            time.sleep(2)

    type_map = {r["id"]: r["type"] for r in all_results}
    type_counts = {}
    for t in type_map.values():
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nClassified {len(type_map)}/{len(entities)}:")
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:15s}: {cnt}")

    if not execute:
        print(f"\n[DRY RUN] {len(type_map)} labels would be applied.")
        driver.close()
        return True

    applied = 0
    for uuid, label in type_map.items():
        try:
            apply_label(driver, db, uuid, label)
            applied += 1
        except Exception as e:
            print(f"  [ERROR] {uuid}: {e}")
    print(f"Done: {applied}/{len(type_map)} labels applied.")
    driver.close()
    return True


# ── Main ───────────────────────────────────────────────────────

STEP_FNS = {
    "backup":    lambda execute, **kw: run_backup(),
    "merge":     lambda execute, **kw: run_merge(execute),
    "normalize": lambda execute, **kw: run_normalize(execute),
    "orphans":   lambda execute, **kw: run_orphans(execute),
    "classify":  lambda execute, **kw: run_classify(execute),
}


def main():
    parser = argparse.ArgumentParser(
        description="KG 전체 정리 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + " → ".join(STEPS),
    )
    parser.add_argument("--execute", action="store_true",
                        help="실제 변경 적용 (기본: dry-run)")
    parser.add_argument("--step", choices=STEPS,
                        help="특정 단계만 실행")
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else STEPS
    mode = "EXECUTE" if args.execute else "DRY RUN"

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  KG Cleanup Pipeline [{mode:^8s}]       ║")
    print(f"║  Steps: {' → '.join(steps_to_run):<32s} ║")
    print(f"╚══════════════════════════════════════════╝")

    for step_name in steps_to_run:
        fn = STEP_FNS[step_name]
        try:
            ok = fn(execute=args.execute)
            if not ok:
                print(f"\n[ABORT] Step '{step_name}' failed. Stopping pipeline.")
                sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Step '{step_name}' raised: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
