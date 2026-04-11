#!/usr/bin/env python3
"""
KG 엔티티 병합 — 동일 실체의 중복 노드를 canonical 노드로 통합.

1. Neo4j에서 전체 Entity name 목록 + 메타데이터 수집
2. Gemini로 배치 분류 ("같은 실체인 이름들을 묶어라")
3. 각 그룹에서 canonical 노드 선택
4. duplicate 노드의 RELATES_TO/MENTIONS 엣지를 canonical로 이전
5. duplicate 노드 삭제

Usage:
    python merge_entities.py              # dry-run (기본)
    python merge_entities.py --execute    # 실제 병합
    python merge_entities.py --batch-size 40
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

BACKUP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "kg_backups")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "kg_backups")

# ── Gemini ─────────────────────────────────────────────────────

GROUPING_PROMPT = """\
You are deduplicating a knowledge graph. Given a list of entity names, group names \
that refer to the SAME real-world entity. Only group truly identical entities — \
different branches, subsidiaries, or related-but-distinct entities must stay separate.

Rules:
- "US", "USA", "U.S.", "United States", "United States of America" → same entity
- "South Korea", "ROK", "Republic of Korea" → same entity
- "Donald Trump", "Trump" → same entity (if no other Trump)
- "Bank of China" vs "People's Bank of China" → DIFFERENT entities
- "Samsung Electronics" vs "Samsung" → only merge if context clearly indicates same
- For each group, choose the best canonical name (full official English name)

Input: JSON array of entity names.
Output: JSON array of groups. Each group: {"canonical": "Full Name", "variants": ["name1", "name2", ...]}
Only include groups with 2+ names. Singletons are omitted.
Respond ONLY with valid JSON array, no markdown fences.

Entity names:
"""


def group_names_batch(names: list[str]) -> list[dict]:
    """Send a batch of names to Gemini and return merge groups."""
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = GROUPING_PROMPT + json.dumps(names, ensure_ascii=False)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(temperature=0.0, max_output_tokens=8192),
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return json.loads(text)


# ── Neo4j helpers ──────────────────────────────────────────────

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def fix_null_uuids(driver):
    """NULL uuid 노드에 randomUUID() 부여. merge의 전제 조건."""
    with driver.session(database=NEO4J_DB) as s:
        result = s.run("""
            MATCH (n:Entity)
            WHERE n.uuid IS NULL
            WITH n, randomUUID() AS new_uuid
            SET n.uuid = new_uuid
            RETURN n.name AS name, new_uuid
        """)
        fixed = [(r["name"], r["new_uuid"]) for r in result]
        if fixed:
            print(f"  NULL UUID 수정: {len(fixed)}개")
            for name, uuid in fixed[:10]:
                print(f"    {name} → {uuid}")
            if len(fixed) > 10:
                print(f"    ... 외 {len(fixed)-10}개")
        else:
            print("  NULL UUID 없음")
    return len(fixed)


def fetch_all_entities(driver):
    """Return list of {uuid, name, summary, labels, rel_count}."""
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


def select_canonical(entities_in_group: list[dict]) -> tuple[dict, list[dict]]:
    """Pick the best node as canonical. Return (canonical, duplicates)."""
    def score(e):
        s = 0
        if e["uuid"]:
            s += 1000
        if e["summary"]:
            s += 100
        s += e["rel_count"]
        return s

    ranked = sorted(entities_in_group, key=score, reverse=True)
    return ranked[0], ranked[1:]


def merge_one_group(driver, canonical: dict, duplicates: list[dict], execute: bool) -> dict:
    """Merge duplicates into canonical. Returns stats."""
    stats = {"canonical": canonical["name"], "canonical_uuid": canonical["uuid"],
             "merged": [], "edges_moved": 0, "mentions_moved": 0}

    canon_uuid = canonical["uuid"]

    for dup in duplicates:
        dup_uuid = dup["uuid"]
        dup_name = dup["name"]
        stats["merged"].append({"name": dup_name, "uuid": dup_uuid})

        if not execute:
            continue

        with driver.session(database=NEO4J_DB) as s:
            # ── Move outgoing RELATES_TO ──
            # NOTE: previous version did `OPTIONAL MATCH (canon)-[existing]->(t)`
            # and then referenced `canon.uuid` inside a CALL subquery. When the
            # OPTIONAL MATCH didn't bind (the common case — no pre-existing edge),
            # `canon` was NULL and `canon.uuid` silently produced no rows, so
            # the CREATE never ran, count(*) returned 0, and the next
            # `DETACH DELETE dup` then destroyed the edge. The fix: don't rely
            # on `canon` from OPTIONAL MATCH; re-MATCH it via the parameter.
            moved = s.run("""
                MATCH (dup:Entity {uuid: $dup_uuid})-[r:RELATES_TO]->(t)
                WHERE t.uuid <> $canon_uuid
                WITH dup, r, t
                OPTIONAL MATCH (:Entity {uuid: $canon_uuid})-[existing:RELATES_TO]->(t)
                WHERE existing.name = r.name
                WITH dup, r, t, existing
                WHERE existing IS NULL
                MATCH (canon:Entity {uuid: $canon_uuid})
                CREATE (canon)-[r2:RELATES_TO]->(t)
                SET r2 = properties(r)
                DELETE r
                RETURN count(r2) AS cnt
            """, dup_uuid=dup_uuid, canon_uuid=canon_uuid)
            stats["edges_moved"] += moved.single()["cnt"]

            # ── Move incoming RELATES_TO ──
            moved = s.run("""
                MATCH (src)-[r:RELATES_TO]->(dup:Entity {uuid: $dup_uuid})
                WHERE src.uuid <> $canon_uuid
                WITH src, r, dup
                OPTIONAL MATCH (src)-[existing:RELATES_TO]->(:Entity {uuid: $canon_uuid})
                WHERE existing.name = r.name
                WITH src, r, dup, existing
                WHERE existing IS NULL
                MATCH (canon:Entity {uuid: $canon_uuid})
                CREATE (src)-[r2:RELATES_TO]->(canon)
                SET r2 = properties(r)
                DELETE r
                RETURN count(r2) AS cnt
            """, dup_uuid=dup_uuid, canon_uuid=canon_uuid)
            stats["edges_moved"] += moved.single()["cnt"]

            # ── Move MENTIONS ──
            moved = s.run("""
                MATCH (ep)-[r:MENTIONS]->(dup:Entity {uuid: $dup_uuid})
                WITH ep, r, dup
                OPTIONAL MATCH (ep)-[existing:MENTIONS]->(canon:Entity {uuid: $canon_uuid})
                WITH ep, r, dup, existing
                WHERE existing IS NULL
                CALL {
                    WITH ep
                    MATCH (c:Entity {uuid: $canon_uuid_inner})
                    CREATE (ep)-[:MENTIONS]->(c)
                    RETURN c
                }
                DELETE r
                RETURN count(*) AS cnt
            """, dup_uuid=dup_uuid, canon_uuid=canon_uuid, canon_uuid_inner=canon_uuid)
            stats["mentions_moved"] += moved.single()["cnt"]

            # ── Merge summary if canonical has none ──
            if not canonical["summary"] and dup.get("summary"):
                s.run("""
                    MATCH (c:Entity {uuid: $uuid})
                    SET c.summary = $summary
                """, uuid=canon_uuid, summary=dup["summary"])

            # ── Delete duplicate (and its remaining edges) ──
            s.run("""
                MATCH (dup:Entity {uuid: $dup_uuid})
                DETACH DELETE dup
            """, dup_uuid=dup_uuid)

    return stats


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge duplicate KG entities")
    parser.add_argument("--execute", action="store_true",
                        help="Actually merge (default: dry-run)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Names per Gemini request (default: 50)")
    args = parser.parse_args()

    driver = get_driver()

    # ── Step 0: Fix NULL UUIDs ──
    print("Checking NULL UUIDs...")
    fix_null_uuids(driver)

    # ── Step 1: Fetch entities ──
    print("\nFetching all entities...")
    entities = fetch_all_entities(driver)
    print(f"  {len(entities)} entities found")

    # Build name→entities map (multiple nodes can have same name)
    name_to_entities = {}
    for e in entities:
        name = e["name"]
        if not name:
            continue
        name_to_entities.setdefault(name, []).append(e)

    unique_names = sorted(name_to_entities.keys())
    print(f"  {len(unique_names)} unique names")

    # ── Step 2: LLM grouping ──
    print(f"\nGrouping via Gemini (batch_size={args.batch_size})...")
    all_groups = []
    for i in range(0, len(unique_names), args.batch_size):
        batch = unique_names[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total = (len(unique_names) + args.batch_size - 1) // args.batch_size
        print(f"  [Batch {batch_num}/{total}] {len(batch)} names...")

        try:
            groups = group_names_batch(batch)
            all_groups.extend(groups)
            print(f"    → {len(groups)} merge groups found")
        except Exception as e:
            print(f"    [ERROR] {e}")

        if i + args.batch_size < len(unique_names):
            time.sleep(2)

    # ── Also add exact-name duplicates (same name, multiple nodes) ──
    for name, ents in name_to_entities.items():
        if len(ents) > 1:
            # Check if already covered by LLM groups
            already_grouped = False
            for g in all_groups:
                if name in g.get("variants", []) or name == g.get("canonical"):
                    already_grouped = True
                    break
            if not already_grouped:
                all_groups.append({"canonical": name, "variants": [name]})

    if not all_groups:
        print("\nNo merge groups found. KG is clean!")
        driver.close()
        return

    # ── Step 3: Build merge plan ──
    print(f"\n{'='*60}")
    print(f"MERGE PLAN — {len(all_groups)} groups")
    print(f"{'='*60}\n")

    merge_plan = []
    for g in all_groups:
        canonical_name = g["canonical"]
        variant_names = set(g.get("variants", []))
        variant_names.add(canonical_name)

        # Collect all entity nodes for these names
        group_entities = []
        for vname in variant_names:
            group_entities.extend(name_to_entities.get(vname, []))

        if len(group_entities) < 2:
            continue

        canonical, duplicates = select_canonical(group_entities)
        # Override canonical name if LLM suggested a better one
        merge_plan.append({
            "canonical": canonical,
            "duplicates": duplicates,
            "canonical_display": canonical_name,
        })

    for i, plan in enumerate(merge_plan, 1):
        c = plan["canonical"]
        dups = plan["duplicates"]
        print(f"  [{i}] {plan['canonical_display']}")
        print(f"      canonical: '{c['name']}' (uuid={c['uuid']}, rels={c['rel_count']}, "
              f"summary={'yes' if c['summary'] else 'no'})")
        for d in dups:
            print(f"      ← merge: '{d['name']}' (uuid={d['uuid']}, rels={d['rel_count']})")
        print()

    total_dups = sum(len(p["duplicates"]) for p in merge_plan)
    print(f"Total: {len(merge_plan)} groups, {total_dups} nodes to merge\n")

    if not args.execute:
        print("[DRY RUN] No changes made. Use --execute to apply.\n")
        driver.close()
        return

    # ── Step 4: Execute merges ──
    print("Executing merges...")
    all_stats = []
    for i, plan in enumerate(merge_plan, 1):
        print(f"  [{i}/{len(merge_plan)}] Merging → {plan['canonical_display']}...", end=" ")
        try:
            stats = merge_one_group(driver, plan["canonical"], plan["duplicates"], execute=True)
            all_stats.append(stats)
            print(f"edges={stats['edges_moved']}, mentions={stats['mentions_moved']}")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Save log ──
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"merge_log_{ts}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nMerge log: {log_path}")

    total_edges = sum(s["edges_moved"] for s in all_stats)
    total_mentions = sum(s["mentions_moved"] for s in all_stats)
    print(f"Summary: {total_dups} nodes merged, {total_edges} edges moved, {total_mentions} mentions moved")

    driver.close()


if __name__ == "__main__":
    main()
