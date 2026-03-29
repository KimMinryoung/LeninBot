#!/usr/bin/env python3
"""
KG 관계명 정규화 — 비표준/NULL r.name을 10개 표준 타입으로 분류.

1. r.name IS NULL 또는 비표준인 RELATES_TO 엣지 수집
2. Gemini 배치 분류: fact + source/target 정보 → 표준 타입
3. r.name 업데이트

Usage:
    python normalize_edge_names.py            # dry-run (기본)
    python normalize_edge_names.py --execute  # 실제 업데이트
"""
import os
import json
import time
import argparse

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

STANDARD_TYPES = [
    "Affiliation", "OrgRelation", "PersonalRelation", "Funding",
    "AssetTransfer", "ThreatAction", "Involvement", "Presence",
    "PolicyEffect", "Participation",
]

CLASSIFICATION_PROMPT = """\
Classify each relationship into exactly one of these 10 types:
Affiliation, OrgRelation, PersonalRelation, Funding, AssetTransfer,
ThreatAction, Involvement, Presence, PolicyEffect, Participation

Type definitions:
- Affiliation: Person belongs to / works at Organization
- OrgRelation: Organization↔Organization relations (partner, competitor, supplier, alliance)
- PersonalRelation: Person↔Person relations (colleague, family, mentor)
- Funding: Money flow (investment, grant, contract, aid, sanctions-related funds)
- AssetTransfer: Technology/product/weapon/IP transfer or trade
- ThreatAction: Attack, espionage, sabotage, sanctions enforcement, military action
- Involvement: Entity's role in an Incident or Campaign (perpetrator, victim, witness)
- Presence: Entity is located at / operates in / deployed to a Location
- PolicyEffect: Policy/law/sanction affects an entity (enacts, targets, restricts, exempts)
- Participation: Entity participates in a Campaign (leads, supports, opposes)

For each edge, I give you:
- id: edge identifier
- current_name: current relation name (may be NULL or non-standard)
- fact: the relationship description
- source: source entity (name + type)
- target: target entity (name + type)

Pick the SINGLE best matching type for each edge.
Respond ONLY with valid JSON array: [{"id": "...", "new_name": "..."}]
No markdown fences.

Edges to classify:
"""


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def fetch_nonstandard_edges(driver):
    """Return edges with NULL or non-standard r.name."""
    with driver.session(database=NEO4J_DB) as s:
        result = s.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.name IS NULL
               OR NOT r.name IN $standard_types
            RETURN r.uuid AS uuid, r.name AS current_name, r.fact AS fact,
                   s.name AS source_name,
                   [l IN labels(s) WHERE l <> 'Entity'][0] AS source_type,
                   t.name AS target_name,
                   [l IN labels(t) WHERE l <> 'Entity'][0] AS target_type
        """, standard_types=STANDARD_TYPES)
        return [dict(r) for r in result]


def classify_batch(edges: list[dict]) -> list[dict]:
    """Send a batch to Gemini and return classifications."""
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    batch_input = []
    for e in edges:
        fact = (e.get("fact") or "")[:300]
        batch_input.append({
            "id": e["uuid"],
            "current_name": e["current_name"],
            "fact": fact,
            "source": f"{e['source_name']} ({e['source_type'] or 'Entity'})",
            "target": f"{e['target_name']} ({e['target_type'] or 'Entity'})",
        })

    prompt = CLASSIFICATION_PROMPT + json.dumps(batch_input, ensure_ascii=False)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(temperature=0.0, max_output_tokens=16384),
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Robust JSON parsing — handle truncated output
    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        # Try fixing truncated JSON array by closing it
        fixed = text.rstrip().rstrip(",")
        if not fixed.endswith("]"):
            # Find last complete object
            last_close = fixed.rfind("}")
            if last_close > 0:
                fixed = fixed[:last_close + 1] + "]"
        try:
            results = json.loads(fixed)
        except json.JSONDecodeError:
            raise

    valid = []
    for r in results:
        if r.get("new_name") in STANDARD_TYPES:
            valid.append(r)
        else:
            print(f"  [WARN] Invalid type '{r.get('new_name')}' for {r.get('id')} — skipping")
    return valid


def main():
    parser = argparse.ArgumentParser(description="Normalize KG edge names")
    parser.add_argument("--execute", action="store_true",
                        help="Actually update (default: dry-run)")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Edges per Gemini request (default: 30)")
    args = parser.parse_args()

    driver = get_driver()

    print("Fetching non-standard edges...")
    edges = fetch_nonstandard_edges(driver)
    print(f"  {len(edges)} edges to normalize\n")

    if not edges:
        print("All edge names are standard! Nothing to do.")
        driver.close()
        return

    # ── Current distribution ──
    dist = {}
    for e in edges:
        name = e["current_name"] or "NULL"
        dist[name] = dist.get(name, 0) + 1
    print("Current non-standard distribution:")
    for name, cnt in sorted(dist.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name:25s} {cnt}")
    print()

    # ── Classify in batches ──
    all_results = []
    for i in range(0, len(edges), args.batch_size):
        batch = edges[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total = (len(edges) + args.batch_size - 1) // args.batch_size
        print(f"[Batch {batch_num}/{total}] Classifying {len(batch)} edges...")

        try:
            results = classify_batch(batch)
            all_results.extend(results)
            print(f"  → {len(results)} classified")
        except Exception as e:
            print(f"  [ERROR] {e}")

        if i + args.batch_size < len(edges):
            time.sleep(2)

    # ── Summary ──
    new_dist = {}
    for r in all_results:
        new_dist[r["new_name"]] = new_dist.get(r["new_name"], 0) + 1

    print(f"\n{'='*60}")
    print(f"Classification results ({len(all_results)}/{len(edges)} edges):")
    for t in STANDARD_TYPES:
        if t in new_dist:
            print(f"  {t:20s}: {new_dist[t]}")
    print(f"{'='*60}\n")

    # ── Sample changes ──
    uuid_to_edge = {e["uuid"]: e for e in edges}
    print("Sample changes (first 15):")
    for r in all_results[:15]:
        e = uuid_to_edge.get(r["id"], {})
        old = e.get("current_name") or "NULL"
        print(f"  {old:25s} → {r['new_name']:20s}  "
              f"({e.get('source_name', '?')} → {e.get('target_name', '?')})")
    print()

    if not args.execute:
        print("[DRY RUN] No changes made. Use --execute to apply.\n")
        driver.close()
        return

    # ── Apply ──
    print(f"Applying {len(all_results)} updates...")
    applied = 0
    with driver.session(database=NEO4J_DB) as s:
        for r in all_results:
            try:
                s.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    WHERE r.uuid = $uuid
                    SET r.name = $new_name
                """, uuid=r["id"], new_name=r["new_name"])
                applied += 1
            except Exception as e:
                print(f"  [ERROR] uuid={r['id']}: {e}")

    print(f"Done! {applied}/{len(all_results)} edges updated.")
    driver.close()


if __name__ == "__main__":
    main()
