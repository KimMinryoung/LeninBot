"""Batch-classify untyped KG entities using Gemini.

Reads all entities with only the 'Entity' label (no subtype) from Neo4j,
sends them in batches to Gemini for classification into one of 8 types,
then updates Neo4j with the assigned label.

Usage:
    python scripts/classify_untyped_entities.py [--dry-run]
"""

import os
import sys
import json
import time
import argparse

from dotenv import load_dotenv
load_dotenv()

# -- Neo4j sync driver -------------------------------------------------------

def get_driver():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "")
    if not uri:
        raise RuntimeError("NEO4J_URI not configured")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver, db


def fetch_untyped_entities(driver, db):
    """Return list of {uuid, name, summary} for entities with only 'Entity' label."""
    with driver.session(database=db) as s:
        result = s.run(
            "MATCH (n:Entity) "
            "WHERE size(labels(n)) = 1 "
            "RETURN n.uuid AS uuid, n.name AS name, "
            "       n.summary AS summary"
        )
        return [dict(r) for r in result]


def apply_label(driver, db, uuid: str, label: str):
    """Add a subtype label to an entity node."""
    # Label must be one of the known types — validated before calling
    query = f"MATCH (n:Entity {{uuid: $uuid}}) SET n:{label}"
    with driver.session(database=db) as s:
        s.run(query, uuid=uuid)


# -- Gemini classification ---------------------------------------------------

VALID_TYPES = ["Person", "Organization", "Location", "Asset", "Incident",
               "Policy", "Campaign", "Concept"]

CLASSIFICATION_PROMPT = """\
You are classifying knowledge graph entities into exactly one of these 8 types:

1. **Person** — Individual humans (leaders, experts, activists, etc.)
2. **Organization** — Groups, companies, governments, militaries, parties, institutions, sectors
3. **Location** — Geographic places (countries, cities, regions, facilities)
4. **Asset** — Tangible/intangible assets, technologies, products, weapons, infrastructure, publications, software
5. **Incident** — Specific events with a time: attacks, elections, conferences, crises, disasters
6. **Policy** — Laws, regulations, sanctions, treaties, doctrines, formal rules
7. **Campaign** — Sustained multi-step operations: military ops, influence campaigns, protest movements
8. **Concept** — Abstract ideas, theories, ideologies, social phenomena, academic disciplines, economic indicators

Rules:
- Each entity gets EXACTLY ONE type
- If an entity could be multiple types, pick the most specific one
- Economic indicators (inflation, trade deficit) → Concept
- Named publications/books → Asset
- Named protests/movements → Campaign
- Industry sectors (IT sector, manufacturing sector) → Organization
- Roles without a specific person name (Minister of X) → Person
- Respond ONLY with valid JSON array, no markdown fences

Input: a JSON array of {id, name, summary} objects.
Output: a JSON array of {id, type} objects where type is one of the 8 types above.

Entities to classify:
"""


def classify_batch(entities: list[dict]) -> list[dict]:
    """Send a batch to Gemini and return classifications."""
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Prepare input — strip long summaries for token efficiency
    batch_input = []
    for e in entities:
        summary = (e.get("summary") or "")[:200]
        batch_input.append({
            "id": e["uuid"],
            "name": e["name"],
            "summary": summary,
        })

    prompt = CLASSIFICATION_PROMPT + json.dumps(batch_input, ensure_ascii=False)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(temperature=0.0, max_output_tokens=4096),
    )

    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    results = json.loads(text)

    # Validate
    valid = []
    for r in results:
        if r.get("type") in VALID_TYPES:
            valid.append(r)
        else:
            print(f"  [WARN] Invalid type '{r.get('type')}' for {r.get('id')} — skipping")
    return valid


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify untyped KG entities")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print classifications without updating Neo4j")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Entities per Gemini request (default: 30)")
    args = parser.parse_args()

    driver, db = get_driver()

    print("Fetching untyped entities from Neo4j...")
    entities = fetch_untyped_entities(driver, db)
    print(f"Found {len(entities)} untyped entities\n")

    if not entities:
        print("Nothing to do!")
        driver.close()
        return

    # Process in batches
    all_results = []
    for i in range(0, len(entities), args.batch_size):
        batch = entities[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(entities) + args.batch_size - 1) // args.batch_size
        print(f"[Batch {batch_num}/{total_batches}] Classifying {len(batch)} entities...")

        try:
            results = classify_batch(batch)
            all_results.extend(results)
            print(f"  → {len(results)} classified")
        except Exception as e:
            print(f"  [ERROR] Batch {batch_num} failed: {e}")
            # Continue with next batch

        if i + args.batch_size < len(entities):
            time.sleep(2)  # Rate limit courtesy

    # Build uuid→type map
    type_map = {r["id"]: r["type"] for r in all_results}

    # Summary
    type_counts = {}
    for t in type_map.values():
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\n{'='*60}")
    print(f"Classification summary ({len(type_map)}/{len(entities)} entities):")
    for t in VALID_TYPES:
        if t in type_counts:
            print(f"  {t:15s}: {type_counts[t]}")
    print(f"{'='*60}\n")

    # Print full mapping
    name_map = {e["uuid"]: e["name"] for e in entities}
    for uuid, label in sorted(type_map.items(), key=lambda x: x[1]):
        name = name_map.get(uuid, "?")
        print(f"  [{label:12s}] {name}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made to Neo4j.")
    else:
        print(f"\nApplying {len(type_map)} labels to Neo4j...")
        applied = 0
        for uuid, label in type_map.items():
            try:
                apply_label(driver, db, uuid, label)
                applied += 1
            except Exception as e:
                name = name_map.get(uuid, uuid)
                print(f"  [ERROR] Failed to label {name}: {e}")
        print(f"Done! {applied}/{len(type_map)} labels applied.")

    driver.close()


if __name__ == "__main__":
    main()
