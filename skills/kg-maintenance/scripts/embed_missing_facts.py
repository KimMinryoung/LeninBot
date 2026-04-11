#!/usr/bin/env python3
"""
Generate fact_embedding for any RELATES_TO edge that has a fact text but no
embedding. Used after the recover_lost_edges.py recovery, since the pre-recovery
backup did not preserve embeddings.

Uses the same Gemini embedder as graph_memory.service so the new embeddings
are interchangeable with graphiti-managed ones.

Usage:
    python embed_missing_facts.py             # dry-run (counts only)
    python embed_missing_facts.py --execute   # actually embed and write back
"""
import argparse
import asyncio
import os
import time

from dotenv import load_dotenv
load_dotenv("/home/grass/leninbot/.env")

from neo4j import GraphDatabase

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]
NEO4J_DB = os.environ.get("NEO4J_DATABASE", "neo4j")


def fetch_missing_embedding_edges(driver) -> list[dict]:
    """Edges that have a non-empty fact but no fact_embedding."""
    with driver.session(database=NEO4J_DB) as s:
        rows = s.run("""
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE r.fact IS NOT NULL AND r.fact <> ''
              AND r.fact_embedding IS NULL
            RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact,
                   a.name AS source_name, b.name AS target_name
        """)
        return [dict(r) for r in rows]


async def embed_all(edges: list[dict], execute: bool):
    """Generate embeddings via the same GeminiEmbedder graph_memory uses."""
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=os.environ["GEMINI_API_KEY"],
            embedding_model="gemini-embedding-001",
        )
    )

    print(f"  embedding {len(edges)} facts via Gemini (gemini-embedding-001)...")
    results = []
    failed = 0
    t0 = time.time()
    for i, e in enumerate(edges, 1):
        text = (e["fact"] or "").replace("\n", " ").strip()
        if not text:
            failed += 1
            continue
        try:
            vec = await embedder.create(input_data=[text])
            results.append({"uuid": e["uuid"], "embedding": vec})
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"  [embed error] uuid={e['uuid'][:8]} {exc}")
        if i % 25 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  ... {i}/{len(edges)} embedded ({rate:.1f}/s, {failed} failed)")
        # Light rate limiting — 0.05s between calls is plenty for Gemini paid tier
        await asyncio.sleep(0.05)

    print(f"  embedded {len(results)}/{len(edges)} successfully ({failed} failed)")

    if not execute:
        print("\n[DRY RUN] No writes to Neo4j. Re-run with --execute.")
        return

    # Write back
    print("\n  writing embeddings back to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    written = 0
    with driver.session(database=NEO4J_DB) as s:
        for r in results:
            try:
                s.run("""
                    MATCH ()-[e:RELATES_TO]->()
                    WHERE e.uuid = $uuid
                    SET e.fact_embedding = $emb
                """, uuid=r["uuid"], emb=r["embedding"])
                written += 1
            except Exception as exc:
                print(f"  [write error] uuid={r['uuid'][:8]} {exc}")
    driver.close()
    print(f"  wrote {written}/{len(results)} embeddings")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--execute", action="store_true",
                        help="Actually embed and write back (default: dry-run preview)")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    edges = fetch_missing_embedding_edges(driver)
    driver.close()

    print(f"Edges with fact but no fact_embedding: {len(edges)}")
    if not edges:
        print("Nothing to do.")
        return

    # Show a few examples
    print("\nExamples (first 5):")
    for e in edges[:5]:
        print(f"  [{e['name'] or 'NULL':15s}] {e['source_name']} -> {e['target_name']}")
        print(f"    fact: {(e['fact'] or '')[:120]}")

    if not args.execute:
        # In dry-run, still allow user to see counts but don't call Gemini
        print(f"\n[DRY RUN] Would embed {len(edges)} edges. Re-run with --execute.")
        return

    asyncio.run(embed_all(edges, execute=True))


if __name__ == "__main__":
    main()
