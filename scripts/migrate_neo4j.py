"""
AuraDB → Local Neo4j 마이그레이션 스크립트
==========================================

사용법:
  1. .env에 AuraDB 접속정보 설정 (AURA_NEO4J_URI, AURA_NEO4J_USER, AURA_NEO4J_PASSWORD, AURA_NEO4J_DATABASE)
  2. 로컬 Neo4j가 bolt://localhost:7687 에서 실행 중이어야 함
  3. python scripts/migrate_neo4j.py

단계:
  Phase 1: AuraDB에서 전체 노드/관계 export (JSON)
  Phase 2: 로컬 Neo4j에 import
  Phase 3: 벡터 인덱스 + 제약조건 생성
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv()

# --- Config ---
AURA_URI = os.getenv("AURA_NEO4J_URI") or os.getenv("NEO4J_URI")
AURA_USER = os.getenv("AURA_NEO4J_USER") or os.getenv("NEO4J_USER")
AURA_PASSWORD = os.getenv("AURA_NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
AURA_DATABASE = os.getenv("AURA_NEO4J_DATABASE") or os.getenv("NEO4J_DATABASE")

LOCAL_URI = os.getenv("LOCAL_NEO4J_URI", "bolt://localhost:7687")
LOCAL_USER = os.getenv("LOCAL_NEO4J_USER", "neo4j")
LOCAL_PASSWORD = os.getenv("LOCAL_NEO4J_PASSWORD", os.getenv("NEO4J_LOCAL_PASSWORD", "changeme"))
LOCAL_DATABASE = os.getenv("LOCAL_NEO4J_DATABASE", "neo4j")

DUMP_DIR = Path("scripts/neo4j_dump")


async def export_from_aura():
    """AuraDB에서 전체 노드와 관계를 JSON으로 export."""
    print(f"=== Phase 1: AuraDB Export ===")
    print(f"  URI: {AURA_URI}")
    print(f"  Database: {AURA_DATABASE}")

    driver = AsyncGraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Export all nodes
        async with driver.session(database=AURA_DATABASE) as session:
            result = await session.run(
                """
                MATCH (n)
                RETURN elementId(n) AS id,
                       labels(n) AS labels,
                       properties(n) AS props
                """
            )
            nodes = []
            async for record in result:
                node = {
                    "id": record["id"],
                    "labels": record["labels"],
                    "props": _serialize_props(record["props"]),
                }
                nodes.append(node)

            print(f"  Nodes exported: {len(nodes)}")

            # Export all relationships
            result = await session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN elementId(a) AS src_id,
                       elementId(b) AS dst_id,
                       type(r) AS rel_type,
                       properties(r) AS props
                """
            )
            rels = []
            async for record in result:
                rel = {
                    "src_id": record["src_id"],
                    "dst_id": record["dst_id"],
                    "rel_type": record["rel_type"],
                    "props": _serialize_props(record["props"]),
                }
                rels.append(rel)

            print(f"  Relationships exported: {len(rels)}")

        # Save to JSON
        with open(DUMP_DIR / "nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes, f, ensure_ascii=False, default=str)

        with open(DUMP_DIR / "relationships.json", "w", encoding="utf-8") as f:
            json.dump(rels, f, ensure_ascii=False, default=str)

        print(f"  Saved to {DUMP_DIR}/")

    finally:
        await driver.close()

    return nodes, rels


def _serialize_props(props: dict) -> dict:
    """Neo4j DateTime 등 특수 타입을 직렬화."""
    result = {}
    for k, v in props.items():
        if hasattr(v, "iso_format"):
            result[k] = v.iso_format()
        elif isinstance(v, list):
            result[k] = [
                item.iso_format() if hasattr(item, "iso_format") else item
                for item in v
            ]
        else:
            result[k] = v
    return result


async def import_to_local(nodes: list, rels: list):
    """로컬 Neo4j에 노드와 관계를 import."""
    print(f"\n=== Phase 2: Local Neo4j Import ===")
    print(f"  URI: {LOCAL_URI}")
    print(f"  Database: {LOCAL_DATABASE}")

    driver = AsyncGraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PASSWORD))

    try:
        async with driver.session(database=LOCAL_DATABASE) as session:
            # Clean existing data
            print("  Clearing existing data...")
            await session.run("MATCH (n) DETACH DELETE n")

            # Import nodes in batches
            print(f"  Importing {len(nodes)} nodes...")
            batch_size = 100
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                for node in batch:
                    labels_str = ":".join(node["labels"]) if node["labels"] else "Node"
                    # Use APOC or manual CREATE
                    await session.run(
                        f"CREATE (n:{labels_str}) SET n = $props, n._old_id = $old_id",
                        props=node["props"],
                        old_id=node["id"],
                    )
                print(f"    {min(i + batch_size, len(nodes))}/{len(nodes)} nodes done")

            # Import relationships
            print(f"  Importing {len(rels)} relationships...")
            for i, rel in enumerate(rels):
                await session.run(
                    f"""
                    MATCH (a {{_old_id: $src_id}})
                    MATCH (b {{_old_id: $dst_id}})
                    CREATE (a)-[r:{rel['rel_type']}]->(b)
                    SET r = $props
                    """,
                    src_id=rel["src_id"],
                    dst_id=rel["dst_id"],
                    props=rel["props"],
                )
                if (i + 1) % 100 == 0:
                    print(f"    {i + 1}/{len(rels)} relationships done")

            # Clean up _old_id
            print("  Cleaning up migration metadata...")
            await session.run("MATCH (n) REMOVE n._old_id")

            print("  Import complete!")

    finally:
        await driver.close()


async def create_indexes():
    """벡터 인덱스 + 제약조건 생성."""
    print(f"\n=== Phase 3: Create Indexes ===")

    driver = AsyncGraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PASSWORD))

    try:
        async with driver.session(database=LOCAL_DATABASE) as session:
            # Vector indexes
            print("  Creating vector indexes...")
            await session.run("""
                CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS
                FOR (n:Entity) ON (n.name_embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

            await session.run("""
                CREATE VECTOR INDEX edge_fact_embedding IF NOT EXISTS
                FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

            print("  Vector indexes created!")

            # Let Graphiti's build_indices_and_constraints handle the rest
            print("  (Remaining indexes will be created by Graphiti on first init)")

    finally:
        await driver.close()


async def verify():
    """마이그레이션 검증."""
    print(f"\n=== Verification ===")

    driver = AsyncGraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PASSWORD))

    try:
        async with driver.session(database=LOCAL_DATABASE) as session:
            result = await session.run("MATCH (n) RETURN count(n) AS cnt")
            record = await result.single()
            print(f"  Total nodes: {record['cnt']}")

            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            record = await result.single()
            print(f"  Total relationships: {record['cnt']}")

            result = await session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
            )
            print("  Node breakdown:")
            async for record in result:
                print(f"    {record['label']}: {record['cnt']}")

    finally:
        await driver.close()


async def main():
    if not AURA_URI:
        print("ERROR: AURA_NEO4J_URI (or NEO4J_URI) not set in .env")
        sys.exit(1)

    # Check if dump already exists (skip export)
    nodes_file = DUMP_DIR / "nodes.json"
    rels_file = DUMP_DIR / "relationships.json"

    if nodes_file.exists() and rels_file.exists():
        print(f"Found existing dump in {DUMP_DIR}/")
        choice = input("Use existing dump? (y/n): ").strip().lower()
        if choice == "y":
            with open(nodes_file, encoding="utf-8") as f:
                nodes = json.load(f)
            with open(rels_file, encoding="utf-8") as f:
                rels = json.load(f)
            print(f"  Loaded {len(nodes)} nodes, {len(rels)} relationships from dump")
        else:
            nodes, rels = await export_from_aura()
    else:
        nodes, rels = await export_from_aura()

    await import_to_local(nodes, rels)
    await create_indexes()
    await verify()

    print("\n=== Migration Complete ===")
    print("Next steps:")
    print("  1. Update .env: NEO4J_URI=bolt://localhost:7687")
    print("  2. Update .env: NEO4J_USER=neo4j")
    print("  3. Update .env: NEO4J_PASSWORD=<your-local-password>")
    print("  4. Update .env: NEO4J_DATABASE=neo4j")
    print("  5. Restart services: systemctl restart leninbot-api leninbot-telegram")


if __name__ == "__main__":
    asyncio.run(main())
