#!/usr/bin/env python3
"""KG 복원 — backup_kg.py가 만든 entities/edges/mentions JSON을 Neo4j에 재주입.

Usage (drill against throwaway Neo4j on port 7688):
    python restore_kg.py \\
        --entities entities_YYYYMMDD_HHMMSS.json \\
        --edges    edges_YYYYMMDD_HHMMSS.json \\
        --mentions mentions_YYYYMMDD_HHMMSS.json \\
        --target-uri bolt://localhost:7688 \\
        --target-password testpw \\
        --clear

Safety rails:
  - Refuses to write to production NEO4J_URI unless --force-production.
  - --clear wipes target DB first; never allowed on production URI.

Backup gap (documented): Episodic nodes are not dumped. MENTIONS edges are
restored by creating minimal :Episodic stubs (uuid + name) so the graph
structure is preserved, but episode content (source, valid_at, narrative,
embedding) is lost on restore.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

BATCH = 500
_SAFE_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_label(label: str) -> str:
    if not _SAFE_LABEL.match(label):
        raise ValueError(f"Refusing unsafe label name: {label!r}")
    return label


def _batched(items: list, size: int = BATCH):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _restore_entities(tx, entities: list[dict]) -> int:
    """MERGE entities by uuid. Groups by extra type label so the label can be set statically."""
    by_labels: dict[tuple, list[dict]] = defaultdict(list)
    for e in entities:
        extras = tuple(sorted(l for l in (e.get("labels") or []) if l != "Entity"))
        by_labels[extras].append(e)

    total = 0
    for extras, rows in by_labels.items():
        for lbl in extras:
            _safe_label(lbl)
        label_clause = ":" + ":".join(["Entity", *extras]) if extras else ":Entity"
        query = f"""
        UNWIND $rows AS row
        MERGE (n:Entity {{uuid: row.uuid}})
        SET n{label_clause},
            n.name = row.name,
            n.summary = row.summary,
            n.group_id = row.group_id,
            n.created_at = row.created_at,
            n.name_embedding = row.name_embedding
        """
        for chunk in _batched(rows):
            tx.run(query, rows=chunk)
            total += len(chunk)
    return total


def _restore_episodic_stubs(tx, mentions: list[dict]) -> int:
    """Create minimal :Episodic stubs (uuid + name only) so MENTIONS edges can attach."""
    seen: dict[str, dict] = {}
    for m in mentions:
        uuid = m.get("episode_uuid")
        if uuid and uuid not in seen:
            seen[uuid] = {"uuid": uuid, "name": m.get("episode_name")}
    rows = list(seen.values())
    query = """
    UNWIND $rows AS row
    MERGE (ep:Episodic {uuid: row.uuid})
    ON CREATE SET ep.name = row.name, ep.restored_stub = true
    """
    for chunk in _batched(rows):
        tx.run(query, rows=chunk)
    return len(rows)


def _restore_relates(tx, edges: list[dict]) -> int:
    query = """
    UNWIND $rows AS row
    MATCH (s:Entity {uuid: row.source_uuid})
    MATCH (t:Entity {uuid: row.target_uuid})
    MERGE (s)-[r:RELATES_TO {uuid: row.uuid}]->(t)
    SET r.name = row.name,
        r.fact = row.fact,
        r.fact_embedding = row.fact_embedding,
        r.group_id = row.group_id,
        r.created_at = row.created_at,
        r.valid_at = row.valid_at,
        r.invalid_at = row.invalid_at,
        r.expired_at = row.expired_at,
        r.episodes = row.episodes,
        r.attributes = row.attributes
    """
    total = 0
    for chunk in _batched(edges):
        tx.run(query, rows=chunk)
        total += len(chunk)
    return total


def _restore_mentions(tx, mentions: list[dict]) -> int:
    query = """
    UNWIND $rows AS row
    MATCH (ep:Episodic {uuid: row.episode_uuid})
    MATCH (n:Entity {uuid: row.entity_uuid})
    MERGE (ep)-[:MENTIONS]->(n)
    """
    total = 0
    for chunk in _batched(mentions):
        tx.run(query, rows=chunk)
        total += len(chunk)
    return total


def _final_counts(session) -> dict:
    return {
        "entities": session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"],
        "relates_to": session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"],
        "mentions": session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c").single()["c"],
        "episodic_stubs": session.run(
            "MATCH (ep:Episodic) WHERE ep.restored_stub = true RETURN count(ep) AS c"
        ).single()["c"],
    }


def _clear(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--entities", required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--mentions", required=True)
    p.add_argument("--target-uri", default=os.getenv("RESTORE_NEO4J_URI"))
    p.add_argument("--target-user", default=os.getenv("RESTORE_NEO4J_USER", "neo4j"))
    p.add_argument("--target-password", default=os.getenv("RESTORE_NEO4J_PASSWORD"))
    p.add_argument("--clear", action="store_true",
                   help="Wipe target DB before restore (drill only; blocked on production URI).")
    p.add_argument("--force-production", action="store_true",
                   help="Allow writing to the production NEO4J_URI from .env (dangerous).")
    args = p.parse_args(argv)

    if not args.target_uri or not args.target_password:
        print("ERROR: --target-uri and --target-password are required (or set RESTORE_NEO4J_URI/PASSWORD).",
              file=sys.stderr)
        return 2

    prod_uri = os.getenv("NEO4J_URI", "")
    writing_to_prod = prod_uri and args.target_uri == prod_uri
    if writing_to_prod and not args.force_production:
        print(f"ERROR: --target-uri matches production NEO4J_URI ({prod_uri}). "
              f"Pass --force-production to proceed.", file=sys.stderr)
        return 3
    if writing_to_prod and args.clear:
        print("ERROR: --clear is not allowed on the production URI.", file=sys.stderr)
        return 3

    print(f"Loading backup JSON...")
    entities = _load_json(args.entities)
    edges = _load_json(args.edges)
    mentions = _load_json(args.mentions)
    print(f"  entities: {len(entities):,}  edges: {len(edges):,}  mentions: {len(mentions):,}")

    driver = GraphDatabase.driver(args.target_uri, auth=(args.target_user, args.target_password))
    try:
        with driver.session() as session:
            if args.clear:
                print("Clearing target DB...")
                _clear(session)

            print("Restoring entities...")
            n_ent = session.execute_write(_restore_entities, entities)
            print(f"  done: {n_ent:,} merged")

            print("Creating Episodic stubs for MENTIONS targets...")
            n_stubs = session.execute_write(_restore_episodic_stubs, mentions)
            print(f"  done: {n_stubs:,} stubs")

            print("Restoring RELATES_TO edges...")
            n_rel = session.execute_write(_restore_relates, edges)
            print(f"  done: {n_rel:,} merged")

            print("Restoring MENTIONS edges...")
            n_men = session.execute_write(_restore_mentions, mentions)
            print(f"  done: {n_men:,} merged")

            print("\n── Final counts on target ──")
            counts = _final_counts(session)
            for k, v in counts.items():
                print(f"  {k}: {v:,}")

            print("\n── Expected (from backup JSON) ──")
            print(f"  entities:   {len(entities):,}")
            print(f"  relates_to: {len(edges):,}")
            print(f"  mentions:   {len(mentions):,}")

            ok = (
                counts["entities"] == len(entities)
                and counts["relates_to"] == len(edges)
                and counts["mentions"] == len(mentions)
            )
            print(f"\nResult: {'PASS' if ok else 'MISMATCH'}")
            return 0 if ok else 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
