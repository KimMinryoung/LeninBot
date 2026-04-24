"""
Backfill null group_id / created_at on legacy KG edges (and entities).

Background:
  AuraDB→local 마이그레이션 시 일부 RELATES_TO 엣지의 group_id·created_at가
  null로 남아, knowledge_graph_search가 pydantic 검증에서 통째로 실패했다.
  runtime patch(graph_memory/graphiti_patches.py)는 파싱만 방어하고, 실제 DB는
  여전히 null. 이 스크립트는 DB 자체를 정리해 patch 없이도 깨끗하게 만든다.

Usage:
  NEO4J_PASSWORD=<pw> venv/bin/python scripts/fix_legacy_kg_edges.py [--dry-run]
  또는 systemd-run 으로 credstore 로드:
    sudo systemd-run --pty --same-dir --uid=grass \\
      --property=LoadCredentialEncrypted=neo4j_password:/etc/credstore.encrypted/neo4j_password.cred \\
      --setenv=NEO4J_PASSWORD_FROM_CRED=1 \\
      venv/bin/python scripts/fix_legacy_kg_edges.py

Defaults applied:
  group_id    → "legacy"
  created_at  → 1970-01-01T00:00:00Z  (epoch; 분명한 placeholder)
  episodes    → []                    (null 배열은 []로)
  fact        → ""                    (null은 빈 문자열)
  name        → "RELATES_TO"          (null은 기본값)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# credstore 경유(systemd LoadCredential) 시 파일에서 읽어 env로 노출
if os.environ.get("NEO4J_PASSWORD_FROM_CRED"):
    cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if cred_dir:
        p = Path(cred_dir) / "neo4j_password"
        if p.is_file():
            os.environ["NEO4J_PASSWORD"] = p.read_text().rstrip("\n")

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase


URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

LEGACY_GROUP_ID = "legacy"
EPOCH_CYPHER = "datetime('1970-01-01T00:00:00Z')"


EDGE_AUDIT = """
MATCH ()-[r:RELATES_TO]->()
RETURN
  count(r)                                                 AS total,
  sum(CASE WHEN r.group_id   IS NULL THEN 1 ELSE 0 END)    AS null_group_id,
  sum(CASE WHEN r.created_at IS NULL THEN 1 ELSE 0 END)    AS null_created_at,
  sum(CASE WHEN r.episodes   IS NULL THEN 1 ELSE 0 END)    AS null_episodes,
  sum(CASE WHEN r.fact       IS NULL THEN 1 ELSE 0 END)    AS null_fact,
  sum(CASE WHEN r.name       IS NULL THEN 1 ELSE 0 END)    AS null_name
"""

NODE_AUDIT = """
MATCH (n:Entity)
RETURN
  count(n)                                                 AS total,
  sum(CASE WHEN n.group_id   IS NULL THEN 1 ELSE 0 END)    AS null_group_id,
  sum(CASE WHEN n.created_at IS NULL THEN 1 ELSE 0 END)    AS null_created_at,
  sum(CASE WHEN n.name       IS NULL THEN 1 ELSE 0 END)    AS null_name
"""

EDGE_FIX = f"""
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id   IS NULL
   OR r.created_at IS NULL
   OR r.episodes   IS NULL
   OR r.fact       IS NULL
   OR r.name       IS NULL
SET r.group_id   = coalesce(r.group_id,   '{LEGACY_GROUP_ID}'),
    r.created_at = coalesce(r.created_at, {EPOCH_CYPHER}),
    r.episodes   = coalesce(r.episodes,   []),
    r.fact       = coalesce(r.fact,       ''),
    r.name       = coalesce(r.name,       'RELATES_TO')
RETURN count(r) AS fixed
"""

NODE_FIX = f"""
MATCH (n:Entity)
WHERE n.group_id   IS NULL
   OR n.created_at IS NULL
   OR n.name       IS NULL
SET n.group_id   = coalesce(n.group_id,   '{LEGACY_GROUP_ID}'),
    n.created_at = coalesce(n.created_at, {EPOCH_CYPHER}),
    n.name       = coalesce(n.name,       '')
RETURN count(n) AS fixed
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Audit only — no writes.")
    args = ap.parse_args()

    if not PASSWORD:
        print("ERROR: NEO4J_PASSWORD not set. See header for credstore usage.",
              file=sys.stderr)
        return 1

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session(database=DATABASE) as s:
            print("=== Edge audit (before) ===")
            row = s.run(EDGE_AUDIT).single()
            for k, v in row.items():
                print(f"  {k:20s} = {v}")

            print("\n=== Entity audit (before) ===")
            row = s.run(NODE_AUDIT).single()
            for k, v in row.items():
                print(f"  {k:20s} = {v}")

            if args.dry_run:
                print("\n(dry-run — no writes)")
                return 0

            print("\n=== Applying fixes ===")
            row = s.run(EDGE_FIX).single()
            print(f"  edges  fixed = {row['fixed']}")
            row = s.run(NODE_FIX).single()
            print(f"  nodes  fixed = {row['fixed']}")

            print("\n=== Edge audit (after) ===")
            row = s.run(EDGE_AUDIT).single()
            for k, v in row.items():
                print(f"  {k:20s} = {v}")

            print("\n=== Entity audit (after) ===")
            row = s.run(NODE_AUDIT).single()
            for k, v in row.items():
                print(f"  {k:20s} = {v}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
