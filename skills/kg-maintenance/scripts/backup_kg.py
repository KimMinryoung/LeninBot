#!/usr/bin/env python3
"""
KG 백업 — Entity 노드 + RELATES_TO/MENTIONS 엣지를 JSON 덤프.
파괴적 작업 전 필수 실행.
"""
import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "kg_backups")


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "iso_format"):
            return obj.iso_format()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        return super().default(obj)


def backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        # ── Entities ──
        entities = session.run("""
            MATCH (n:Entity)
            RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                   n.summary AS summary, n.group_id AS group_id,
                   n.created_at AS created_at,
                   n.name_embedding AS name_embedding
        """)
        entity_list = [dict(r) for r in entities]

        # ── RELATES_TO edges ──
        relates = session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact,
                   s.uuid AS source_uuid, s.name AS source_name,
                   t.uuid AS target_uuid, t.name AS target_name,
                   r.group_id AS group_id, r.created_at AS created_at,
                   r.valid_at AS valid_at, r.invalid_at AS invalid_at
        """)
        relates_list = [dict(r) for r in relates]

        # ── MENTIONS edges ──
        mentions = session.run("""
            MATCH (ep:Episodic)-[r:MENTIONS]->(n:Entity)
            RETURN ep.uuid AS episode_uuid, ep.name AS episode_name,
                   n.uuid AS entity_uuid, n.name AS entity_name
        """)
        mentions_list = [dict(r) for r in mentions]

    driver.close()

    # ── Write files ──
    ent_path = os.path.join(BACKUP_DIR, f"entities_{ts}.json")
    rel_path = os.path.join(BACKUP_DIR, f"edges_{ts}.json")
    men_path = os.path.join(BACKUP_DIR, f"mentions_{ts}.json")

    # Drop embeddings for smaller file size
    for e in entity_list:
        e.pop("name_embedding", None)

    for path, data, label in [
        (ent_path, entity_list, "entities"),
        (rel_path, relates_list, "RELATES_TO edges"),
        (men_path, mentions_list, "MENTIONS edges"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1, cls=_Encoder)
        print(f"  {label}: {len(data):,}건 → {path}")

    print(f"\n백업 완료 (timestamp: {ts})")
    return ts


if __name__ == "__main__":
    backup()
