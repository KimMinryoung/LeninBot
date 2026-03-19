#!/usr/bin/env python3
"""
관계 없는 고아 엔티티 탐지 및 삭제.
에피소드에도 미등장하는 노드만 삭제.
"""
import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

def cleanup_orphans(dry_run=True):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        # 관계 없는 엔티티 조회
        result = session.run("""
            MATCH (n:Entity)
            WHERE NOT (n)-[]-()
            RETURN n.name AS name, n.type AS type
            LIMIT 200
        """)
        orphans = [(r["name"], r["type"]) for r in result]
        print(f"고아 엔티티: {len(orphans)}개")

        if dry_run:
            print("\n[DRY RUN] 삭제 대상:")
            for name, etype in orphans[:20]:
                print(f"  - {name} ({etype or 'untyped'})")
            if len(orphans) > 20:
                print(f"  ... 외 {len(orphans)-20}개")
        else:
            deleted = session.run("""
                MATCH (n:Entity)
                WHERE NOT (n)-[]-()
                DELETE n
                RETURN count(n) AS cnt
            """)
            cnt = deleted.single()["cnt"]
            print(f"삭제 완료: {cnt}개")

    driver.close()

if __name__ == "__main__":
    import sys
    dry = "--execute" not in sys.argv
    if dry:
        print("※ 실제 삭제하려면 --execute 플래그 사용\n")
    cleanup_orphans(dry_run=dry)
