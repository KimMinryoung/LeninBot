"""Exercise KG merge against Neo4j inside an always-rolled-back transaction."""
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_runtime.identity import merge_entity_nodes_sync
from kg_runtime.search import _get_neo4j_sync_driver


def main():
    ids = [str(uuid4()) for _ in range(3)]
    with _get_neo4j_sync_driver() as (driver, database):
        with driver.session(database=database) as session:
            with session.begin_transaction() as tx:
                try:
                    tx.run("UNWIND $ids AS id CREATE (:Entity:Concept {uuid: id, name: id})", ids=ids).consume()
                    for source, target, edge, text, expired in (
                        (ids[0], ids[2], 'out-a', 'first claim', False),
                        (ids[1], ids[2], 'out-b', 'different claim', True),
                        (ids[2], ids[0], 'in-a', 'same words', False),
                        (ids[2], ids[1], 'in-b', 'same words', False),
                    ):
                        tx.run("""MATCH (s:Entity {uuid:$source}), (t:Entity {uuid:$target})
                            CREATE (s)-[r:RELATES_TO {uuid:$edge, name:'Statement', fact:$text,
                                sync_key:$edge, episodes:[$edge]}]->(t)
                            SET r.expired_at = CASE WHEN $expired THEN datetime() ELSE null END""",
                            source=source, target=target, edge=ids[0]+edge, text=text, expired=expired).consume()
                    stats = merge_entity_nodes_sync(tx, ids[0], [ids[1]])
                    assert stats['edges_moved'] == 2, stats
                    rows = [dict(r) for r in tx.run("""MATCH (n:Entity {uuid:$id})-[r:RELATES_TO]-()
                        RETURN r.uuid AS uuid, r.sync_key AS key, r.episodes AS episodes,
                               r.expired_at IS NOT NULL AS expired""", id=ids[0])]
                    assert len(rows) == 4, rows
                    assert all(r['uuid'] == r['key'] == r['episodes'][0] for r in rows), rows
                    assert sum(r['expired'] for r in rows) == 1, rows
                    assert tx.run("MATCH (n:Entity {uuid:$id}) RETURN count(n) AS n", id=ids[1]).single()['n'] == 0
                finally:
                    tx.rollback()
    print('PASS: incoming/outgoing claims, sync keys, provenance and expiry preserved; transaction rolled back')


if __name__ == '__main__':
    main()
