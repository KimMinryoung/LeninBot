"""Direct Knowledge Graph admin helpers."""

import logging

from kg_runtime.integrity import check_kg_integrity
from kg_runtime.search import _get_neo4j_sync_driver

logger = logging.getLogger(__name__)

_KG_WRITE_BLOCKED_PATTERNS = [
    "DETACH DELETE", "DELETE", "DROP", "REMOVE", "CREATE INDEX", "DROP INDEX",
    "CREATE CONSTRAINT", "DROP CONSTRAINT",
]


def kg_cypher(query: str, write: bool = False) -> dict:
    """Execute Cypher on Neo4j KG.

    Args:
        query: Cypher query string.
        write: If True, execute as write transaction. Destructive operations
               (DETACH DELETE, DROP, etc.) are blocked for safety.

    Returns dict with 'rows' (list of dicts) and 'count'.
    """
    # Block destructive write operations
    if write:
        upper = query.upper().strip()
        for pattern in _KG_WRITE_BLOCKED_PATTERNS:
            if pattern in upper:
                return {"error": f"Blocked: destructive operation '{pattern}' not allowed via kg_cypher. Use dedicated functions.", "rows": [], "count": 0}

    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            with sync_driver.session(database=neo4j_db) as session:
                if write:
                    result = session.execute_write(lambda tx: [dict(r) for r in tx.run(query)])
                else:
                    result = session.execute_read(lambda tx: [dict(r) for r in tx.run(query)])
            response = {"rows": result, "count": len(result)}
            if write:
                response["integrity"] = check_kg_integrity()
            return response
    except Exception as e:
        logger.error("[shared] kg_cypher error: %s", e)
        return {"error": str(e), "rows": [], "count": 0}


def kg_delete_episode(episode_name: str) -> dict:
    """Delete an episode and its orphaned entities from KG.

    Deletes:
    1. The Episodic node with matching name
    2. Entity nodes that have no remaining MENTIONS relationships after deletion

    Returns dict with 'deleted_episode', 'deleted_entities', 'error'.
    """
    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            def _delete(tx):
                # Find the episode
                ep_result = list(tx.run(
                    "MATCH (e:Episodic {name: $name}) RETURN e.uuid AS uuid",
                    name=episode_name
                ))
                if not ep_result:
                    return {"deleted_episode": 0, "deleted_entities": 0, "not_found": True}

                # Delete MENTIONS relationships and track entities
                tx.run(
                    "MATCH (e:Episodic {name: $name})-[r:MENTIONS]->(n:Entity) DELETE r",
                    name=episode_name
                )

                # Delete orphaned entities (no more MENTIONS relationships)
                orphan_result = list(tx.run(
                    "MATCH (n:Entity) WHERE NOT (()-[:MENTIONS]->(n)) "
                    "AND NOT (n)-[:RELATES_TO]-() AND NOT ()-[:RELATES_TO]->(n) "
                    "WITH n, n.name AS name DELETE n RETURN count(n) AS cnt"
                ))
                orphan_count = orphan_result[0]["cnt"] if orphan_result else 0

                # Delete the episode itself
                tx.run("MATCH (e:Episodic {name: $name}) DELETE e", name=episode_name)

                return {"deleted_episode": 1, "deleted_entities": orphan_count, "not_found": False}

            with sync_driver.session(database=neo4j_db) as session:
                result = session.execute_write(_delete)
            result["integrity"] = check_kg_integrity()
            return result
    except Exception as e:
        logger.error("[shared] kg_delete_episode error: %s", e)
        return {"error": str(e)}


def kg_merge_entities(source_name: str, target_name: str) -> dict:
    """Merge source entity into target entity in KG.

    Transfers all relationships from source to target (through
    ``kg_runtime.identity.merge_entity_nodes_sync``, which also unions
    external ids / aliases and keeps the source name as an alias), then
    deletes source.

    Args:
        source_name: Name of the entity to merge FROM (will be deleted).
        target_name: Name of the entity to merge INTO (will be kept).

    Returns dict with 'edges_moved', 'mentions_moved', 'deleted_source', 'merged_into'.
    """
    from kg_runtime.identity import merge_entity_nodes_sync

    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            with sync_driver.session(database=neo4j_db) as session:
                check = list(session.run(
                    "MATCH (s:Entity {name: $src}) MATCH (t:Entity {name: $tgt}) "
                    "RETURN s.uuid AS src_uuid, t.uuid AS tgt_uuid LIMIT 1",
                    src=source_name, tgt=target_name
                ))
                if not check:
                    return {"error": f"One or both entities not found: '{source_name}', '{target_name}'"}
                stats = merge_entity_nodes_sync(session, check[0]["tgt_uuid"], [check[0]["src_uuid"]])
            result = {
                "edges_moved": stats["edges_moved"],
                "mentions_moved": stats["mentions_moved"],
                "deleted_source": source_name,
                "merged_into": target_name,
                "integrity": check_kg_integrity(),
            }
            return result
    except Exception as e:
        logger.error("[shared] kg_merge_entities error: %s", e)
        return {"error": str(e)}
