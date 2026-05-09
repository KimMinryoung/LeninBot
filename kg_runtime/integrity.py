"""Knowledge Graph integrity checks shared by runtime code and scripts."""

from __future__ import annotations

import logging

from kg_runtime.search import _get_neo4j_sync_driver

logger = logging.getLogger(__name__)


EDGE_INTEGRITY_QUERY = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id IS NULL
   OR r.created_at IS NULL
   OR r.episodes IS NULL
   OR r.fact IS NULL
   OR r.name IS NULL
RETURN
  count(r) AS invalid_edges,
  collect(r.uuid)[0..10] AS sample_edge_uuids
"""

NODE_INTEGRITY_QUERY = """
MATCH (n:Entity)
WHERE n.group_id IS NULL
   OR n.created_at IS NULL
   OR n.name IS NULL
RETURN
  count(n) AS invalid_entities,
  collect(n.uuid)[0..10] AS sample_entity_uuids
"""


def check_kg_integrity() -> dict:
    """Return invariant violations that can break Graphiti search parsing."""
    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            with sync_driver.session(database=neo4j_db) as session:
                edge_row = session.execute_read(
                    lambda tx: dict(tx.run(EDGE_INTEGRITY_QUERY).single())
                )
                node_row = session.execute_read(
                    lambda tx: dict(tx.run(NODE_INTEGRITY_QUERY).single())
                )
    except Exception as e:
        logger.error("[KG] integrity check failed: %s", e)
        return {"ok": False, "error": str(e)}

    invalid_edges = int(edge_row.get("invalid_edges") or 0)
    invalid_entities = int(node_row.get("invalid_entities") or 0)
    return {
        "ok": invalid_edges == 0 and invalid_entities == 0,
        "invalid_edges": invalid_edges,
        "invalid_entities": invalid_entities,
        "sample_edge_uuids": edge_row.get("sample_edge_uuids") or [],
        "sample_entity_uuids": node_row.get("sample_entity_uuids") or [],
    }


def format_integrity_status(status: dict) -> str:
    """Compact human-readable integrity status for tool/script output."""
    if status.get("ok"):
        return "KG integrity: ok"
    if status.get("error"):
        return f"KG integrity check failed: {status['error']}"
    return (
        "KG integrity violation: "
        f"{status.get('invalid_edges', 0)} invalid RELATES_TO edges, "
        f"{status.get('invalid_entities', 0)} invalid Entity nodes"
    )
