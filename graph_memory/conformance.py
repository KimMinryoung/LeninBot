"""
graph_memory/conformance.py — write-time conformance gate for graphiti episodes.

After every `graphiti.add_episode()` call, the GraphMemoryService runs the
result through `validate_episode_result()` to catch schema violations as soon
as they happen. The same validator is structured so it can also be used by
the (forthcoming) daily conformance scanner against the whole graph.

Validation categories:

  HARD violations — auto-fixed by service.py if requested:
    - self_loops: edge whose source_uuid == target_uuid (never useful)
    - non_entity_endpoints: RELATES_TO whose source or target is not an
      Entity-labelled node (e.g. (Episodic)-[RELATES_TO]->(Entity), the
      anomalous pattern we cleaned up on 2026-04-11)

  SOFT violations — logged for monitoring, not auto-fixed:
    - non_standard_edge_names: edge.name is not one of the 10 standard
      types from EDGE_TYPES and not None
    - edge_type_map_violations: (source_type, target_type, edge_name) is
      not allowed by EDGE_TYPE_MAP, except for the ("Entity","Entity")
      wildcard (Funding/AssetTransfer) which is universally allowed
    - untyped_nodes: a created node has no subtype label beyond "Entity"

The validator is pure (no Neo4j writes). Auto-fix happens in
GraphMemoryService._apply_conformance_fixes() which takes the report and
issues delete queries for hard violations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Standard schema (mirrors graph_memory/edges.py + entities.py) ────────────
# Imported lazily to avoid pulling pydantic models unless needed.

def _load_schema():
    """Lazy import so this module can be used in tooling without graphiti deps."""
    from .edges import EDGE_TYPES
    from .entities import ENTITY_TYPES
    from .config import EDGE_TYPE_MAP
    return set(EDGE_TYPES.keys()), set(ENTITY_TYPES.keys()), EDGE_TYPE_MAP


@dataclass
class ConformanceReport:
    """Structured outcome of a single conformance check pass."""

    # Hard violations — caller may auto-fix
    self_loops: list[dict] = field(default_factory=list)
    non_entity_endpoints: list[dict] = field(default_factory=list)

    # Soft violations — monitoring only
    non_standard_edge_names: list[dict] = field(default_factory=list)
    edge_type_map_violations: list[dict] = field(default_factory=list)
    untyped_nodes: list[dict] = field(default_factory=list)

    # Counts (for quick logging)
    total_edges_checked: int = 0
    total_nodes_checked: int = 0

    def is_clean(self) -> bool:
        return not (
            self.self_loops
            or self.non_entity_endpoints
            or self.non_standard_edge_names
            or self.edge_type_map_violations
            or self.untyped_nodes
        )

    def hard_violation_count(self) -> int:
        return len(self.self_loops) + len(self.non_entity_endpoints)

    def soft_violation_count(self) -> int:
        return (
            len(self.non_standard_edge_names)
            + len(self.edge_type_map_violations)
            + len(self.untyped_nodes)
        )

    def summary_line(self) -> str:
        parts = [
            f"edges={self.total_edges_checked}",
            f"nodes={self.total_nodes_checked}",
        ]
        if self.self_loops:
            parts.append(f"self_loops={len(self.self_loops)}")
        if self.non_entity_endpoints:
            parts.append(f"non_entity_endpoints={len(self.non_entity_endpoints)}")
        if self.non_standard_edge_names:
            parts.append(f"non_standard_names={len(self.non_standard_edge_names)}")
        if self.edge_type_map_violations:
            parts.append(f"type_map_violations={len(self.edge_type_map_violations)}")
        if self.untyped_nodes:
            parts.append(f"untyped_nodes={len(self.untyped_nodes)}")
        return ", ".join(parts)


def _node_subtype(labels: Iterable[str]) -> str | None:
    """Return the entity subtype from a node's labels list, or None if untyped."""
    for label in labels:
        if label != "Entity" and not label.startswith("Entity_"):
            return label
    return None


def validate_episode_result(
    result: Any,
    *,
    log_audit: bool = True,
) -> ConformanceReport:
    """Validate a graphiti AddEpisodeResults object against the KG schema.

    Args:
        result: The AddEpisodeResults returned by graphiti.add_episode().
            Must expose .nodes (list[EntityNode]) and .edges (list[EntityEdge]).
        log_audit: If True, emit a [KG AUDIT] log line summarizing violations.

    Returns:
        A ConformanceReport. The caller decides whether to auto-fix the hard
        violations (see GraphMemoryService._apply_conformance_fixes).
    """
    standard_edge_names, standard_entity_types, edge_type_map = _load_schema()

    report = ConformanceReport()

    # Build a uuid → subtype lookup from the episode's resolved nodes so we can
    # check edge endpoints without going back to Neo4j. This is best-effort —
    # if an edge references a node that wasn't part of result.nodes, we can't
    # determine its type from this pass and skip the type-map check for it.
    nodes = list(getattr(result, "nodes", []) or [])
    edges = list(getattr(result, "edges", []) or [])
    report.total_nodes_checked = len(nodes)
    report.total_edges_checked = len(edges)

    uuid_to_subtype: dict[str, str | None] = {}
    uuid_to_name: dict[str, str | None] = {}
    for n in nodes:
        n_uuid = getattr(n, "uuid", None)
        if not n_uuid:
            continue
        labels = list(getattr(n, "labels", []) or [])
        subtype = _node_subtype(labels)
        uuid_to_subtype[n_uuid] = subtype
        uuid_to_name[n_uuid] = getattr(n, "name", None)
        if subtype is None:
            report.untyped_nodes.append({
                "uuid": n_uuid,
                "name": uuid_to_name[n_uuid],
                "labels": labels,
            })

    for e in edges:
        e_uuid = getattr(e, "uuid", None)
        e_name = getattr(e, "name", None)
        e_fact = getattr(e, "fact", None)
        src_uuid = getattr(e, "source_node_uuid", None)
        tgt_uuid = getattr(e, "target_node_uuid", None)

        # Self-loop check
        if src_uuid and tgt_uuid and src_uuid == tgt_uuid:
            report.self_loops.append({
                "edge_uuid": e_uuid,
                "node_uuid": src_uuid,
                "name": e_name,
                "fact": e_fact,
            })
            continue

        # Endpoint type check (the EntityEdge model only allows Entity endpoints,
        # but we still record this in case graphiti's runtime contract changes)
        # NOTE: we cannot check Episodic vs Entity here without a Neo4j query;
        # this slot is reserved for the daily scanner against the live graph.

        # Standard edge name check (None / NULL is acceptable graphiti fallback)
        if e_name is not None and e_name not in standard_edge_names:
            report.non_standard_edge_names.append({
                "edge_uuid": e_uuid,
                "name": e_name,
                "fact": e_fact,
            })

        # EDGE_TYPE_MAP conformance — only meaningful if both endpoints are
        # in the resolved-nodes set AND have a subtype label.
        src_type = uuid_to_subtype.get(src_uuid)
        tgt_type = uuid_to_subtype.get(tgt_uuid)
        if src_type and tgt_type and e_name is not None:
            allowed_for_pair = set(edge_type_map.get((src_type, tgt_type), []))
            allowed_wildcard = set(edge_type_map.get(("Entity", "Entity"), []))
            if e_name not in allowed_for_pair and e_name not in allowed_wildcard:
                report.edge_type_map_violations.append({
                    "edge_uuid": e_uuid,
                    "name": e_name,
                    "source_type": src_type,
                    "target_type": tgt_type,
                    "source_name": uuid_to_name.get(src_uuid),
                    "target_name": uuid_to_name.get(tgt_uuid),
                    "fact": e_fact,
                })

    if log_audit:
        episode = getattr(result, "episode", None)
        ep_name = getattr(episode, "name", "<unknown>") if episode else "<unknown>"
        ep_uuid = getattr(episode, "uuid", "<unknown>") if episode else "<unknown>"
        if report.is_clean():
            logger.info(
                "[KG CONFORMANCE] episode=%s uuid=%s clean (%s)",
                ep_name, ep_uuid[:8], report.summary_line(),
            )
        else:
            logger.warning(
                "[KG CONFORMANCE] episode=%s uuid=%s VIOLATIONS (%s)",
                ep_name, ep_uuid[:8], report.summary_line(),
            )
            for viol_name, items in [
                ("self_loop", report.self_loops),
                ("non_entity_endpoint", report.non_entity_endpoints),
                ("non_standard_edge_name", report.non_standard_edge_names),
                ("edge_type_map_violation", report.edge_type_map_violations),
                ("untyped_node", report.untyped_nodes),
            ]:
                for item in items:
                    logger.warning("  [%s] %s", viol_name, item)

    return report


# ── Auto-fix helper (used by GraphMemoryService) ──────────────────────────────

DELETE_EDGE_BY_UUID = """
MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
DELETE r
RETURN count(r) AS deleted
"""


async def apply_hard_fixes(
    driver,
    database: str,
    report: ConformanceReport,
) -> dict:
    """Delete edges that hit a hard violation (self-loop, non-entity endpoint).

    Args:
        driver: graphiti's neo4j AsyncDriver (from service._graphiti.driver.client).
            Must be an AsyncGraphDatabase driver.
        database: Neo4j database name
        report: A ConformanceReport returned by validate_episode_result.

    Returns:
        {"self_loops_deleted": int, "non_entity_endpoints_deleted": int}
    """
    stats = {"self_loops_deleted": 0, "non_entity_endpoints_deleted": 0}

    if not report.self_loops and not report.non_entity_endpoints:
        return stats

    async def _delete_one(session, uuid: str) -> int:
        result = await session.run(DELETE_EDGE_BY_UUID, uuid=uuid)
        record = await result.single()
        return int(record["deleted"]) if record else 0

    async with driver.session(database=database) as session:
        for item in report.self_loops:
            uuid = item.get("edge_uuid")
            if not uuid:
                continue
            try:
                stats["self_loops_deleted"] += await _delete_one(session, uuid)
            except Exception as exc:
                logger.error("[KG CONFORMANCE] failed to delete self-loop %s: %s", uuid, exc)

        for item in report.non_entity_endpoints:
            uuid = item.get("edge_uuid")
            if not uuid:
                continue
            try:
                stats["non_entity_endpoints_deleted"] += await _delete_one(session, uuid)
            except Exception as exc:
                logger.error("[KG CONFORMANCE] failed to delete bad-endpoint edge %s: %s", uuid, exc)

    if any(stats.values()):
        logger.warning("[KG CONFORMANCE] auto-fix: %s", stats)
    return stats
