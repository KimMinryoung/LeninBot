"""
graph_memory/structured_writer.py — direct typed-triple writes to the KG.

Bypasses graphiti's LLM extraction pipeline. Use this when an agent has a
specific (subject, predicate, object) fact to assert and wants it stored
deterministically. Free-text ingestion (news articles, reports) should still
go through GraphMemoryService.ingest_episode().

The flow per write_kg_structured() call is:

  1. Pre-validate every fact against the schema (entity types, predicates,
     EDGE_TYPE_MAP). Invalid facts are rejected individually; valid facts
     continue through the write path so agents can retry only failed items.
  2. For each fact, deterministically resolve the subject and object entities
     by exact (name, type) match against existing canonical nodes. If a node
     exists, reuse its uuid. If not, mint a new uuid + labels for it.
  3. Build a single synthetic Episodic node that holds the provenance footer
     for the entire batch. Every new edge references this episode via its
     `episodes` list, and every new/reused entity gets a MENTIONS edge from
     the synthetic episode.
  4. Hand everything to graphiti's add_nodes_and_edges_bulk() which writes
     embeddings + nodes + edges in one transaction.
  5. Run the conformance gate on the result (belt and suspenders — pre-check
     should already have caught violations, but the gate also picks up bugs
     in this writer itself).

Returns a structured summary of what was written.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

from .config import EDGE_TYPE_MAP
from .edges import EDGE_TYPES
from .entities import ENTITY_TYPES
from .conformance import validate_episode_result, apply_hard_fixes

logger = logging.getLogger(__name__)


# ── Pre-validation ────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = set(ENTITY_TYPES.keys())
VALID_PREDICATES = set(EDGE_TYPES.keys())
WILDCARD_ALLOWED = set(EDGE_TYPE_MAP.get(("Entity", "Entity"), []))


def _reject_fact(idx: int, fact: dict, reason: str) -> dict:
    """Build a retry-friendly rejected-fact record for tool callers."""
    return {
        "index": idx,
        "reason": reason,
        "fact": fact,
    }


def validate_fact(fact: dict, idx: int) -> str | None:
    """Return a violation message if `fact` is invalid, or None if it passes.

    Validates: required fields, type values, predicate value, and
    EDGE_TYPE_MAP conformance. Does not touch the database.
    """
    if not isinstance(fact, dict):
        return f"fact[{idx}] must be an object"

    required = ("subject_name", "subject_type", "predicate",
                "object_name", "object_type", "fact")
    for field in required:
        if not fact.get(field):
            return f"fact[{idx}] missing required field '{field}'"

    s_type = fact["subject_type"]
    t_type = fact["object_type"]
    pred = fact["predicate"]

    if s_type not in VALID_ENTITY_TYPES:
        return f"fact[{idx}] subject_type '{s_type}' not in {sorted(VALID_ENTITY_TYPES)}"
    if t_type not in VALID_ENTITY_TYPES:
        return f"fact[{idx}] object_type '{t_type}' not in {sorted(VALID_ENTITY_TYPES)}"
    if pred not in VALID_PREDICATES:
        return f"fact[{idx}] predicate '{pred}' not in {sorted(VALID_PREDICATES)}"

    allowed_for_pair = set(EDGE_TYPE_MAP.get((s_type, t_type), []))
    if pred not in allowed_for_pair and pred not in WILDCARD_ALLOWED:
        return (
            f"fact[{idx}] predicate '{pred}' not allowed for "
            f"({s_type} -> {t_type}). Allowed for this pair: "
            f"{sorted(allowed_for_pair) or 'none'}; wildcard: {sorted(WILDCARD_ALLOWED)}"
        )

    if fact.get("valid_at"):
        try:
            datetime.fromisoformat(str(fact["valid_at"]))
        except (ValueError, TypeError):
            return f"fact[{idx}] valid_at must be an ISO date or datetime"

    return None


# ── Entity resolution (deterministic, no LLM) ─────────────────────────────────

async def find_canonical_entity_uuid(
    driver_client, database: str, name: str, entity_type: str
) -> str | None:
    """Look up an existing entity by exact (name, type). If multiple, return
    the one with the most relationships. Returns None if no match."""
    async with driver_client.session(database=database) as session:
        result = await session.run(
            f"""
            MATCH (n:Entity {{name: $name}})
            WHERE '{entity_type}' IN labels(n)
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            WITH n, count(r) AS rels
            RETURN n.uuid AS uuid
            ORDER BY rels DESC
            LIMIT 1
            """,
            name=name,
        )
        record = await result.single()
        return record["uuid"] if record else None


# ── Build phase ───────────────────────────────────────────────────────────────

def _make_synthetic_episode(group_id: str, agent: str, mission_id: int | None,
                            facts_count: int, provenance_footer: str,
                            trust_tier: str) -> EpisodicNode:
    ts = datetime.now(timezone.utc)
    label = f"structured-{ts.strftime('%Y%m%d%H%M%S')}-{agent}"
    if mission_id:
        label += f"-m{mission_id}"
    name = f"[T:{trust_tier}]{label}"
    body = (
        f"structured assertion of {facts_count} fact(s) by agent={agent}\n\n"
        + provenance_footer
    )
    return EpisodicNode(
        name=name,
        group_id=group_id,
        source=EpisodeType.text,
        source_description=f"agent_structured_write ({agent})",
        content=body,
        valid_at=ts,
        entity_edges=[],  # filled in later
    )


def _make_entity_node(name: str, entity_type: str, group_id: str,
                      existing_uuid: str | None) -> EntityNode:
    """Build an EntityNode. If existing_uuid is set, reuse it (the bulk
    save uses MERGE on uuid so the existing node is preserved/updated)."""
    return EntityNode(
        uuid=existing_uuid or str(uuid4()),
        name=name,
        group_id=group_id,
        labels=["Entity", entity_type],
        summary="",
        created_at=datetime.now(timezone.utc),
    )


def _make_entity_edge(source_uuid: str, target_uuid: str, predicate: str,
                      fact_text: str, group_id: str,
                      valid_at: datetime | None,
                      episode_uuid: str) -> EntityEdge:
    return EntityEdge(
        uuid=str(uuid4()),
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name=predicate,
        fact=fact_text,
        group_id=group_id,
        created_at=datetime.now(timezone.utc),
        valid_at=valid_at,
        invalid_at=None,
        episodes=[episode_uuid],
        attributes={},
    )


def _make_mentions_edge(episode_uuid: str, entity_uuid: str,
                        group_id: str) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=str(uuid4()),
        source_node_uuid=episode_uuid,
        target_node_uuid=entity_uuid,
        group_id=group_id,
        created_at=datetime.now(timezone.utc),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

async def write_structured_facts(
    graphiti,
    facts: list[dict],
    *,
    group_id: str,
    agent: str = "agent",
    mission_id: int | None = None,
    trust_tier: str = "unverified",
    provenance_footer: str = "",
) -> dict:
    """Write a batch of structured facts to the KG.

    Args:
        graphiti: An initialized graphiti_core.Graphiti instance.
        facts: List of fact dicts. Required keys per fact: subject_name,
            subject_type, predicate, object_name, object_type, fact.
            Optional: valid_at (ISO date string).
        group_id: KG group ID for the episode and all created edges.
        agent: Caller's agent name (for provenance).
        mission_id: Optional mission this batch belongs to.
        trust_tier: anchor / corroborated / single / unverified.
        provenance_footer: Pre-built provenance text (sources etc.) — written
            into the synthetic episode body verbatim.

    Returns:
        {
          "status": "ok"|"partial_success"|"error",
          "message": str,
          "facts_written": int,
          "facts_rejected": int,
          "written_fact_indices": list[int],
          "rejected_facts": list[{"index": int, "reason": str, "fact": dict}],
          "episode_name": str,
          "violations": dict (from conformance gate, may be empty),
        }
    """
    if not facts:
        return {"status": "error", "message": "no facts provided"}

    # ── 1. Pre-validate facts independently ──────────────────────────────
    valid_facts: list[dict] = []
    written_fact_indices: list[int] = []
    rejected_facts: list[dict] = []
    for i, f in enumerate(facts):
        msg = validate_fact(f, i)
        if msg:
            rejected_facts.append(_reject_fact(i, f, msg))
        else:
            valid_facts.append(f)
            written_fact_indices.append(i)

    if not valid_facts:
        msg = (
            f"validation failed for all {len(facts)} fact(s); "
            "no facts written. Retry only the rejected_facts entries after fixing them."
        )
        return {
            "status": "error",
            "message": msg,
            "facts_written": 0,
            "facts_rejected": len(rejected_facts),
            "written_fact_indices": [],
            "rejected_facts": rejected_facts,
        }

    driver_client = graphiti.driver.client
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    # ── 2. Resolve subject + object entities (deterministic) ─────────────
    entity_lookup_cache: dict[tuple[str, str], str] = {}  # (name, type) -> uuid

    async def get_or_assign(name: str, etype: str) -> tuple[str, bool]:
        """Return (uuid, is_new). Reuses existing canonical node by name+type."""
        key = (name, etype)
        if key in entity_lookup_cache:
            return entity_lookup_cache[key], False
        existing = await find_canonical_entity_uuid(driver_client, database, name, etype)
        if existing:
            entity_lookup_cache[key] = existing
            return existing, False
        new_uuid = str(uuid4())
        entity_lookup_cache[key] = new_uuid
        return new_uuid, True

    resolved = []  # list of (fact, src_uuid, src_is_new, tgt_uuid, tgt_is_new)
    for f in valid_facts:
        s_uuid, s_new = await get_or_assign(f["subject_name"], f["subject_type"])
        t_uuid, t_new = await get_or_assign(f["object_name"], f["object_type"])
        resolved.append((f, s_uuid, s_new, t_uuid, t_new))

    # ── 3. Build synthetic episode + entity/edge/mentions objects ────────
    episode = _make_synthetic_episode(
        group_id=group_id, agent=agent, mission_id=mission_id,
        facts_count=len(valid_facts), provenance_footer=provenance_footer,
        trust_tier=trust_tier,
    )

    # Build entity nodes — only NEW entities need to be passed (existing ones
    # remain untouched). add_nodes_and_edges_bulk uses MERGE on uuid, so even
    # passing existing nodes is safe, but it would generate redundant
    # embeddings. Skip existing.
    new_entity_nodes: list[EntityNode] = []
    seen_new_uuids: set[str] = set()
    for f, s_uuid, s_new, t_uuid, t_new in resolved:
        if s_new and s_uuid not in seen_new_uuids:
            seen_new_uuids.add(s_uuid)
            new_entity_nodes.append(
                _make_entity_node(f["subject_name"], f["subject_type"], group_id, s_uuid)
            )
        if t_new and t_uuid not in seen_new_uuids:
            seen_new_uuids.add(t_uuid)
            new_entity_nodes.append(
                _make_entity_node(f["object_name"], f["object_type"], group_id, t_uuid)
            )

    # Build edges
    entity_edges: list[EntityEdge] = []
    for f, s_uuid, _s_new, t_uuid, _t_new in resolved:
        valid_at = None
        if f.get("valid_at"):
            try:
                valid_at = datetime.fromisoformat(f["valid_at"]).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                valid_at = None
        entity_edges.append(
            _make_entity_edge(
                source_uuid=s_uuid,
                target_uuid=t_uuid,
                predicate=f["predicate"],
                fact_text=f["fact"],
                group_id=group_id,
                valid_at=valid_at,
                episode_uuid=episode.uuid,
            )
        )

    # Build MENTIONS edges from synthetic episode to every entity touched
    # (both new and reused). The episode "knows about" all entities it asserts
    # facts on, even if those entities were already in the graph.
    touched_uuids: set[str] = set()
    for _f, s_uuid, _sn, t_uuid, _tn in resolved:
        touched_uuids.add(s_uuid)
        touched_uuids.add(t_uuid)
    episodic_edges: list[EpisodicEdge] = [
        _make_mentions_edge(episode.uuid, eu, group_id) for eu in touched_uuids
    ]

    # Track entity_edges on the episode object so it survives serialization
    episode.entity_edges = [e.uuid for e in entity_edges]

    # ── 4. Bulk save ─────────────────────────────────────────────────────
    try:
        await add_nodes_and_edges_bulk(
            graphiti.driver,
            [episode],
            episodic_edges,
            new_entity_nodes,
            entity_edges,
            graphiti.embedder,
        )
    except Exception as exc:
        logger.error("[KG STRUCTURED] bulk save failed: %s", exc)
        return {
            "status": "error",
            "message": f"bulk save failed: {exc}",
            "facts_written": 0,
            "facts_rejected": len(rejected_facts),
            "written_fact_indices": [],
            "rejected_facts": rejected_facts,
        }

    # ── 5. Conformance gate (defensive) ──────────────────────────────────
    # Build a result-like object the validator can read.
    from types import SimpleNamespace
    fake_result = SimpleNamespace(
        episode=episode,
        nodes=new_entity_nodes,
        edges=entity_edges,
    )
    try:
        report = validate_episode_result(fake_result, log_audit=True)
        if report.hard_violation_count() > 0:
            await apply_hard_fixes(driver_client, database, report)
    except Exception as exc:
        logger.error("[KG STRUCTURED] conformance check failed (non-fatal): %s", exc)
        report = None

    new_count = len(new_entity_nodes)
    reused_count = len(touched_uuids) - new_count
    status = "partial_success" if rejected_facts else "ok"
    msg = (
        f"wrote {len(entity_edges)} fact(s) — "
        f"{new_count} new entity(ies), {reused_count} reused, "
        f"episode={episode.name}"
    )
    if rejected_facts:
        msg += (
            f" | rejected {len(rejected_facts)} invalid fact(s); "
            "retry only rejected_facts after fixing schema errors"
        )
    if report and not report.is_clean():
        msg += f" | conformance: {report.summary_line()}"
    logger.info(
        "[KG STRUCTURED] %s | agent=%s | mission=%s | tier=%s",
        msg, agent, mission_id, trust_tier,
    )
    return {
        "status": status,
        "message": msg,
        "facts_written": len(entity_edges),
        "facts_rejected": len(rejected_facts),
        "written_fact_indices": written_fact_indices,
        "rejected_facts": rejected_facts,
        "new_entities": new_count,
        "reused_entities": reused_count,
        "episode_name": episode.name,
        "violations": report.summary_line() if report else "",
    }
