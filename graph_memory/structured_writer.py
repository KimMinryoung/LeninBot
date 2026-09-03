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

from .config import EDGE_TYPE_MAP, SYNC_ONLY_ENTITY_TYPES, SYNC_ONLY_PREDICATES, sync_predicate_allowed
from .edges import EDGE_TYPES
from .entities import ENTITY_TYPES
from .conformance import validate_episode_result, apply_hard_fixes
from kg_runtime.identity import (
    build_identity_props,
    resolve_entity_async,
    upsert_identity_async,
)

logger = logging.getLogger(__name__)


# ── Pre-validation ────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = set(ENTITY_TYPES.keys())
VALID_PREDICATES = set(EDGE_TYPES.keys())
WILDCARD_ALLOWED = set(EDGE_TYPE_MAP.get(("Entity", "Entity"), []))

# Per-fact identity hints. Only sync jobs set these (the agent tool schema does
# not expose them); they let a writer attach stable external ids / aliases /
# a deterministic summary to the entities it touches.
IDENTITY_FACT_FIELDS = (
    "subject_external_id", "object_external_id",
    "subject_aliases", "object_aliases",
    "subject_summary", "object_summary",
    "subject_name_ko", "subject_name_en", "object_name_ko", "object_name_en",
)


def _reject_fact(idx: int, fact: dict, reason: str) -> dict:
    """Build a retry-friendly rejected-fact record for tool callers."""
    return {
        "index": idx,
        "reason": reason,
        "fact": fact,
    }


def validate_fact(fact: dict, idx: int, *, allow_sync_predicates: bool = False) -> str | None:
    """Return a violation message if `fact` is invalid, or None if it passes.

    Validates: required fields, type values, predicate value, and
    EDGE_TYPE_MAP conformance. Does not touch the database.

    ``allow_sync_predicates`` unlocks the sync-only schema (``Document`` nodes,
    ``Reference`` edges) for deterministic mirror jobs; agent writes never
    pass it, so those types stay out of the tool surface.
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

    # Common nouns and schema labels are not entities (2026-09-03: 국가/개인/
    # 경찰/Organization nodes from the first document extraction). Lazy import:
    # kg_runtime.identity imports this module's schema helpers.
    from kg_runtime.identity import is_generic_entity_name
    for side in ("subject_name", "object_name"):
        if is_generic_entity_name(fact[side]):
            return (f"fact[{idx}] {side} '{fact[side]}' is a generic noun or type label, not an entity; "
                    "name the concrete actor (person, organization, place, titled work) or drop the fact")

    agent_entity_types = VALID_ENTITY_TYPES if allow_sync_predicates else VALID_ENTITY_TYPES - SYNC_ONLY_ENTITY_TYPES
    agent_predicates = VALID_PREDICATES if allow_sync_predicates else VALID_PREDICATES - SYNC_ONLY_PREDICATES

    if s_type not in agent_entity_types:
        return f"fact[{idx}] subject_type '{s_type}' not in {sorted(agent_entity_types)}"
    if t_type not in agent_entity_types:
        return f"fact[{idx}] object_type '{t_type}' not in {sorted(agent_entity_types)}"
    if pred not in agent_predicates:
        return f"fact[{idx}] predicate '{pred}' not in {sorted(agent_predicates)}"
    if fact.get("attributes") is not None and not isinstance(fact.get("attributes"), dict):
        return f"fact[{idx}] attributes must be an object"

    allowed_for_pair = set(EDGE_TYPE_MAP.get((s_type, t_type), []))
    if pred in SYNC_ONLY_PREDICATES:
        if not sync_predicate_allowed(s_type, t_type, pred):
            return f"fact[{idx}] predicate '{pred}' not allowed for ({s_type} -> {t_type})"
    elif pred not in allowed_for_pair and pred not in WILDCARD_ALLOWED:
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
    driver_client, database: str, name: str, entity_type: str,
    *, external_id: str | None = None, aliases=(), embedder=None,
) -> str | None:
    """Resolve an existing entity through the identity layer
    (``kg_runtime.identity.resolve_entity_async``): external id, then
    normalized alias/name match across group_ids (same label required), then
    the optional name-embedding step. Returns None if no match."""
    async with driver_client.session(database=database) as session:
        hit = await resolve_entity_async(
            session,
            name=name,
            entity_type=entity_type,
            external_id=external_id,
            aliases=aliases,
            embedder=embedder,
        )
        return hit.uuid


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
                      existing_uuid: str | None,
                      identity: dict | None = None,
                      summary: str = "") -> EntityNode:
    """Build an EntityNode. If existing_uuid is set, reuse it (the bulk
    save uses MERGE on uuid so the existing node is preserved/updated).
    ``identity`` (external_ids / aliases / alias_keys / …) is stored through
    ``attributes`` so graphiti writes it in the same bulk save."""
    return EntityNode(
        uuid=existing_uuid or str(uuid4()),
        name=name,
        group_id=group_id,
        labels=["Entity", entity_type],
        summary=summary or "",
        created_at=datetime.now(timezone.utc),
        attributes=dict(identity or {}),
    )


def _make_entity_edge(source_uuid: str, target_uuid: str, predicate: str,
                      fact_text: str, group_id: str,
                      valid_at: datetime | None,
                      episode_uuid: str,
                      attributes: dict | None = None,
                      invalid_at: datetime | None = None) -> EntityEdge:
    return EntityEdge(
        uuid=str(uuid4()),
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name=predicate,
        fact=fact_text,
        group_id=group_id,
        created_at=datetime.now(timezone.utc),
        valid_at=valid_at,
        invalid_at=invalid_at,
        episodes=[episode_uuid],
        attributes=dict(attributes or {}),
    )


def _parse_iso_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    # naive일 때만 UTC 부여 — 오프셋이 이미 있는 입력을 replace로
    # 덮으면 실제 시각이 틀어진다.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _side(fact: dict, side: str) -> dict:
    """Identity hints for one side ('subject' | 'object') of a fact."""
    aliases = fact.get(f"{side}_aliases") or ()
    if isinstance(aliases, str):
        aliases = [aliases]
    return {
        "name": fact[f"{side}_name"],
        "type": fact[f"{side}_type"],
        "external_id": fact.get(f"{side}_external_id") or None,
        "aliases": [a for a in aliases if a],
        "summary": fact.get(f"{side}_summary") or "",
        "name_ko": fact.get(f"{side}_name_ko") or None,
        "name_en": fact.get(f"{side}_name_en") or None,
    }


def _make_mentions_edge(episode_uuid: str, entity_uuid: str,
                        group_id: str) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=str(uuid4()),
        source_node_uuid=episode_uuid,
        target_node_uuid=entity_uuid,
        group_id=group_id,
        created_at=datetime.now(timezone.utc),
    )


EMBED_BATCH_SIZE = 50


async def _embed_in_batches(embedder, nodes: list[EntityNode], edges: list[EntityEdge]) -> None:
    """Fill name/fact embeddings with batched embedder calls (chunks of
    EMBED_BATCH_SIZE) so the bulk save skips its per-item requests."""
    if embedder is None:
        return
    from graphiti_core.nodes import create_entity_node_embeddings
    from graphiti_core.edges import create_entity_edge_embeddings

    pending_nodes = [n for n in nodes if n.name_embedding is None and n.name]
    for i in range(0, len(pending_nodes), EMBED_BATCH_SIZE):
        await create_entity_node_embeddings(embedder, pending_nodes[i:i + EMBED_BATCH_SIZE])
    pending_edges = [e for e in edges if e.fact_embedding is None and e.fact]
    for i in range(0, len(pending_edges), EMBED_BATCH_SIZE):
        await create_entity_edge_embeddings(embedder, pending_edges[i:i + EMBED_BATCH_SIZE])


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
    allow_sync_predicates: bool = False,
) -> dict:
    """Write a batch of structured facts to the KG.

    Args:
        graphiti: An initialized graphiti_core.Graphiti instance.
        facts: List of fact dicts. Required keys per fact: subject_name,
            subject_type, predicate, object_name, object_type, fact.
            Optional: valid_at / invalid_at (ISO date string), attributes
            (dict stored on the edge), and the identity hints in
            IDENTITY_FACT_FIELDS (sync jobs only).
        group_id: KG group ID for the episode and all created edges.
        agent: Caller's agent name (for provenance).
        mission_id: Optional mission this batch belongs to.
        trust_tier: anchor / corroborated / single / unverified.
        provenance_footer: Pre-built provenance text (sources etc.) — written
            into the synthetic episode body verbatim.
        allow_sync_predicates: unlock Document / Reference for mirror jobs.

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
        msg = validate_fact(f, i, allow_sync_predicates=allow_sync_predicates)
        if msg:
            rejected_facts.append(_reject_fact(i, f, msg))
        else:
            from graph_memory.graphiti_patches import normalize_entity_names_in_text
            nf = dict(f)
            for key in ("subject_name", "object_name", "fact"):
                if isinstance(nf.get(key), str):
                    nf[key] = normalize_entity_names_in_text(nf[key])
            valid_facts.append(nf)
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
    # uuid -> identity hints to attach (new nodes: at build time; reused
    # nodes: via upsert_identity after the bulk save)
    identity_hints: dict[str, dict] = {}
    new_uuids: set[str] = set()
    embedder = getattr(graphiti, "embedder", None)

    def _remember_hints(uuid: str, side: dict) -> None:
        h = identity_hints.setdefault(
            uuid, {"external_ids": [], "aliases": [], "summary": "", "name_ko": None, "name_en": None,
                   "name": side["name"]},
        )
        if side["external_id"] and side["external_id"] not in h["external_ids"]:
            h["external_ids"].append(side["external_id"])
        for a in side["aliases"]:
            if a and a != side["name"] and a not in h["aliases"]:
                h["aliases"].append(a)
        if side["summary"] and not h["summary"]:
            h["summary"] = side["summary"]
        h["name_ko"] = h["name_ko"] or side["name_ko"]
        h["name_en"] = h["name_en"] or side["name_en"]

    async def get_or_assign(side: dict) -> tuple[str, bool]:
        """Return (uuid, is_new). Reuses an existing node through the identity
        resolver (external id → alias/name across groups → optional NN)."""
        name, etype = side["name"], side["type"]
        key = (side["external_id"] or name, etype)
        if key in entity_lookup_cache:
            uuid = entity_lookup_cache[key]
            _remember_hints(uuid, side)
            return uuid, uuid in new_uuids
        existing = await find_canonical_entity_uuid(
            driver_client, database, name, etype,
            external_id=side["external_id"], aliases=side["aliases"], embedder=embedder,
        )
        if existing:
            entity_lookup_cache[key] = existing
            _remember_hints(existing, side)
            return existing, False
        new_uuid = str(uuid4())
        entity_lookup_cache[key] = new_uuid
        new_uuids.add(new_uuid)
        _remember_hints(new_uuid, side)
        return new_uuid, True

    resolved = []  # list of (fact, src_uuid, src_is_new, tgt_uuid, tgt_is_new)
    for f in valid_facts:
        s_uuid, s_new = await get_or_assign(_side(f, "subject"))
        t_uuid, t_new = await get_or_assign(_side(f, "object"))
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

    def _new_node(name: str, etype: str, uuid: str) -> EntityNode:
        hints = identity_hints.get(uuid, {})
        identity = build_identity_props(
            name,
            aliases=hints.get("aliases", ()),
            external_ids=hints.get("external_ids", ()),
            name_ko=hints.get("name_ko"),
            name_en=hints.get("name_en"),
        )
        return _make_entity_node(name, etype, group_id, uuid,
                                 identity=identity, summary=hints.get("summary", ""))

    for f, s_uuid, s_new, t_uuid, t_new in resolved:
        if s_new and s_uuid not in seen_new_uuids:
            seen_new_uuids.add(s_uuid)
            new_entity_nodes.append(_new_node(f["subject_name"], f["subject_type"], s_uuid))
        if t_new and t_uuid not in seen_new_uuids:
            seen_new_uuids.add(t_uuid)
            new_entity_nodes.append(_new_node(f["object_name"], f["object_type"], t_uuid))

    # Build edges
    entity_edges: list[EntityEdge] = []
    for f, s_uuid, _s_new, t_uuid, _t_new in resolved:
        entity_edges.append(
            _make_entity_edge(
                source_uuid=s_uuid,
                target_uuid=t_uuid,
                predicate=f["predicate"],
                fact_text=f["fact"],
                group_id=group_id,
                valid_at=_parse_iso_datetime(f.get("valid_at")),
                invalid_at=_parse_iso_datetime(f.get("invalid_at")),
                episode_uuid=episode.uuid,
                attributes=f.get("attributes") or {},
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
    # add_nodes_and_edges_bulk embeds one node/edge per request when the
    # embedding is missing; batch them first so a 200-fact sync call costs a
    # handful of embedding requests instead of hundreds.
    try:
        await _embed_in_batches(graphiti.embedder, new_entity_nodes, entity_edges)
    except Exception as exc:
        logger.warning("[KG STRUCTURED] batch embedding failed, falling back to per-item: %s", exc)
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

    # ── 4b. Attach identity hints to REUSED nodes ────────────────────────
    # New nodes carried their identity in attributes; existing nodes were not
    # re-saved (that would re-embed them), so union ids/aliases in place.
    identity_updates = 0
    for uuid, hints in identity_hints.items():
        if uuid in seen_new_uuids:
            continue
        if not (hints["external_ids"] or hints["aliases"] or hints["summary"]
                or hints["name_ko"] or hints["name_en"]):
            continue
        try:
            async with driver_client.session(database=database) as session:
                await upsert_identity_async(
                    session, uuid,
                    external_ids=hints["external_ids"], aliases=hints["aliases"],
                    name_ko=hints["name_ko"], name_en=hints["name_en"],
                    summary=hints["summary"] or None, name=hints.get("name"),
                )
            identity_updates += 1
        except Exception as exc:
            logger.warning("[KG STRUCTURED] identity upsert failed for %s: %s", uuid[:8], exc)

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
        "identity_updates": identity_updates,
        "entity_uuids": {k[0]: v for k, v in entity_lookup_cache.items()},
        "episode_name": episode.name,
        "violations": report.summary_line() if report else "",
    }
