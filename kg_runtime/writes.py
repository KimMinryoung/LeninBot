"""Knowledge graph write helpers."""

import asyncio
import logging
from datetime import datetime, timezone

from kg_runtime.service_runtime import get_kg_service, reset_kg_service, run_kg_task

logger = logging.getLogger(__name__)
from shared import KST

def add_kg_episode(
    content: str,
    name: str = "",
    source_type: str = "internal_report",
    group_id: str = "agent_knowledge",
    *,
    trust_tier: str = "unverified",
    provenance_footer: str = "",
) -> dict:
    """Add an episode to the Knowledge Graph (sync version — for scripts/cron).

    For use inside an asyncio event loop (e.g. telegram bot), use add_kg_episode_async() instead
    to avoid 'Cannot run the event loop while another loop is running' errors.

    Returns dict with 'status' ('ok' or 'error') and 'message'.
    """
    from datetime import timezone

    svc = get_kg_service()
    if svc is None:
        return {"status": "error", "message": "Knowledge Graph service unavailable"}

    if not content or not content.strip():
        return {"status": "error", "message": "Content cannot be empty"}

    # Encode trust tier into the episode name as a stable prefix so the
    # search side can show it without an extra metadata table.
    if trust_tier not in ("anchor", "corroborated", "single", "unverified"):
        trust_tier = "unverified"
    if not name:
        ts = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
        name = f"agent-note-{ts}"
    if not name.startswith("[T:"):
        name = f"[T:{trust_tier}]{name}"

    body = content.strip()
    if provenance_footer:
        body = body + "\n\n" + provenance_footer.strip()

    try:
        run_kg_task(
            svc.ingest_episode,
            name=name,
            body=body,
            source_type=source_type,
            reference_time=datetime.now(timezone.utc),
            group_id=group_id,
            preprocess_news=False,
            max_body_chars=3500,
        )
        return {"status": "ok", "message": f"Episode '{name}' added to group '{group_id}'"}
    except Exception as e:
        logger.error("[shared] add_kg_episode error: %s", e)
        err_str = str(e).lower()
        if any(k in err_str for k in ("dns", "connection", "timeout", "unavailable")):
            reset_kg_service()
        return {"status": "error", "message": str(e)}


async def add_kg_episode_async(
    content: str,
    name: str = "",
    source_type: str = "internal_report",
    group_id: str = "agent_knowledge",
    *,
    trust_tier: str = "unverified",
    provenance_footer: str = "",
) -> dict:
    """Add an episode to the Knowledge Graph from async callers.

    Important: Graphiti/Neo4j objects are bound to the dedicated KG loop thread
    created in shared.py. Async callers must therefore hop to that loop via
    run_kg_async() in a worker thread instead of awaiting svc.ingest_episode()
    on the caller's own event loop.
    """
    return await asyncio.to_thread(
        add_kg_episode,
        content,
        name,
        source_type,
        group_id,
        trust_tier=trust_tier,
        provenance_footer=provenance_footer,
    )


def add_kg_structured(
    facts: list[dict],
    *,
    group_id: str = "agent_knowledge",
    agent: str = "agent",
    mission_id: int | None = None,
    trust_tier: str = "unverified",
    provenance_footer: str = "",
    allow_sync_predicates: bool = False,
    cross_script_guard: bool = False,
) -> dict:
    """Write structured facts to the KG (sync — for scripts/cron).

    See graph_memory.structured_writer.write_structured_facts for details.
    Runs on the dedicated KG event loop to avoid cross-loop contamination.
    """
    svc = get_kg_service()
    if svc is None:
        return {"status": "error", "message": "Knowledge Graph service unavailable"}

    try:
        from graph_memory.structured_writer import write_structured_facts
        return run_kg_task(
            write_structured_facts,
            svc.graphiti,
            facts,
            group_id=group_id,
            agent=agent,
            mission_id=mission_id,
            trust_tier=trust_tier,
            provenance_footer=provenance_footer,
            allow_sync_predicates=allow_sync_predicates,
            cross_script_guard=cross_script_guard,
        )
    except Exception as e:
        logger.error("[shared] add_kg_structured error: %s", e)
        err_str = str(e).lower()
        if any(k in err_str for k in ("dns", "connection", "timeout", "unavailable")):
            reset_kg_service()
        return {"status": "error", "message": str(e)}


async def add_kg_structured_async(
    facts: list[dict],
    *,
    group_id: str = "agent_knowledge",
    agent: str = "agent",
    mission_id: int | None = None,
    trust_tier: str = "unverified",
    provenance_footer: str = "",
    cross_script_guard: bool = False,
) -> dict:
    """Async wrapper around add_kg_structured. Hops to the KG loop via
    asyncio.to_thread for the same reasons as add_kg_episode_async."""
    return await asyncio.to_thread(
        add_kg_structured,
        facts,
        group_id=group_id,
        agent=agent,
        mission_id=mission_id,
        trust_tier=trust_tier,
        provenance_footer=provenance_footer,
        cross_script_guard=cross_script_guard,
    )




# ── Fact retraction (agents) ──────────────────────────────────────────────────
#
# The KG is append-only: a wrong fact is expired (expired_at + reason), never
# deleted, and stays visible to knowledge_graph_search(include_expired=True).
# Before 2026-10-06 agents could only add a correction next to a wrong fact,
# so both stayed active side by side.

RETRACT_MIN_REASON = 15

CYPHER_RETRACT_CANDIDATES = """
MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
WHERE r.expired_at IS NULL AND r.name = $predicate
  AND (s.name = $subject OR $subject_key IN coalesce(s.alias_keys, []))
  AND (o.name = $object OR $object_key IN coalesce(o.alias_keys, []))
  AND ($edge_id = '' OR r.uuid STARTS WITH $edge_id)
RETURN r.uuid AS uuid, s.name AS subject, o.name AS object, r.fact AS fact,
       r.sync_key AS sync_key, r.valid_at AS valid_at
ORDER BY r.created_at
LIMIT 20
"""

CYPHER_RETRACT = """
MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
WHERE r.expired_at IS NULL AND r.sync_key IS NULL
SET r.expired_at = datetime(), r.retracted_reason = $reason,
    r.retracted_by = $agent, r.retracted_task = $task
RETURN r.uuid AS uuid
"""


def retract_kg_fact(
    *,
    subject_name: str,
    predicate: str,
    object_name: str,
    reason: str,
    edge_id: str = "",
    agent: str = "agent",
    task: str | None = None,
) -> dict:
    """Expire one active agent-written fact. Returns {status, message, ...}.

    status: retracted | ambiguous (candidates listed, call again with edge_id)
    | not_found | refused | error.
    """
    from kg_runtime.identity import normalize_alias_key
    from kg_runtime.search import _get_neo4j_sync_driver

    reason = (reason or "").strip()
    if len(reason) < RETRACT_MIN_REASON:
        return {"status": "refused", "message": f"reason must say why the fact is wrong (at least {RETRACT_MIN_REASON} characters, with the source that contradicts it)"}
    edge_id = (edge_id or "").strip()
    if edge_id and len(edge_id) < 8:
        return {"status": "refused", "message": "edge_id needs at least its first 8 characters"}
    params = {
        "subject": subject_name, "subject_key": normalize_alias_key(subject_name),
        "object": object_name, "object_key": normalize_alias_key(object_name),
        "predicate": predicate, "edge_id": edge_id,
    }
    try:
        with _get_neo4j_sync_driver() as (driver, database):
            with driver.session(database=database) as session:
                rows = [dict(r) for r in session.run(CYPHER_RETRACT_CANDIDATES, **params)]
                if not rows:
                    return {"status": "not_found", "message": (
                        f"no active fact '{subject_name} —{predicate}→ {object_name}'. Use the names exactly as "
                        "knowledge_graph_search shows them; the fact may already be expired.")}
                candidates = [{"edge_id": r["uuid"][:8], "fact": (r["fact"] or "")[:200],
                               "subject": r["subject"], "object": r["object"],
                               "source_owned": bool(r["sync_key"])} for r in rows]
                if len(rows) > 1:
                    return {"status": "ambiguous", "candidates": candidates, "message": (
                        f"{len(rows)} active facts match; call again with edge_id of the wrong one")}
                row = rows[0]
                if row["sync_key"]:
                    return {"status": "refused", "candidates": candidates, "message": (
                        "this fact is mirrored from a source record (CommuLingo or a published document) and the "
                        "next sync would restore it; report the source error instead")}
                done = session.run(CYPHER_RETRACT, uuid=row["uuid"], reason=reason, agent=agent,
                                   task=str(task) if task is not None else None).single()
    except Exception as exc:
        logger.error("[KG retract] failed: %s", exc)
        return {"status": "error", "message": f"retraction failed: {exc}"}
    if not done:
        return {"status": "error", "message": "fact changed while retracting; search again"}
    logger.info("[KG AUDIT] retract | edge=%s | agent=%s | reason=%s", row["uuid"][:8], agent, reason[:200])
    return {"status": "retracted", "edge_id": row["uuid"][:8], "fact": candidates[0]["fact"],
            "message": "fact expired; it no longer appears in default search"}
