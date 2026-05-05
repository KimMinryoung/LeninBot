"""Knowledge graph write helpers."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from kg_runtime.service_runtime import get_kg_service, reset_kg_service, run_kg_task

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

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
        if any(k in err_str for k in ("dns", "connection", "timeout", "unavailable", "graphiti")):
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
        )
    except Exception as e:
        logger.error("[shared] add_kg_structured error: %s", e)
        err_str = str(e).lower()
        if any(k in err_str for k in ("dns", "connection", "timeout", "unavailable", "graphiti")):
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
    )


