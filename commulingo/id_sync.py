"""Follow CommuLingo id renames in leninbot's own work state.

CommuLingo (the frontend) owns people and terms and retires a renamed id into
commulingo_id_redirects. It does not reach into leninbot's tables, so open
curation gaps and pipeline jobs that still name a retired id are moved here,
from the admin MCP's id_redirects_list. Finished rows keep the id they were
recorded under. Idempotent; cheap enough to run at the start of every worker.
"""
from __future__ import annotations

import logging

from commulingo.mcp_client import call_tool
from db import get_conn

logger = logging.getLogger(__name__)

KINDS = ("person", "term")
ACTIVE_JOBS = ("ready", "running", "deferred", "escalated")


def _redirects(entity_type: str) -> list[tuple[str, str]]:
    pairs, offset = [], 0
    while True:
        page = call_tool("id_redirects_list", {"entityType": entity_type, "limit": 100, "offset": offset})
        pairs += [(row["from_id"], row["to_id"]) for row in page["items"]]
        offset += len(page["items"])
        if not page["items"] or offset >= page["total"]:
            return pairs


def follow_renames() -> dict:
    """Retarget open gaps and jobs from retired ids; returns rows moved per table."""
    moved = {"commulingo_curation_gaps": 0, "commulingo_pipeline_jobs": 0}
    with get_conn() as conn, conn.cursor() as cur:
        for kind in KINDS:
            for old, new in _redirects(kind):
                cur.execute("""UPDATE commulingo_curation_gaps SET target_id = %s, updated_at = NOW()
                                WHERE kind = %s AND target_id = %s AND status IN ('pending', 'claimed')""",
                            (new, kind, old))
                moved["commulingo_curation_gaps"] += cur.rowcount
                # One active job per (kind, target, topic): an old-id job whose
                # topic the new id already has open is a duplicate, so it is
                # cancelled rather than moved onto the unique index.
                cur.execute("""UPDATE commulingo_pipeline_jobs j SET status = 'cancelled'
                                WHERE j.kind = %s AND j.target = %s AND j.status IN %s
                                  AND EXISTS (SELECT 1 FROM commulingo_pipeline_jobs n
                                               WHERE n.kind = j.kind AND n.target = %s AND n.topic = j.topic
                                                 AND n.status IN %s)""",
                            (kind, old, ACTIVE_JOBS, new, ACTIVE_JOBS))
                cur.execute("""UPDATE commulingo_pipeline_jobs SET target = %s
                                WHERE kind = %s AND target = %s AND status NOT IN ('complete', 'cancelled')""",
                            (new, kind, old))
                moved["commulingo_pipeline_jobs"] += cur.rowcount
    if any(moved.values()):
        logger.info("followed CommuLingo renames: %s", moved)
    return moved


def follow_renames_best_effort() -> dict | None:
    """For worker start-up: a CommuLingo outage must not stop unrelated work."""
    try:
        return follow_renames()
    except Exception as exc:  # noqa: BLE001 - logged and skipped by design
        logger.warning("could not follow CommuLingo renames: %s", exc)
        return None
