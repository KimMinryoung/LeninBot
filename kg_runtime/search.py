"""Knowledge graph direct stats and formatted search helpers."""

import logging
import os
from contextlib import contextmanager

from kg_runtime.service_runtime import get_kg_service, reset_kg_service, run_kg_task
from secrets_loader import get_secret

logger = logging.getLogger(__name__)

@contextmanager
def _get_neo4j_sync_driver():
    """Create a lightweight sync Neo4j driver for direct Cypher queries.

    Does NOT trigger Graphiti async init — avoids 'no running event loop' errors.
    Yields (driver, database_name). Driver is automatically closed on exit.
    """
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "")
    if not uri:
        raise RuntimeError("NEO4J_URI not configured")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = get_secret("NEO4J_PASSWORD", "") or ""
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        yield driver, db
    finally:
        driver.close()


def fetch_kg_stats() -> dict:
    """Get knowledge graph statistics from Neo4j.

    Uses a direct sync Neo4j driver (not Graphiti) to avoid async init errors.
    Returns dict with entity_count, edge_count, episode_count,
    entity_types breakdown, and recent_episodes with their extracted knowledge.
    """
    try:
        with _get_neo4j_sync_driver() as (sync_driver, neo4j_db):
            def _run_cypher(query):
                with sync_driver.session(database=neo4j_db) as s:
                    return [dict(r) for r in s.run(query)]

            entity_counts = _run_cypher(
                "MATCH (n:Entity) "
                "RETURN labels(n) AS labels, count(n) AS cnt"
            )
            edge_count_rows = _run_cypher(
                "MATCH ()-[r:RELATES_TO]->() "
                "RETURN count(r) AS cnt"
            )
            episode_rows = _run_cypher(
                "MATCH (e:Episodic) RETURN count(e) AS cnt"
            )

            # Recent episodes WITH their mentioned entities and linked facts
            # Note: created_at may be STRING (old) or DATE_TIME (new) — use toString() for consistent sorting
            recent_episodes_raw = _run_cypher(
                "MATCH (e:Episodic) "
                "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
                "WITH e, collect(DISTINCT {name: n.name, labels: labels(n)}) AS entities "
                "OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "  WHERE e.uuid IN r.episodes "
                "WITH e, entities, "
                "  collect(DISTINCT {fact: r.fact, from: a.name, to: b.name}) AS facts "
                "RETURN e.name AS name, toString(e.created_at) AS created_at, "
                "  e.group_id AS group_id, e.source AS source, "
                "  entities, facts "
                "ORDER BY toString(e.created_at) DESC LIMIT 10"
            )

        # Format recent episodes with knowledge detail
        recent_episodes = []
        for ep in recent_episodes_raw:
            # Filter out null entries from OPTIONAL MATCH
            entities = [
                {"name": e["name"], "labels": e["labels"]}
                for e in ep.get("entities", [])
                if e.get("name")
            ]
            facts = [
                {"fact": f["fact"], "from": f["from"], "to": f["to"]}
                for f in ep.get("facts", [])
                if f.get("fact")
            ]
            recent_episodes.append({
                "name": str(ep.get("name", ""))[:100],
                "group_id": str(ep.get("group_id", "")),
                "source": str(ep.get("source", "")),
                "created_at": str(ep.get("created_at", "")),
                "entities": entities,
                "facts": facts,
            })

        return {
            "entity_types": {
                str(r.get("labels", [])): r.get("cnt", 0)
                for r in entity_counts
            },
            "edge_count": edge_count_rows[0]["cnt"] if edge_count_rows else 0,
            "episode_count": episode_rows[0]["cnt"] if episode_rows else 0,
            "recent_episodes": recent_episodes,
        }
    except Exception as e:
        logger.error("[shared] fetch_kg_stats error: %s", e)
        return {"error": str(e)}


def search_knowledge_graph(query: str, num_results: int = 10, query_en: str | None = None) -> str | None:
    """Search the knowledge graph and return formatted results.

    Handles connection resets with retry + auto-reset.
    If query_en is provided, searches with both queries and merges results.
    """
    svc = get_kg_service()
    if not svc:
        return None

    _CONN_ERRORS = ("connection reset", "defunct", "connectionreseterror")
    _RESET_KEYWORDS = ("dns", "connection", "timeout", "unavailable", "graphiti")

    def _do_search(q):
        _svc_ref = [svc]
        for attempt in range(2):
            try:
                return run_kg_task(_svc_ref[0].search, query=q, group_ids=None, num_results=num_results)
            except Exception as e:
                err_msg = str(e).lower()
                is_conn_error = any(k in err_msg for k in _CONN_ERRORS)

                if is_conn_error and attempt == 0:
                    logger.info("[KG] connection reset, retrying... query=%s", q[:50])
                    reset_kg_service()
                    _svc_ref[0] = get_kg_service()
                    if not _svc_ref[0]:
                        return None
                    continue

                if is_conn_error:
                    logger.warning("[KG] retry failed. query=%s", q[:50])
                else:
                    logger.warning("[KG] search error (query=%s): %s", q[:50], e)
                if any(k in err_msg for k in _RESET_KEYWORDS):
                    reset_kg_service()
                return None
        return None

    all_nodes, all_edges = [], []
    seen_nodes, seen_edges = set(), set()

    for q in [query, query_en] if query_en and query_en != query else [query]:
        result = _do_search(q)
        if not result:
            continue
        for n in result.get("nodes", []):
            if n.get("uuid") and n["uuid"] not in seen_nodes:
                seen_nodes.add(n["uuid"])
                all_nodes.append(n)
        for e in result.get("edges", []):
            if e.get("uuid") and e["uuid"] not in seen_edges:
                seen_edges.add(e["uuid"])
                all_edges.append(e)

    if not all_nodes and not all_edges:
        return None

    # ── Trust-tier lookup: pull source episode names for all edges in one
    # Cypher pass and extract the [T:tier] prefix encoded by add_kg_episode.
    # Edges without a known tier fall back to "?".
    # Graphiti's search() returns edges without their `episodes` property, so
    # we go to Neo4j directly: (1) fetch episode UUIDs per edge, then (2) fetch
    # episode names and parse the sanitized "T-{tier}-..." prefix that
    # add_kg_episode encoded. Edges without a recognizable tier → "?".
    edge_tier: dict[str, str] = {}
    try:
        edge_uuids = [e.get("uuid") for e in all_edges if e.get("uuid")]
        if edge_uuids:
            with _get_neo4j_sync_driver() as (drv, db):
                with drv.session(database=db) as s:
                    edge_eps_rows = list(s.run(
                        "MATCH ()-[r:RELATES_TO]->() WHERE r.uuid IN $euuids "
                        "RETURN r.uuid AS edge_uuid, r.episodes AS episodes",
                        euuids=edge_uuids,
                    ))
                    edge_to_eps: dict[str, list[str]] = {}
                    all_ep_uuids: set[str] = set()
                    for r in edge_eps_rows:
                        eps = r.get("episodes") or []
                        edge_to_eps[r["edge_uuid"]] = [str(u) for u in eps if u]
                        for u in eps:
                            if u:
                                all_ep_uuids.add(str(u))
                    uuid_to_tier: dict[str, str] = {}
                    if all_ep_uuids:
                        ep_rows = list(s.run(
                            "MATCH (e:Episodic) WHERE e.uuid IN $uuids "
                            "RETURN e.uuid AS uuid, e.name AS name",
                            uuids=list(all_ep_uuids),
                        ))
                        for r in ep_rows:
                            nm = str(r.get("name") or "")
                            if nm.startswith("T-"):
                                rest = nm[2:]
                                for _t in ("corroborated", "unverified", "single", "anchor"):
                                    if rest == _t or rest.startswith(_t + "-"):
                                        uuid_to_tier[r["uuid"]] = _t
                                        break
            _tier_rank = {"anchor": 4, "corroborated": 3, "single": 2, "unverified": 1}
            for edge_uuid, eps in edge_to_eps.items():
                best = None
                for u in eps:
                    t = uuid_to_tier.get(u)
                    if t and (best is None or _tier_rank[t] > _tier_rank[best]):
                        best = t
                if best:
                    edge_tier[edge_uuid] = best
    except Exception as _tier_err:
        logger.debug("[KG] tier lookup skipped: %s", _tier_err)

    lines = []
    if all_nodes:
        lines.append("[Knowledge Graph: Entities]")
        for n in all_nodes:
            summary = (n.get("summary", "") or "")[:300]
            if len(n.get("summary", "") or "") > 300:
                summary += "..."
            lines.append(f"- {n['name']} ({', '.join(n.get('labels', []))}): {summary}")
    if all_edges:
        lines.append("[Knowledge Graph: Facts/Relations]")
        for e in all_edges:
            tier = edge_tier.get(e.get("uuid", ""), "?")
            lines.append(f"- [T:{tier}] {e['fact']}")
    return "\n".join(lines)


