"""Knowledge graph direct stats and formatted search helpers."""

import logging
import os
import re
from contextlib import contextmanager

from kg_runtime.service_runtime import get_kg_service, reset_kg_service, run_kg_task
from secrets_loader import get_secret

logger = logging.getLogger(__name__)

KG_QUERY_ALIASES = {
    "diamat": ["디아마트 (DiaMat)", "디아마트 (Diamat)"],
    "dia mat": ["디아마트 (DiaMat)", "디아마트 (Diamat)"],
    "다이아마트": ["디아마트 (DiaMat)", "디아마트 (Diamat)"],
    "디아마트": ["디아마트 (DiaMat)", "디아마트 (Diamat)"],
    "webzine banlan": ["웹진 반란(Uprising)", "웹진 반란 (uprising.kr)"],
    "banlan": ["웹진 반란(Uprising)", "웹진 반란 (uprising.kr)"],
    "uprising.kr": ["웹진 반란(Uprising)", "웹진 반란 (uprising.kr)"],
    "uprising": ["웹진 반란(Uprising)", "웹진 반란 (uprising.kr)"],
    "웹진 반란": ["웹진 반란(Uprising)", "웹진 반란 (uprising.kr)"],
    "shin hyunjoon": ["신현준"],
    "shin hyun-joon": ["신현준"],
    "shin hyeonjun": ["신현준"],
    "shin hyeon-joon": ["신현준"],
    "신현준": ["신현준"],
}


def _expand_query_aliases(query: str) -> str:
    """Append canonical KG names when a query uses common LLM-made aliases."""
    if not query:
        return query
    lowered = query.lower()
    additions = []
    for alias, canonicals in KG_QUERY_ALIASES.items():
        if alias in lowered:
            for canonical in canonicals:
                if canonical not in query and canonical not in additions:
                    additions.append(canonical)
    if not additions:
        return query
    return query + " " + " ".join(additions)


def _canonical_alias_hits(query: str) -> list[str]:
    """Return canonical names implied by alias matches in the query."""
    if not query:
        return []
    lowered = query.lower()
    hits = []
    for alias, canonicals in KG_QUERY_ALIASES.items():
        if alias in lowered:
            for canonical in canonicals:
                if canonical not in hits:
                    hits.append(canonical)
    return hits


def _prioritize_canonical_hits(nodes: list[dict], edges: list[dict], query: str) -> tuple[list[dict], list[dict]]:
    canonicals = _canonical_alias_hits(query)
    if not canonicals:
        return nodes, edges

    def node_score(n: dict) -> int:
        name = str(n.get("name") or "")
        summary = str(n.get("summary") or "")
        return max(
            (
                3 if name == canonical
                else 2 if canonical in name
                else 1 if canonical in summary
                else 0
            )
            for canonical in canonicals
        )

    def edge_score(e: dict) -> int:
        fact = str(e.get("fact") or "")
        return max((1 if canonical in fact else 0) for canonical in canonicals)

    return (
        sorted(nodes, key=node_score, reverse=True),
        sorted(edges, key=edge_score, reverse=True),
    )


def _format_kg_results(nodes: list[dict], edges: list[dict], edge_tier: dict[str, str] | None = None) -> str:
    lines = []
    if nodes:
        lines.append("[Knowledge Graph: Entities]")
        for n in nodes:
            summary = (n.get("summary", "") or "")[:300]
            if len(n.get("summary", "") or "") > 300:
                summary += "..."
            lines.append(f"- {n['name']} ({', '.join(n.get('labels', []))}): {summary}")
    if edges:
        lines.append("[Knowledge Graph: Facts/Relations]")
        edge_tier = edge_tier or {}
        for e in edges:
            tier = edge_tier.get(e.get("uuid", ""), "?")
            lines.append(f"- [T:{tier}] {e['fact']}")
    return "\n".join(lines)


def _direct_cypher_search(query: str, num_results: int = 10) -> str | None:
    """Exact text fallback for when Graphiti semantic search fails.

    This is intentionally simpler than Graphiti search. It exists so parser,
    embedder, or LLM failures do not masquerade as "no KG data".
    """
    expanded_query = _expand_query_aliases(query)
    raw_terms = [query.strip(), expanded_query.strip()]
    raw_terms.extend(re.findall(r"[0-9A-Za-z가-힣_.-]{2,}", expanded_query))
    for alias, canonicals in KG_QUERY_ALIASES.items():
        if alias in expanded_query.lower():
            raw_terms.append(alias)
            raw_terms.extend(canonicals)
    terms = []
    seen = set()
    for term in raw_terms:
        clean = term.strip().lower()
        if len(clean) < 2 or clean in seen:
            continue
        seen.add(clean)
        terms.append(clean)
        if len(terms) >= 12:
            break
    if not terms:
        return None

    try:
        with _get_neo4j_sync_driver() as (drv, db):
            with drv.session(database=db) as s:
                node_rows = list(s.run(
                    "MATCH (n:Entity) "
                    "WHERE any(term IN $terms WHERE "
                    "  toLower(coalesce(n.name, '')) CONTAINS term OR "
                    "  toLower(coalesce(n.summary, '')) CONTAINS term) "
                    "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, "
                    "       coalesce(n.summary, '') AS summary "
                    "LIMIT $limit",
                    terms=terms,
                    limit=num_results,
                ))
                edge_rows = list(s.run(
                    "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                    "WHERE any(term IN $terms WHERE "
                    "  toLower(coalesce(r.fact, '')) CONTAINS term OR "
                    "  toLower(coalesce(a.name, '')) CONTAINS term OR "
                    "  toLower(coalesce(b.name, '')) CONTAINS term) "
                    "RETURN r.uuid AS uuid, coalesce(r.fact, '') AS fact, "
                    "       a.name AS source, b.name AS target "
                    "LIMIT $limit",
                    terms=terms,
                    limit=num_results,
                ))
    except Exception as e:
        logger.warning("[KG] direct Cypher fallback failed (query=%s): %s", query[:50], e)
        return None

    nodes = [dict(r) for r in node_rows if r.get("name")]
    edges = []
    for r in edge_rows:
        row = dict(r)
        if not row.get("fact"):
            row["fact"] = f"{row.get('source', '?')} RELATES_TO {row.get('target', '?')}"
        edges.append(row)
    if not nodes and not edges:
        return None
    return "[Knowledge Graph fallback: direct Cypher text match]\n" + _format_kg_results(nodes, edges)

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
            # LIMIT 먼저: 전체 에피소드 × 전체 엣지를 스캔한 뒤 자르면 그래프가 클수록 느려진다.
            recent_episodes_raw = _run_cypher(
                "MATCH (e:Episodic) "
                "WITH e ORDER BY toString(e.created_at) DESC LIMIT 10 "
                "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
                "WITH e, collect(DISTINCT {name: n.name, labels: labels(n)}) AS entities "
                "OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "  WHERE e.uuid IN r.episodes "
                "WITH e, entities, "
                "  collect(DISTINCT {fact: r.fact, from: a.name, to: b.name}) AS facts "
                "RETURN e.name AS name, toString(e.created_at) AS created_at, "
                "  e.group_id AS group_id, e.source AS source, "
                "  entities, facts "
                "ORDER BY toString(e.created_at) DESC"
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
    _CONN_ERRORS = ("connection reset", "defunct", "connectionreseterror")
    _RESET_KEYWORDS = ("dns", "connection", "timeout", "unavailable")
    search_errors: list[str] = []

    query = _expand_query_aliases(query)
    if query_en:
        query_en = _expand_query_aliases(query_en)

    svc = get_kg_service()
    if not svc:
        fallback = _direct_cypher_search(query, num_results)
        if fallback:
            return (
                "Knowledge graph semantic search failed because the Graphiti service "
                "is unavailable; using direct Cypher fallback.\n"
                + fallback
            )
        return (
            "Knowledge graph search failed; do not treat this as no KG data. "
            "Graphiti service unavailable and direct Cypher fallback found no exact text matches."
        )

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
                search_errors.append(str(e))
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

    if not all_nodes and not all_edges and search_errors:
        fallback = _direct_cypher_search(query, num_results)
        if fallback:
            return (
                "Knowledge graph semantic search failed; using direct Cypher fallback. "
                f"Graphiti error: {search_errors[-1][:500]}\n"
                + fallback
            )
        return (
            "Knowledge graph search failed; do not treat this as no KG data. "
            "Direct Cypher fallback found no exact text matches. "
            f"Graphiti error: {search_errors[-1][:500]}"
        )

    if not all_nodes and not all_edges:
        return None

    all_nodes, all_edges = _prioritize_canonical_hits(all_nodes, all_edges, query)

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

    return _format_kg_results(all_nodes, all_edges, edge_tier)
