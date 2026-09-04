"""Knowledge graph direct stats and formatted search helpers.

Read path (2026-09-03 redesign):

- ``search_knowledge_graph`` first matches the query against the in-process
  alias index (``kg_runtime.identity.AliasIndex`` — names, name_ko/en and
  curated aliases, no embedding call). One unambiguous entity hit switches to
  entity-centric mode: its 1-hop neighbourhood (active facts first, then
  expired ones) is returned without touching Graphiti. Otherwise Graphiti's
  hybrid search runs as before, and any alias hits get a small neighbourhood
  prepended.
- Every returned edge is hydrated in one Cypher pass: subject/object with
  labels, predicate, valid/invalid/expired dates, trust tier (parsed from the
  episode name, both the ``[T:x]`` and sanitized ``T-x`` forms) and a source
  label (commulingo / research:<slug> / scout / news …).
"""

import logging
import os
import re
from contextlib import contextmanager

from kg_runtime.service_runtime import get_kg_service, reset_kg_service, run_kg_task
from secrets_loader import get_secret
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)
# Neo4j emits a WARNING notification whenever a query touches a property key
# or label that no node has yet (external_ids / alias_keys / :Document before
# the first sync). They are informational; keep them out of service logs.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

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

TIER_RE = re.compile(r"^(?:\[T:|T-)(anchor|corroborated|single|unverified)")
_TIER_RANK = {"anchor": 4, "corroborated": 3, "single": 2, "unverified": 1}

ENTITY_NEIGHBORHOOD_CAP = 25
MINI_NEIGHBORHOOD_CAP = 6
SUMMARY_CHARS = 220


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


# ── Neo4j sync driver ─────────────────────────────────────────────────────────

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


def _run_rows(query: str, **params) -> list[dict]:
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            return [dict(r) for r in s.run(query, **params)]


# ── Hydration ─────────────────────────────────────────────────────────────────

_HYDRATE_EDGES_CYPHER = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.uuid IN $uuids
OPTIONAL MATCH (ep:Episodic) WHERE ep.uuid IN coalesce(r.episodes, [])
WITH a, r, b, collect(DISTINCT ep.name) AS ep_names,
     collect(DISTINCT ep.source_description) AS ep_sources
RETURN r.uuid AS uuid, r.name AS predicate, coalesce(r.fact, '') AS fact,
       a.name AS subject, labels(a) AS subject_labels,
       b.name AS object, labels(b) AS object_labels,
       toString(r.valid_at) AS valid_at, toString(r.invalid_at) AS invalid_at,
       toString(r.expired_at) AS expired_at, toString(r.created_at) AS created_at,
       r.group_id AS group_id, r.sync_key AS sync_key, r.reference_type AS reference_type,
       r.doc_ref AS doc_ref, ep_names, ep_sources
"""

_HYDRATE_NODES_CYPHER = """
MATCH (n:Entity) WHERE n.uuid IN $uuids
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, coalesce(n.summary, '') AS summary,
       coalesce(n.aliases, []) AS aliases, coalesce(n.external_ids, []) AS external_ids
"""

# Active first; substantive predicates (Involvement, Affiliation, Statement…)
# before curated Reference links; then newest. Without the predicate rank a
# CommuLingo person's 60 term references would crowd out the dated facts.
_NEIGHBORHOOD_CYPHER = """
MATCH (n:Entity {uuid: $uuid})
MATCH (n)-[r:RELATES_TO]-(m:Entity)
WITH r, (r.expired_at IS NULL) AS active,
     CASE WHEN r.name = 'Reference' THEN 1 ELSE 0 END AS ref_rank
ORDER BY active DESC, ref_rank ASC, coalesce(r.valid_at, r.created_at) DESC
RETURN r.uuid AS uuid, active
LIMIT $cap
"""


def _tier_from_names(names) -> str | None:
    best = None
    for nm in names or ():
        m = TIER_RE.match(str(nm or ""))
        if m:
            t = m.group(1)
            if best is None or _TIER_RANK[t] > _TIER_RANK[best]:
                best = t
    return best


def _source_label(row: dict) -> str:
    key = row.get("sync_key") or ""
    if key:
        head = key.split(":", 1)[0]
        return head
    doc_ref = row.get("doc_ref")
    if doc_ref:
        return str(doc_ref)
    for src in row.get("ep_sources") or ():
        s = str(src or "")
        m = re.match(r"agent_structured_write \((\w+)\)", s)
        if m:
            return m.group(1)
        if s.startswith("Open source news"):
            return "news"
        if s.startswith("Internal analyst"):
            return "analyst"
        if s:
            return s[:24]
    return "?"


def _label_of(labels) -> str:
    for l in labels or ():
        if l != "Entity" and not str(l).startswith("Entity_"):
            return str(l)
    return "Entity"


def _hydrate_edges(edge_uuids: list[str]) -> dict[str, dict]:
    """One Cypher pass: endpoints, predicate, dates, tier, source per edge."""
    uuids = [u for u in edge_uuids if u]
    if not uuids:
        return {}
    out: dict[str, dict] = {}
    for row in _run_rows(_HYDRATE_EDGES_CYPHER, uuids=uuids):
        row["tier"] = _tier_from_names(row.get("ep_names"))
        row["source"] = _source_label(row)
        row["subject_type"] = _label_of(row.get("subject_labels"))
        row["object_type"] = _label_of(row.get("object_labels"))
        out[row["uuid"]] = row
    return out


def _hydrate_nodes(node_uuids: list[str]) -> dict[str, dict]:
    uuids = [u for u in node_uuids if u]
    if not uuids:
        return {}
    return {row["uuid"]: row for row in _run_rows(_HYDRATE_NODES_CYPHER, uuids=uuids)}


def _entity_neighborhood(uuid: str, cap: int = ENTITY_NEIGHBORHOOD_CAP) -> tuple[dict | None, list[dict]]:
    """Node + its 1-hop facts (active first, newest first). No embedding call."""
    nodes = _hydrate_nodes([uuid])
    node = nodes.get(uuid)
    if not node:
        return None, []
    rows = _run_rows(_NEIGHBORHOOD_CYPHER, uuid=uuid, cap=int(cap))
    hydrated = _hydrate_edges([r["uuid"] for r in rows])
    edges = [hydrated[r["uuid"]] for r in rows if r["uuid"] in hydrated]
    return node, edges


_FACTS_BY_EXTERNAL_ID_CYPHER = """
MATCH (n:Entity) WHERE $eid IN coalesce(n.external_ids, [])
MATCH (n)-[r:RELATES_TO]-(m:Entity)
WHERE r.expired_at IS NULL AND (r.sync_key IS NULL OR NOT r.sync_key STARTS WITH $skip_prefix)
WITH r ORDER BY coalesce(r.valid_at, r.created_at) DESC
RETURN r.uuid AS uuid LIMIT $limit
"""


def kg_facts_for_external_id(external_id: str, *, limit: int = 5, skip_prefix: str = "commulingo:") -> list[str]:
    """Formatted non-mirror facts attached to the node carrying ``external_id``
    (e.g. news/analysis facts about a CommuLingo person). Empty on any failure."""
    try:
        rows = _run_rows(_FACTS_BY_EXTERNAL_ID_CYPHER, eid=external_id, skip_prefix=skip_prefix, limit=int(limit))
        hydrated = _hydrate_edges([r["uuid"] for r in rows])
        return [_format_edge_line(hydrated[r["uuid"]]) for r in rows if r["uuid"] in hydrated]
    except Exception as exc:
        logger.debug("[KG] facts_for_external_id skipped (%s): %s", external_id, exc)
        return []


# ── Formatting ────────────────────────────────────────────────────────────────

def _date_part(value) -> str:
    return str(value or "")[:10]


def _format_node_line(n: dict) -> str:
    label = _label_of(n.get("labels"))
    extras = []
    aliases = [a for a in (n.get("aliases") or []) if a][:2]
    if aliases:
        extras.append("aka: " + ", ".join(aliases))
    ids = [i for i in (n.get("external_ids") or []) if i][:1]
    if ids:
        extras.append("id: " + ids[0])
    summary = (n.get("summary") or "").strip()
    if len(summary) > SUMMARY_CHARS:
        summary = summary[:SUMMARY_CHARS].rstrip() + "…"
    head = f"- {n.get('name')} [{label}]"
    if extras:
        head += " (" + "; ".join(extras) + ")"
    return head + (f": {summary}" if summary else "")


def _format_edge_line(e: dict) -> str:
    flags = [e.get("tier") or "?"]
    if e.get("expired_at"):
        flags.append("expired")
    subj = e.get("subject") or "?"
    obj = e.get("object") or "?"
    pred = e.get("predicate") or "RELATES_TO"
    meta = []
    va, ia = _date_part(e.get("valid_at")), _date_part(e.get("invalid_at"))
    if va or ia:
        meta.append(f"valid {va or '…'} → {ia or '…'}" if ia else f"valid {va}")
    src = e.get("source")
    if src and src != "?":
        meta.append(f"src: {src}")
    line = f"- [{'|'.join(flags)}] {subj} —{pred}→ {obj}: {e.get('fact') or ''}"
    if meta:
        line += " (" + "; ".join(meta) + ")"
    return line


def _format_kg_results(nodes: list[dict], edges: list[dict], edge_tier: dict[str, str] | None = None,
                       *, entity_header: str | None = None) -> str:
    """Render nodes + hydrated edges. ``edge_tier`` is kept for callers that
    pass legacy {uuid: tier} maps; hydrated rows carry ``tier`` themselves."""
    lines = []
    if entity_header:
        lines.append(entity_header)
    if nodes:
        lines.append("[Knowledge Graph: Entities]")
        for n in nodes:
            lines.append(_format_node_line(n))
    if edges:
        lines.append("[Knowledge Graph: Facts/Relations]")
        for e in edges:
            if edge_tier and not e.get("tier"):
                e = dict(e, tier=edge_tier.get(e.get("uuid", ""), "?"))
            if e.get("subject") is None and e.get("fact"):
                lines.append(f"- [{e.get('tier') or '?'}] {e['fact']}")
            else:
                lines.append(_format_edge_line(e))
    return "\n".join(lines)


# ── Direct Cypher fallback ────────────────────────────────────────────────────

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
        node_rows = _run_rows(
            "MATCH (n:Entity) "
            "WHERE any(term IN $terms WHERE "
            "  toLower(coalesce(n.name, '')) CONTAINS term OR "
            "  toLower(coalesce(n.alias_text, '')) CONTAINS term OR "
            "  toLower(coalesce(n.summary, '')) CONTAINS term) "
            "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, "
            "       coalesce(n.summary, '') AS summary, coalesce(n.aliases, []) AS aliases, "
            "       coalesce(n.external_ids, []) AS external_ids "
            "LIMIT $limit",
            terms=terms, limit=num_results,
        )
        edge_rows = _run_rows(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "WHERE any(term IN $terms WHERE "
            "  toLower(coalesce(r.fact, '')) CONTAINS term OR "
            "  toLower(coalesce(a.name, '')) CONTAINS term OR "
            "  toLower(coalesce(b.name, '')) CONTAINS term) "
            "RETURN r.uuid AS uuid "
            "LIMIT $limit",
            terms=terms, limit=num_results,
        )
    except Exception as e:
        logger.warning("[KG] direct Cypher fallback failed (query=%s): %s", query[:50], e)
        return None

    nodes = [r for r in node_rows if r.get("name")]
    hydrated = _hydrate_edges([r["uuid"] for r in edge_rows])
    edges = list(hydrated.values())
    if not nodes and not edges:
        return None
    return "[Knowledge Graph fallback: direct Cypher text match]\n" + _format_kg_results(nodes, edges)


# ── Stats ─────────────────────────────────────────────────────────────────────

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


# ── Entity matching ───────────────────────────────────────────────────────────

def _alias_hits(text: str, limit: int = 5, *, broad: bool = True):
    """Alias-index hits for ``text`` (empty on any failure). ``broad=False``
    skips category words (사회주의, 에너지 …) — used by recall."""
    try:
        from kg_runtime.identity import get_alias_index
        idx = get_alias_index()
        if not idx.ensure_loaded():
            return []
        return idx.match(text, limit=limit, broad=broad)
    except Exception as exc:
        logger.debug("[KG] alias match skipped: %s", exc)
        return []


def _resolve_entity_arg(entity: str):
    """Resolve an explicit ``entity=`` argument: exact/alias-index match, then
    a name/alias_text lookup in Neo4j."""
    hits = _alias_hits(entity, limit=3)
    exact = [h for h in hits if h.name == entity or h.key == entity.lower().strip()]
    if exact:
        return exact[0]
    if len(hits) == 1:
        return hits[0]
    try:
        from kg_runtime.identity import AliasHit, normalize_alias_key
        key = normalize_alias_key(entity)
        rows = _run_rows(
            "MATCH (n:Entity) WHERE n.name = $name OR $key IN coalesce(n.alias_keys, []) "
            "OR toLower(n.name) = $key "
            "OPTIONAL MATCH (n)-[r:RELATES_TO]-() WITH n, count(r) AS d "
            "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels ORDER BY d DESC LIMIT 1",
            name=entity, key=key,
        )
        if rows:
            r = rows[0]
            return AliasHit(r["uuid"], r["name"], [l for l in r["labels"] if l != "Entity"], key)
    except Exception as exc:
        logger.debug("[KG] entity arg lookup failed: %s", exc)
    return hits[0] if hits else None


def _entity_mode_result(hit, cap: int = ENTITY_NEIGHBORHOOD_CAP) -> str | None:
    node, edges = _entity_neighborhood(hit.uuid, cap=cap)
    if not node:
        return None
    active = sum(1 for e in edges if not e.get("expired_at"))
    header = (
        f"[Knowledge Graph: entity view — {node.get('name')} "
        f"({active} active fact(s), {len(edges) - active} expired; matched via '{hit.key}')]"
    )
    return _format_kg_results([node], edges, entity_header=header)


# ── Public search ─────────────────────────────────────────────────────────────

def search_knowledge_graph(query: str, num_results: int = 10, query_en: str | None = None,
                           *, entity: str | None = None, mode: str = "auto") -> str | None:
    """Search the knowledge graph and return formatted results.

    mode:
      - "auto"     alias-index match first; one unambiguous entity → entity view
                   (no embedding call); otherwise semantic search + mini views
      - "entity"   entity view for ``entity`` (or the query's alias hit)
      - "semantic" Graphiti hybrid search only

    Handles connection resets with retry + auto-reset.
    If query_en is provided, searches with both queries and merges results.
    """
    mode = (mode or "auto").lower()
    if mode not in ("auto", "entity", "semantic"):
        mode = "auto"

    # ── Entity-centric path (no embedding) ──
    hits = []
    if mode != "semantic":
        if entity:
            hit = _resolve_entity_arg(entity)
            hits = [hit] if hit else []
        else:
            hits = _alias_hits(query, limit=4)
        if mode == "entity" or (mode == "auto" and len(hits) == 1):
            if hits:
                try:
                    rendered = _entity_mode_result(hits[0])
                    if rendered:
                        return rendered
                except Exception as exc:
                    logger.warning("[KG] entity view failed (%s); falling back to semantic: %s", hits[0].name, exc)
            elif mode == "entity":
                return None

    # ── Semantic path ──
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
        return ToolFailure(
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
        return ToolFailure(
            "Knowledge graph search failed; do not treat this as no KG data. "
            "Direct Cypher fallback found no exact text matches. "
            f"Graphiti error: {search_errors[-1][:500]}"
        )

    # Mini entity views for alias hits (multiple matched entities).
    sections: list[str] = []
    shown_nodes: set[str] = set()
    shown_edges: set[str] = set()
    for hit in hits[:2]:
        try:
            node, edges = _entity_neighborhood(hit.uuid, cap=MINI_NEIGHBORHOOD_CAP)
        except Exception as exc:
            logger.debug("[KG] mini view failed for %s: %s", hit.name, exc)
            continue
        if not node:
            continue
        shown_nodes.add(node["uuid"])
        shown_edges.update(e["uuid"] for e in edges)
        sections.append(_format_kg_results(
            [node], edges, entity_header=f"[Knowledge Graph: entity view — {node.get('name')} (matched via '{hit.key}')]",
        ))
    # Don't repeat what the entity views already showed.
    all_nodes = [n for n in all_nodes if n.get("uuid") not in shown_nodes]
    all_edges = [e for e in all_edges if e.get("uuid") not in shown_edges]

    if not all_nodes and not all_edges and not sections:
        return None

    all_nodes, all_edges = _prioritize_canonical_hits(all_nodes, all_edges, query)

    # Hydrate: endpoints, dates, tier, source in one pass each.
    try:
        node_rows = _hydrate_nodes([n["uuid"] for n in all_nodes])
        all_nodes = [dict(n, **node_rows.get(n["uuid"], {})) for n in all_nodes]
    except Exception as exc:
        logger.debug("[KG] node hydration skipped: %s", exc)
    try:
        edge_rows = _hydrate_edges([e["uuid"] for e in all_edges])
        all_edges = [edge_rows.get(e["uuid"], e) for e in all_edges]
    except Exception as exc:
        logger.debug("[KG] edge hydration skipped: %s", exc)

    body = _format_kg_results(all_nodes, all_edges)
    if body:
        sections.append(body)
    return "\n\n".join(sections)
