"""KG health metrics: graph shape, growth, sync lag, search usage.

``collect_kg_metrics()`` returns one JSON-serialisable dict used by
``scripts/check_kg_integrity.py --metrics`` and the weekly report
(``scripts/kg_weekly_report.py``). Read-only; each block is best-effort so
one failing source does not hide the others.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _cypher_rows(query: str, **params) -> list[dict]:
    from kg_runtime.search import _get_neo4j_sync_driver
    with _get_neo4j_sync_driver() as (drv, db):
        with drv.session(database=db) as s:
            return [dict(r) for r in s.run(query, **params)]


def graph_metrics() -> dict:
    out: dict = {}
    one = lambda q, **p: (_cypher_rows(q, **p) or [{}])[0]  # noqa: E731
    out["entities"] = one("MATCH (n:Entity) RETURN count(n) AS c").get("c", 0)
    out["edges"] = one("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").get("c", 0)
    out["episodes"] = one("MATCH (n:Episodic) RETURN count(n) AS c").get("c", 0)
    out["by_label"] = {
        (next((l for l in r["labels"] if l != "Entity" and not l.startswith("Entity_")), "Entity")): r["c"]
        for r in _cypher_rows("MATCH (n:Entity) RETURN labels(n) AS labels, count(*) AS c")
    }
    out["orphans"] = one("MATCH (n:Entity) WHERE NOT (n)-[:RELATES_TO]-() RETURN count(n) AS c").get("c", 0)
    out["raw_empty_summary"] = one("MATCH (n:Entity) WHERE coalesce(n.summary, '') = '' RETURN count(n) AS c").get("c", 0)
    out["empty_summary"] = one("MATCH (n:Entity) WHERE coalesce(n.curated_summary, '') = '' AND coalesce(n.summary, '') = '' RETURN count(n) AS c").get("c", 0)
    out["curated_profiles"] = one("MATCH (n:Entity) WHERE n.curated_source IS NOT NULL RETURN count(n) AS c").get("c", 0)
    out["with_external_ids"] = one("MATCH (n:Entity) WHERE size(coalesce(n.external_ids, [])) > 0 RETURN count(n) AS c").get("c", 0)
    dup = one("MATCH (n:Entity) WITH n.name AS nm, count(*) AS c WHERE c > 1 RETURN count(*) AS groups, sum(c) AS nodes")
    out["duplicate_name_groups"] = dup.get("groups", 0)
    out["duplicate_name_nodes"] = dup.get("nodes", 0)
    out["expired_edges"] = one("MATCH ()-[r:RELATES_TO]->() WHERE r.expired_at IS NOT NULL RETURN count(r) AS c").get("c", 0)
    out["edges_by_source"] = {
        (r["src"] or "agent/news"): r["c"]
        for r in _cypher_rows(
            "MATCH ()-[r:RELATES_TO]->() WITH CASE WHEN r.sync_key IS NULL THEN NULL "
            "ELSE split(r.sync_key, ':')[0] END AS src RETURN src, count(*) AS c ORDER BY c DESC"
        )
    }
    out["documents_by_kind"] = {
        (r["k"] or "?"): r["c"] for r in _cypher_rows("MATCH (n:Entity:Document) RETURN n.doc_kind AS k, count(*) AS c")
    }
    out["edges_per_week"] = [
        {"week": r["w"], "edges": r["c"]}
        for r in _cypher_rows(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.created_at >= datetime() - duration('P56D') "
            "WITH date(datetime(r.created_at)) AS d, count(*) AS c "
            "WITH toString(d.year) + '-W' + toString(d.week) AS w, sum(c) AS c "
            "RETURN w, c ORDER BY w"
        )
    ]
    degree = one(
        "MATCH (n:Entity) OPTIONAL MATCH (n)-[r:RELATES_TO]-() WITH n, count(r) AS d "
        "RETURN sum(CASE WHEN d <= 2 THEN 1 ELSE 0 END) AS low, sum(CASE WHEN d > 5 THEN 1 ELSE 0 END) AS high, count(*) AS total"
    )
    out["degree_le2_share"] = round(degree.get("low", 0) / max(degree.get("total", 1), 1), 3)
    out["degree_gt5"] = degree.get("high", 0)
    return out


def sync_metrics() -> dict:
    try:
        from db import query as db_query
        rows = db_query("SELECT source, watermark, last_run_at, last_full_at, last_attempt_at, stats FROM kg_sync_state")
    except Exception as exc:
        if "kg_sync_state" in str(exc) and "does not exist" in str(exc):
            return {}
        return {"error": str(exc)}
    now = datetime.now(timezone.utc)
    out = {}
    for r in rows:
        last = r.get("last_run_at")
        stats = r.get("stats") or {}
        out[r["source"]] = {
            "last_run_at": str(last)[:19] if last else None,
            "lag_hours": round((now - last).total_seconds() / 3600, 1) if last else None,
            "last_full_at": str(r.get("last_full_at"))[:19] if r.get("last_full_at") else None,
            "mode": stats.get("mode"), "error": stats.get("error"),
            "last_attempt_at": str(r.get("last_attempt_at"))[:19] if r.get("last_attempt_at") else None,
            "complete": stats.get("complete"), "remaining": stats.get("remaining"), "failed": stats.get("failed"),
            "phase": stats.get("phase", "finished"),
            "attempt_lag_hours": round((now - r["last_attempt_at"]).total_seconds() / 3600, 2) if r.get("last_attempt_at") else None,
            "written": (stats.get("write") or {}).get("written", stats.get("written")),
        }
    return out


def usage_metrics(days: int = 14) -> dict:
    try:
        from db import query as db_query
        rows = db_query(
            """
            SELECT tool_name, interface, coalesce(agent_name, '') AS agent, result_status,
                   count(*) AS n, round(avg(latency_ms)) AS avg_ms,
                   round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric) AS p95_ms,
                   count(*) FILTER (WHERE result_metadata->>'empty' IS NOT NULL) AS measured,
                   count(*) FILTER (WHERE result_metadata->>'empty' = 'true') AS empty,
                   count(*) FILTER (WHERE result_metadata->>'fallback' = 'true') AS fallback,
                   count(*) FILTER (WHERE result_metadata IS NOT NULL) AS diagnosed
            FROM tool_audit_log
            WHERE ts > now() - (%s || ' days')::interval
              AND tool_name IN ('knowledge_graph_search', 'write_kg_structured')
              AND agent_name IS DISTINCT FROM 'kg_verification'
            GROUP BY 1, 2, 3, 4 ORDER BY 5 DESC
            """,
            (str(days),),
        )
    except Exception as exc:
        return {"error": str(exc)}
    searches = sum(r["n"] for r in rows if r["tool_name"] == "knowledge_graph_search")
    writes = sum(r["n"] for r in rows if r["tool_name"] == "write_kg_structured")
    callers = {}
    for row in rows:
        if row["tool_name"] != "knowledge_graph_search":
            continue
        key = (row["interface"], row["agent"])
        c = callers.setdefault(key, {"interface": key[0], "agent": key[1], "n": 0, "failed": 0,
                                     "empty": 0, "measured": 0, "fallback": 0, "diagnosed": 0})
        c["n"] += row["n"]
        c["failed"] += row["n"] if row["result_status"] != "ok" else 0
        for field in ("empty", "measured", "fallback", "diagnosed"):
            c[field] += row[field]
    for c in callers.values():
        c["failure_rate"] = round(c["failed"] / c["n"], 3)
        c["empty_rate"] = round(c["empty"] / c["measured"], 3) if c["measured"] else None
        c["fallback_rate"] = round(c["fallback"] / c["diagnosed"], 3) if c["diagnosed"] else None
        c["unknown"] = c["n"] - c["measured"]
    return {
        "days": days, "searches": searches, "writes": writes,
        "callers": list(callers.values()),
        "search_failure_rate": round(sum(r["n"] for r in rows if r["tool_name"] == "knowledge_graph_search" and r["result_status"] != "ok") / searches, 3) if searches else None,
        "by_caller": [
            {"tool": r["tool_name"], "interface": r["interface"], "agent": r["agent"], "status": r["result_status"],
             "n": r["n"], "avg_ms": int(r["avg_ms"]) if r["avg_ms"] is not None else None,
             "p95_ms": int(r["p95_ms"]) if r["p95_ms"] is not None else None,
             "empty": r["empty"], "measured": r["measured"], "unknown": r["n"] - r["measured"],
             "empty_rate": round(r["empty"] / r["measured"], 3) if r["measured"] else None,
             "fallback_rate": round(r["fallback"] / r["diagnosed"], 3) if r["diagnosed"] else None}
            for r in rows[:20]
        ],
    }


def source_coverage_metrics() -> dict:
    """Read-only reconciliation against current source rows, not run timestamps."""
    from jobs.kg_sync_commulingo import load_source, build_facts, existing_sync_edges, fact_changed
    from jobs.kg_sync_documents import load_records, _commulingo_names
    from kg_runtime import doc_extract as dx
    from graph_memory.graphiti_patches import normalize_entity_names_in_text
    expected = build_facts(load_source())
    existing = existing_sync_edges()
    missing = changed = 0
    for fact in expected:
        edge = existing.get(fact["attributes"]["sync_key"])
        if edge is None:
            missing += 1
        elif fact_changed(fact, edge):
            changed += 1
    records = load_records()
    docs = {r.ref: r["sha"] for r in records}
    names = _commulingo_names()
    document_edges = existing_sync_edges(prefix="doc:")
    links = [f for rec in records for f in [dx.collection_fact(rec), *dx.curated_link_facts(rec, names)]]
    link_missing = sum(f["attributes"]["sync_key"] not in document_edges for f in links)
    link_changed = sum(fact_changed(f, document_edges[f["attributes"]["sync_key"]]) for f in links
                       if f["attributes"]["sync_key"] in document_edges)
    graph_docs = {r["ref"]: r["sha"] for r in _cypher_rows(
        "MATCH (n:Entity:Document) UNWIND coalesce(n.external_ids, []) AS ref "
        "RETURN ref, n.content_sha256 AS sha")}
    return {"commulingo_expected": len(expected), "commulingo_missing": missing,
            "commulingo_changed": changed, "documents_expected": len(docs),
            "document_links_missing": link_missing, "document_links_changed": link_changed,
            "documents_missing": sorted(docs.keys() - graph_docs.keys()),
            "documents_changed": sum(graph_docs.get(ref) != sha for ref, sha in docs.items() if ref in graph_docs)}


def collect_kg_metrics(*, usage_days: int = 14) -> dict:
    out: dict = {"collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key, fn in (("graph", graph_metrics), ("sync", sync_metrics), ("usage", lambda: usage_metrics(usage_days)), ("coverage", source_coverage_metrics)):
        try:
            out[key] = fn()
        except Exception as exc:
            logger.warning("[KG metrics] %s failed: %s", key, exc)
            out[key] = {"error": str(exc)}
    return out


def sync_unhealthy(state: dict) -> bool:
    if state.get("phase") == "running" and state.get("attempt_lag_hours") is not None and state["attempt_lag_hours"] < 2:
        return False
    return bool(state.get("error") or state.get("complete") is False
                or state.get("lag_hours") is None or state["lag_hours"] > 48)


def format_report(m: dict) -> str:
    g, s, u = m.get("graph", {}), m.get("sync", {}), m.get("usage", {})
    lines = [f"📊 KG 주간 리포트 ({m.get('collected_at', '')[:10]})"]
    if "error" in g:
        lines.append(f"graph: ERROR {g['error'][:120]}")
    else:
        lines.append(
            f"노드 {g.get('entities', 0):,} · 엣지 {g.get('edges', 0):,} (만료 {g.get('expired_edges', 0):,}) · 에피소드 {g.get('episodes', 0):,}"
        )
        lines.append(
            f"고립 {g.get('orphans', 0):,} · 차수≤2 {g.get('degree_le2_share', 0):.0%} · 빈 summary {g.get('empty_summary', 0):,} · "
            f"동명 중복 {g.get('duplicate_name_groups', 0)}그룹/{g.get('duplicate_name_nodes', 0)}노드 · 외부id {g.get('with_external_ids', 0):,}"
        )
        docs = g.get("documents_by_kind") or {}
        if docs:
            lines.append("문서 노드: " + ", ".join(f"{k} {v}" for k, v in docs.items()))
        weeks = g.get("edges_per_week") or []
        if weeks:
            lines.append("주간 신규 엣지: " + " · ".join(f"{w['week'][-3:]}={w['edges']}" for w in weeks[-8:]))
        src = g.get("edges_by_source") or {}
        if src:
            lines.append("출처별 엣지: " + ", ".join(f"{k} {v:,}" for k, v in list(src.items())[:6]))
    if "error" in s:
        lines.append(f"sync: ERROR {s['error'][:120]}")
    elif s:
        for name, st in s.items():
            flag = " ⚠️" if sync_unhealthy(st) else ""
            lines.append(f"sync {name}: {st.get('mode')} {st.get('last_run_at')} (lag {st.get('lag_hours')}h, wrote {st.get('written')}, remaining {st.get('remaining')}, failed {st.get('failed')}){flag}")
    else:
        lines.append("sync: 아직 실행 기록 없음")
    if "error" in u:
        lines.append(f"usage: ERROR {u['error'][:120]}")
    else:
        lines.append(f"최근 {u.get('days')}일 검색 {u.get('searches', 0)} · 쓰기 {u.get('writes', 0)}")
        for c in u.get("callers") or []:
            lines.append(f"{c['interface']}/{c['agent'] or '-'}: {c['n']}건, 실패 {c['failed']} ({c['failure_rate']:.0%}), "
                         f"empty={c['empty_rate']}, fallback={c['fallback_rate']}, 미측정 {c['unknown']}")
        top = [c for c in (u.get("by_caller") or []) if c["tool"] == "knowledge_graph_search"][:4]
        if top:
            lines.append("검색 호출자: " + ", ".join(f"{c['interface']}/{c['agent'] or '-'} {c['status']} {c['n']} (empty={c['empty_rate']}, fallback={c['fallback_rate']}, p95={c['p95_ms']}ms)" for c in top))
    coverage = m.get("coverage") or {}
    if coverage:
        lines.append("원천 대조: " + str(coverage))
    return "\n".join(lines)
