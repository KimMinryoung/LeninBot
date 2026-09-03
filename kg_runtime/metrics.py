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
    out["empty_summary"] = one("MATCH (n:Entity) WHERE coalesce(n.summary, '') = '' RETURN count(n) AS c").get("c", 0)
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
        rows = db_query("SELECT source, watermark, last_run_at, last_full_at, stats FROM kg_sync_state")
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
                   sum(CASE WHEN error_excerpt ILIKE '%%No knowledge graph results%%'
                             OR args_summary ILIKE '%%No knowledge graph results%%' THEN 1 ELSE 0 END) AS empty
            FROM tool_audit_log
            WHERE ts > now() - (%s || ' days')::interval
              AND tool_name IN ('knowledge_graph_search', 'write_kg_structured')
            GROUP BY 1, 2, 3, 4 ORDER BY 5 DESC
            """,
            (str(days),),
        )
    except Exception as exc:
        return {"error": str(exc)}
    searches = sum(r["n"] for r in rows if r["tool_name"] == "knowledge_graph_search")
    writes = sum(r["n"] for r in rows if r["tool_name"] == "write_kg_structured")
    return {
        "days": days, "searches": searches, "writes": writes,
        "by_caller": [
            {"tool": r["tool_name"], "interface": r["interface"], "agent": r["agent"], "status": r["result_status"],
             "n": r["n"], "avg_ms": r["avg_ms"]}
            for r in rows[:20]
        ],
    }


def collect_kg_metrics(*, usage_days: int = 14) -> dict:
    out: dict = {"collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key, fn in (("graph", graph_metrics), ("sync", sync_metrics), ("usage", lambda: usage_metrics(usage_days))):
        try:
            out[key] = fn()
        except Exception as exc:
            logger.warning("[KG metrics] %s failed: %s", key, exc)
            out[key] = {"error": str(exc)}
    return out


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
            flag = " ⚠️" if (st.get("error") or (st.get("lag_hours") or 0) > 48) else ""
            lines.append(f"sync {name}: {st.get('mode')} {st.get('last_run_at')} (lag {st.get('lag_hours')}h, wrote {st.get('written')}){flag}")
    else:
        lines.append("sync: 아직 실행 기록 없음")
    if "error" in u:
        lines.append(f"usage: ERROR {u['error'][:120]}")
    else:
        lines.append(f"최근 {u.get('days')}일 검색 {u.get('searches', 0)} · 쓰기 {u.get('writes', 0)}")
        top = [c for c in (u.get("by_caller") or []) if c["tool"] == "knowledge_graph_search"][:4]
        if top:
            lines.append("검색 호출자: " + ", ".join(f"{c['interface']}/{c['agent'] or '-'} {c['n']}" for c in top))
    return "\n".join(lines)
