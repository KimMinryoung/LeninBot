"""Check Knowledge Graph invariants that Graphiti search requires.

Use after any manual KG correction, merge, delete, or restore:
  venv/bin/python scripts/check_kg_integrity.py
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from _notify import notify_telegram as _notify_telegram
from kg_runtime.integrity import check_kg_integrity, format_integrity_status


def _run_smoke_search(query: str, *, mode: str = "auto", expected_entity: str | None = None) -> dict:
    from kg_runtime.search import search_knowledge_graph

    try:
        result = search_knowledge_graph(query, 3, mode=mode)
    except Exception as exc:
        return {"ok": False, "query": query, "error": str(exc)}
    if not result:
        return {"ok": False, "query": query, "error": "no result returned"}

    degraded_prefixes = (
        "Knowledge graph semantic search failed",
        "Knowledge graph search failed",
    )
    metadata = getattr(result, "result_metadata", {}) or {}
    degraded = result.startswith(degraded_prefixes) or bool(metadata.get("fallback"))
    expected_match = not expected_entity or f"- {expected_entity} [" in result
    if mode == "semantic" and metadata.get("path") != "semantic":
        degraded = True
    return {
        "ok": not degraded and expected_match,
        "expected_entity": expected_entity, "expected_match": expected_match, "metadata": metadata,
        "query": query,
        "degraded": degraded,
        "preview": result[:1000],
    }


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--smoke-query", help="Run an end-to-end KG search smoke test.")
    parser.add_argument("--expected-entity", help="Exact entity name required in the smoke result")
    parser.add_argument("--semantic-query", help="Additionally exercise embedding/hybrid search")
    parser.add_argument("--metrics", action="store_true", help="Include graph, sync, usage and source coverage")
    parser.add_argument("--notify", action="store_true", help="Notify Telegram on failure/degradation.")
    args = parser.parse_args()

    status = check_kg_integrity()
    smoke = None
    if args.smoke_query:
        smoke = _run_smoke_search(args.smoke_query, expected_entity=args.expected_entity)
        status["smoke_search"] = smoke

    if args.semantic_query:
        status["semantic_search"] = _run_smoke_search(args.semantic_query, mode="semantic")
    if args.metrics:
        from kg_runtime.metrics import collect_kg_metrics
        status["metrics"] = collect_kg_metrics()
    print(format_integrity_status(status))
    print(json.dumps(status, ensure_ascii=False, indent=2))

    ok = bool(status.get("ok")) and (smoke is None or bool(smoke.get("ok")))
    ok = ok and status.get("semantic_search", {}).get("ok", True)
    if args.metrics:
        metrics = status["metrics"]
        ok = ok and not any(isinstance(v, dict) and v.get("error") for v in metrics.values())
        from kg_runtime.metrics import sync_unhealthy
        sync = metrics.get("sync", {})
        ok = ok and all(name in sync and not sync_unhealthy(sync[name]) for name in ("commulingo", "documents"))
        # Fresh source edits may legitimately wait for tonight's sync. Coverage
        # is reported, while failed/incomplete/stale runs trigger alerts.
    if not ok and args.notify:
        _notify_telegram(
            "KG integrity/search healthcheck failed\n"
            + json.dumps(status, ensure_ascii=False, indent=2)[:3500]
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
