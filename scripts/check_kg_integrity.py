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

from scripts._notify import notify_telegram as _notify_telegram
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
    reason = None
    if degraded:
        reason = result.splitlines()[0][:450]
        if not reason:
            reason = f"search path={metadata.get('path') or 'unknown'}, fallback={bool(metadata.get('fallback'))}"
    elif not expected_match:
        reason = f"expected entity missing: {expected_entity}"
    return {
        "ok": not degraded and expected_match,
        "error": reason,
        "expected_entity": expected_entity, "expected_match": expected_match, "metadata": metadata,
        "query": query,
        "degraded": degraded,
        "preview": result[:1000],
    }


def healthcheck_status(status: dict, *, metrics_enabled: bool = False) -> dict:
    """Aggregate failures separately from the graph-invariant `ok` field."""
    failures = []
    if not status.get("ok"):
        failures.append(format_integrity_status(status))
    for name in ("smoke_search", "semantic_search"):
        result = status.get(name)
        if result is not None and not result.get("ok"):
            failures.append(f"{name}: {result.get('error') or 'degraded search or expected entity missing'}")
    if metrics_enabled:
        from kg_runtime.metrics import sync_unhealthy

        metrics = status.get("metrics", {})
        for name, value in metrics.items():
            if isinstance(value, dict) and value.get("error"):
                failures.append(f"metrics.{name}: {value['error']}")
        sync = metrics.get("sync", {})
        for name in ("commulingo", "documents"):
            state = sync.get(name)
            if state is None:
                failures.append(f"sync.{name}: missing sync state")
            elif sync_unhealthy(state):
                failures.append(
                    f"sync.{name}: {state.get('error') or 'incomplete or stale sync'} "
                    f"(phase={state.get('phase')}, complete={state.get('complete')}, "
                    f"remaining={state.get('remaining')}, failed={state.get('failed')}, "
                    f"lag_hours={state.get('lag_hours')})"
                )
    return {"ok": not failures, "failures": failures}


def format_healthcheck_alert(health: dict) -> str:
    # Keep failure causes ahead of bulky graph metrics and search previews.
    lines = ["KG healthcheck failed"]
    lines.extend(f"- {reason[:450]}" for reason in health["failures"])
    return "\n".join(lines)[:3500]


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
    health = healthcheck_status(status, metrics_enabled=args.metrics)
    status["healthcheck"] = health
    print("KG healthcheck: " + ("ok" if health["ok"] else "failed"))
    print(format_integrity_status(status))
    print(json.dumps(status, ensure_ascii=False, indent=2))

    ok = health["ok"]
    if not ok and args.notify:
        _notify_telegram(format_healthcheck_alert(health))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
