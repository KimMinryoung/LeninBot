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


def _run_smoke_search(query: str) -> dict:
    from kg_runtime.search import search_knowledge_graph

    result = search_knowledge_graph(query, 3)
    if not result:
        return {"ok": False, "query": query, "error": "no result returned"}

    degraded_prefixes = (
        "Knowledge graph semantic search failed",
        "Knowledge graph search failed",
    )
    degraded = result.startswith(degraded_prefixes)
    return {
        "ok": not degraded,
        "query": query,
        "degraded": degraded,
        "preview": result[:1000],
    }


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--smoke-query", help="Run an end-to-end KG search smoke test.")
    parser.add_argument("--notify", action="store_true", help="Notify Telegram on failure/degradation.")
    args = parser.parse_args()

    status = check_kg_integrity()
    smoke = None
    if args.smoke_query:
        smoke = _run_smoke_search(args.smoke_query)
        status["smoke_search"] = smoke

    print(format_integrity_status(status))
    print(json.dumps(status, ensure_ascii=False, indent=2))

    ok = bool(status.get("ok")) and (smoke is None or bool(smoke.get("ok")))
    if not ok and args.notify:
        _notify_telegram(
            "KG integrity/search healthcheck failed\n"
            + json.dumps(status, ensure_ascii=False, indent=2)[:3500]
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
