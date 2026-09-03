#!/usr/bin/env python3
"""Smoke: run representative webchat/analyst queries through the KG read path
and print the rendered results, so before/after comparisons (CommuLingo
mirror, document extraction) are visible in full.

    NEO4J_PASSWORD=... venv/bin/python scripts/smoke_kg_search_modes.py [--out FILE] [QUERY ...]

Read-only. The semantic path costs one Gemini embedding per query (existing
kg_embedding call site); the entity path costs nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

DEFAULT_QUERIES = [
    "자민통 자주적 민주주의 노선 NL 민주노총 통일운동",
    "민주노총 위원장 계열 자민통",
    "김문수 차명진 이재오 박형준 전향 1990년대 민주화운동",
    "스탈린",
    "예조프",
    "니키타 흐루쇼프 비밀연설",
    "대숙청",
    "DiaMat",
    "Anthropic Dario Amodei governance",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "entity", "semantic"])
    ap.add_argument("queries", nargs="*")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    from kg_runtime.search import search_knowledge_graph

    queries = args.queries or DEFAULT_QUERIES
    out_lines = [f"# KG search smoke — {datetime.now().isoformat(timespec='seconds')} mode={args.mode}", ""]
    for q in queries:
        t0 = time.monotonic()
        try:
            res = search_knowledge_graph(q, 10, mode=args.mode)
        except Exception as exc:
            res = f"<error: {exc}>"
        dt = time.monotonic() - t0
        text = str(res) if res else "<no results>"
        n_facts = sum(1 for l in text.splitlines() if l.startswith("- [") )
        n_entities = sum(1 for l in text.splitlines() if l.startswith("- ") and not l.startswith("- ["))
        out_lines.append(f"## {q}   ({dt:.1f}s; entities={n_entities}, facts={n_facts})")
        out_lines.append(text)
        out_lines.append("")
    body = "\n".join(out_lines)
    print(body)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\n[saved {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
