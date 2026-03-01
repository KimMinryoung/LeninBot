"""
KG 질의 CLI — 지식그래프에 자연어로 질의
==========================================

Usage:
  python -m graph_memory.cli "이재명에 대해 설명하라"
  python -m graph_memory.cli "현재 한국 대통령은 누구인가?"
  python -m graph_memory.cli "South Korea president" --group korea_domestic
"""

import argparse
import asyncio

from dotenv import load_dotenv
load_dotenv()

from .service import GraphMemoryService


async def run_query(query: str, group_ids: list[str] | None, num_results: int):
    service = GraphMemoryService()
    await service.initialize()

    try:
        result = await service.query_chatbot(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
        )

        print("\n" + "=" * 60)
        print(f"Query: {result['query']}")
        print("=" * 60)
        print(f"\n{result['answer']}\n")

        ctx = result.get("context", {})
        nodes = ctx.get("nodes", [])
        edges = ctx.get("edges", [])

        if nodes or edges:
            print("-" * 40)
            print(f"Context: {len(nodes)} nodes, {len(edges)} edges")
            for n in nodes[:10]:
                labels = ", ".join(n.get("labels", []))
                summary = n.get("summary") or ""
                print(f"  [node] {n['name']} ({labels})")
                if summary:
                    print(f"         {summary[:120]}")
            for e in edges[:10]:
                print(f"  [edge] {e['fact'][:100]}")
    finally:
        await service.close()


def main():
    parser = argparse.ArgumentParser(description="KG query CLI")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--group", dest="groups", action="append",
                        help="group_id filter (repeatable)")
    parser.add_argument("-n", "--num-results", type=int, default=12,
                        help="Number of search results (default: 12)")
    args = parser.parse_args()

    asyncio.run(run_query(args.query, args.groups, args.num_results))


if __name__ == "__main__":
    main()
