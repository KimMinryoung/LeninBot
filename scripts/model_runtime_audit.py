#!/usr/bin/env python3
"""Print effective provider/model/budget policy for runtime surfaces and agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from ops.model_runtime import build_snapshot


def _print_table(snapshot: dict[str, Any]) -> None:
    print("Runtime surfaces")
    print("surface           provider   tier      model_id")
    for name, row in snapshot["surfaces"].items():
        print(f"{name:<17} {row.get('provider', ''):<10} {str(row.get('tier', '')):<9} {row.get('model_id', '')}")

    print("\nAgents")
    print("agent              provider      model                    budget  rounds  input   output  cont  thinking    think_budget")
    for row in snapshot["agents"]:
        provider = row["provider_effective"]
        model = row["model_id"] or ""
        print(
            f"{row['agent']:<18} {provider:<13} {model:<24} "
            f"{row['budget_usd']:<7.2f} {row['max_rounds']:<7} "
            f"{row['max_input_tokens']:<7} {row['max_output_tokens']:<7} "
            f"{row['max_output_continuations']:<5} {row['thinking_policy']:<11} "
            f"{row['thinking_budget_tokens']}"
        )

    print("\nWriter call policies")
    print("role          input    output   rounds  continuations  thinking")
    for row in snapshot["writer_call_policies"]:
        print(
            f"{row['role']:<13} {row['max_input_tokens']:<8} "
            f"{row['max_output_tokens']:<8} {row['max_rounds']:<7} "
            f"{row['max_output_continuations']:<14} {row['thinking_policy']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit effective LeninBot model/provider runtime policy.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON snapshot.")
    args = parser.parse_args()

    snapshot = build_snapshot()
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))
    else:
        _print_table(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
