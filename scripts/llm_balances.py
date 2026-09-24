#!/usr/bin/env python3
"""Show official balances/costs, with estimates only for providers without cost reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.llm_balances import (  # noqa: E402
    ESTIMATED_PROVIDERS,
    LABELS,
    PROVIDERS,
    collect,
    estimated_summary,
    fetch_official,
    format_telegram_report,
    local_spend_sql,
    official_summary,
    parse_query_db_tsv,
    read_local_spend,
)


def print_table(report: dict) -> None:
    days = report["window_days"]
    print(f"{'PROVIDER':<10} OFFICIAL")
    print(f"{'-' * 10} {'-' * 45}")
    for row in report["providers"]:
        print(f"{LABELS[row['provider']]:<10} {official_summary(row['official'], days)}")
        estimate = estimated_summary(row, report)
        if estimate:
            print("  " + estimate)
    if any(row["provider"] == "openai" for row in report["providers"]):
        print("\nOpenAI credit balance: check https://platform.openai.com/settings/organization/billing/overview (not fetched by this report).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--proxy", default="http://127.0.0.1:8110")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = collect(args.proxy, args.days)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
