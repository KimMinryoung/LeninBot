#!/usr/bin/env python3
"""Weekly KG health report — growth, shape, sync lag, search usage.

    venv/bin/python scripts/kg_weekly_report.py [--notify] [--json]

Runs from systemd/leninbot-kg-report.timer (Mon 09:30 KST) and posts to the
same Telegram channel as the integrity check (scripts/_notify). Read-only.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--notify", action="store_true", help="Send the report to Telegram.")
    parser.add_argument("--json", action="store_true", help="Print raw metrics JSON.")
    parser.add_argument("--days", type=int, default=14, help="Usage window (tool_audit_log).")
    args = parser.parse_args()

    from kg_runtime.metrics import collect_kg_metrics, format_report

    metrics = collect_kg_metrics(usage_days=args.days)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    report = format_report(metrics)
    print(report)
    if args.notify:
        from _notify import notify_telegram
        if not notify_telegram(report):
            print("WARNING: Telegram notify failed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
