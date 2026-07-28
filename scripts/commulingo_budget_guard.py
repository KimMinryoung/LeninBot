#!/usr/bin/env python3
"""Daily budget guard for the CommuLingo curator lanes.

systemd ExecCondition helper: exits 0 while today's combined curator spend is
under the cap (the run proceeds), 1 once the cap is reached (systemd skips the
run without marking the unit failed).

Spend is read from the lanes' own journal output — every curator run logs a
result JSON containing "cost_usd", the same field commulingo_lane_health.py
aggregates for the daily digest.
"""

import os
import re
import subprocess
import sys

LANE_UNITS = [
    "leninbot-commulingo-new",
    "leninbot-commulingo-enrich",
    "leninbot-commulingo-terms",
    "leninbot-commulingo-maintainer",
]
COST = re.compile(r'"cost_usd": ([0-9.]+)')


def main() -> int:
    cap = float(os.environ.get("COMMULINGO_DAILY_CAP_USD", "1.0"))
    spent = 0.0
    for unit in LANE_UNITS:
        out = subprocess.run(
            ["journalctl", "-u", unit, "--since", "today", "-o", "cat", "--no-pager"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        spent += sum(float(v) for v in COST.findall(out))
    if spent >= cap:
        print(f"[budget-guard] today's curator spend ${spent:.2f} >= cap ${cap:.2f} — skipping run")
        return 1
    print(f"[budget-guard] today's curator spend ${spent:.2f} < cap ${cap:.2f} — proceeding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
