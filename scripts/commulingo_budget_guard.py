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

# Every unit whose spend counts against the shared daily cap. The sum is
# compared to each lane's own COMMULINGO_DAILY_CAP_USD, so the lowest cap among
# them is the real ceiling for the day, not a per-lane allowance. A lane missing
# from this list spends nothing as far as the guard is concerned: when the batch
# moved to the gap queue on 2026-08-09 the three new lanes were invisible here
# and the cap would never have bound.
LANE_UNITS = [
    "leninbot-commulingo-gap",
    "leninbot-commulingo-events",
    "leninbot-commulingo-links",
    "leninbot-commulingo-new",
    "leninbot-commulingo-enrich",
    "leninbot-commulingo-terms",
    "leninbot-commulingo-maintainer",
]
# Anchored to the pretty-printed result JSON only ("^\s*"), not the llm_gateway
# audit lines: each round is audited twice (surface external_sdk + loop) and the
# result JSON restates the run total, so an unanchored match trebled real spend.
# The value needs the exponent branch: json.dumps renders sub-$0.0001 costs as
# e.g. 6.933e-05, and a mantissa-only match read that as $6.93 — one cheap flash
# call then tripped the daily cap and skipped the rest of the window (2026-08-11).
COST = re.compile(r'^\s*"cost_usd": ([0-9.]+(?:[eE][+-]?[0-9]+)?)', re.M)

# Units that never print a result JSON: they call the audited SDK client
# directly (through the llm_proxy like everything else), so their one llm_call
# audit line per call IS the spend record. For the loop lanes above that same
# line is a duplicate of the result JSON and must stay excluded. Until
# 2026-08-14 links was in neither sum and its spend escaped the cap entirely.
DIRECT_UNITS = {"leninbot-commulingo-links"}
LLM_CALL_COST = re.compile(
    r'INFO llm_call \{.*?"cost_usd": ([0-9.]+(?:[eE][+-]?[0-9]+)?)')


def main() -> int:
    cap = float(os.environ.get("COMMULINGO_DAILY_CAP_USD", "1.0"))
    spent = 0.0
    for unit in LANE_UNITS:
        out = subprocess.run(
            ["journalctl", "-u", unit, "--since", "today", "-o", "cat", "--no-pager"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        pattern = LLM_CALL_COST if unit in DIRECT_UNITS else COST
        spent += sum(float(v) for v in pattern.findall(out))
    if spent >= cap:
        print(f"[budget-guard] today's curator spend ${spent:.2f} >= cap ${cap:.2f} — skipping run")
        return 1
    print(f"[budget-guard] today's curator spend ${spent:.2f} < cap ${cap:.2f} — proceeding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
