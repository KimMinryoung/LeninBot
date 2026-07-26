#!/usr/bin/env python3
"""Daily health digest for the CommuLingo curation lanes.

Each lane is a oneshot systemd unit that prints a JSON summary per run and
raises on a bad run. Nothing read those tracebacks: the glossary lane failed
roughly one run in five for a full day before anyone noticed, because a failure
looks exactly like a success from outside the journal. This tallies the last
window per lane and, with --notify, sends Telegram only when a lane is actually
unhealthy — a quiet run means the lanes are fine.

Usage:
  scripts/commulingo_lane_health.py                 # digest to stdout
  scripts/commulingo_lane_health.py --since=-6h     # '=' required: the value starts with '-'
  scripts/commulingo_lane_health.py --notify        # daily timer mode
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LANES = {
    "enrich": "leninbot-commulingo-enrich.service",
    "new": "leninbot-commulingo-new.service",
    "terms": "leninbot-commulingo-terms.service",
}

# A lane is unhealthy when it fails this often, or produces nothing at all.
MAX_FAILURE_RATE = 0.10
# The new lane falls back to enrichment when discovery cannot find a real gap.
# Some fallback is normal; a majority means the candidate pool is exhausted.
MAX_FALLBACK_RATE = 0.35

APPLIED = re.compile(r'^\s*"status": "applied"', re.M)
SKIPPED = re.compile(r'^\s*"status": "skipped"', re.M)
# A barren run — rounds spent without a write and without NO_CANDIDATE. It exits
# clean rather than crashing the unit, so it has to be tallied explicitly or it
# would leave no trace here at all and a dead lane would read as a quiet one.
NO_EDIT = re.compile(r'^\s*"status": "no_edit"', re.M)
FALLBACK = re.compile(r'^\s*"mode": "enrich_fallback"', re.M)
FAILED = re.compile(r"^(?:\S+ )*(?:RuntimeError|ValueError|Exception):", re.M)
COST = re.compile(r'"cost_usd": ([0-9.]+)')


def journal(unit: str, since: str) -> str:
    result = subprocess.run(
        ["journalctl", "-u", unit, "--since", since, "--no-pager", "-o", "cat"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"WARNING: journalctl failed for {unit}: {result.stderr.strip()}",
              file=sys.stderr)
    return result.stdout


def tally(unit: str, since: str) -> dict:
    text = journal(unit, since)
    applied = len(APPLIED.findall(text))
    skipped = len(SKIPPED.findall(text))
    failed = len(FAILED.findall(text))
    no_edit = len(NO_EDIT.findall(text))
    total = applied + skipped + failed + no_edit
    return {
        "applied": applied,
        "skipped": skipped,
        "failed": failed,
        "no_edit": no_edit,
        "fallback": len(FALLBACK.findall(text)),
        "total": total,
        "cost": sum(float(v) for v in COST.findall(text)),
    }


def problems(lane: str, stats: dict) -> list[str]:
    found = []
    if stats["total"] == 0:
        return [f"{lane}: no runs recorded"]
    if stats["applied"] == 0:
        found.append(f"{lane}: {stats['total']} runs, nothing applied")
    if stats["no_edit"] / stats["total"] > MAX_FAILURE_RATE:
        found.append(
            f"{lane}: {stats['no_edit']}/{stats['total']} runs ended with no edit "
            f"(rounds exhausted before the write)"
        )
    failure_rate = stats["failed"] / stats["total"]
    if failure_rate > MAX_FAILURE_RATE:
        found.append(
            f"{lane}: {stats['failed']}/{stats['total']} runs failed "
            f"({failure_rate:.0%})"
        )
    if stats["applied"] and stats["fallback"] / stats["applied"] > MAX_FALLBACK_RATE:
        found.append(
            f"{lane}: {stats['fallback']}/{stats['applied']} edits were "
            f"enrich fallbacks — the new-person pool may be exhausted"
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="-24h", help="journalctl --since value")
    parser.add_argument("--notify", action="store_true",
                        help="send Telegram only when a lane is unhealthy")
    args = parser.parse_args()

    lines, alerts, total_cost = [], [], 0.0
    for lane, unit in LANES.items():
        stats = tally(unit, args.since)
        total_cost += stats["cost"]
        lines.append(
            f"{lane:7} applied {stats['applied']:4}  skipped {stats['skipped']:3}  "
            f"failed {stats['failed']:3}  no_edit {stats['no_edit']:3}  "
            f"fallback {stats['fallback']:3}  "
            f"${stats['cost']:.2f}"
        )
        alerts.extend(problems(lane, stats))

    header = f"[commulingo-lanes] since {args.since} — total ${total_cost:.2f}"
    print(header)
    print("\n".join(lines))
    if alerts:
        print("\nPROBLEMS:")
        print("\n".join(f"  - {a}" for a in alerts))

    if args.notify and alerts:
        sys.path.insert(0, str(ROOT))
        from scripts.commulingo_find_name_variants import notify_telegram
        message = "\n".join([header, *lines, "", "PROBLEMS:",
                             *(f"- {a}" for a in alerts)])
        if notify_telegram(message):
            print(f"[commulingo-lanes] notified {len(alerts)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
