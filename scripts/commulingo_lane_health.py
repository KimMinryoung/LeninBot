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

# The lanes the batch actually wants since 2026-08-09 (gap queue first; the old
# new/enrich/terms units are installed but not pulled in — putting them here
# while parked would page "no runs recorded" every morning). Swap this dict back
# in the same commit that re-adds them to leninbot-commulingo-batch.service.
LANES = {
    "gap": "leninbot-commulingo-gap.service",
    "events": "leninbot-commulingo-events.service",
    "links": "leninbot-commulingo-links.service",
}

# Drain lanes clear a queue the other lanes feed instead of picking their own
# subject, so their healthy steady state is the one the curator lanes must
# never show: run after run with nothing to do. They also report in items per
# run ("done: 2 written, 1 skipped"), not one status JSON per run — tallied by
# the curator rules, an idle night reads as "no runs recorded" (paged
# 2026-08-14, links had run 19 times) and a working one as "nothing applied".
DRAIN_LANES = {"links"}

# A lane is unhealthy when it fails this often, or produces nothing at all.
MAX_FAILURE_RATE = 0.10
# The new lane falls back to enrichment when discovery cannot find a real gap.
# Some fallback is normal; a majority means the candidate pool is exhausted.
MAX_FALLBACK_RATE = 0.35
# A tool whose calls are rejected this often is burning paid rounds on retries.
# The July 2026 bulk import spent three days at 56% rejected candidate_select
# calls before anyone read the audit log — this line exists so the next such
# pattern shows up in the next morning's digest instead.
MAX_REJECTION_RATE = 0.20
# Below this many calls a high rate is just noise (1 rejection out of 2 calls).
MIN_CALLS_FOR_REJECTION_ALERT = 5

# Bio-length drift. The curator prompt holds two instructions in tension: write
# the sentences the subject warrants and stop, and count the draft against the
# ceiling before calling. The second was added on 2026-08-01 after the curator
# spent five paid rounds shaving one card (427 -> 401 -> 384 -> 384 against 380).
# The risk it reintroduces is the one that instruction had been removed to
# prevent: once the ceiling is a visible number, "what it warrants" quietly
# becomes "until it is nearly full", and the cards go stilted again. Nothing
# else notices that — it produces no rejection and no failed run, just worse
# prose — so it is measured here or not at all.
BIO_DRIFT_WINDOW_DAYS = 7
# A day's worth of enrichment is ~6 cards, too few to read a distribution from.
MIN_BIOS_FOR_DRIFT_ALERT = 20
# Share of freshly written bios landing in the top tenth of the ceiling. This
# metric read 32% (median 325/380) on 2026-08-01, measured just before the prompt
# change, so 32% is the pre-change baseline to compare against and NOT a healthy
# target — by calendar week the median had already climbed 273 -> 293 -> 312 while
# the prompt still forbade counting characters. The drift predates the change;
# the trip point is set above today's reading to catch it getting worse.
MAX_BIO_TOP_BAND_RATE = 0.40

APPLIED = re.compile(r'^\s*"status": "applied"', re.M)
SKIPPED = re.compile(r'^\s*"status": "skipped"', re.M)
# A barren run — rounds spent without a write and without NO_CANDIDATE. It exits
# clean rather than crashing the unit, so it has to be tallied explicitly or it
# would leave no trace here at all and a dead lane would read as a quiet one.
NO_EDIT = re.compile(r'^\s*"status": "no_edit"', re.M)
FALLBACK = re.compile(r'^\s*"mode": "enrich_fallback"', re.M)
FAILED = re.compile(r"^(?:\S+ )*(?:RuntimeError|ValueError|Exception):", re.M)
# Result-JSON line only, like APPLIED above — unanchored it also swept the two
# llm_gateway audit copies of every round (≈3× real spend), and without the
# exponent branch a sub-$0.0001 cost like 6.933e-05 read as $6.93.
COST = re.compile(r'^\s*"cost_usd": ([0-9.]+(?:[eE][+-]?[0-9]+)?)', re.M)
ROUNDS = re.compile(r'"rounds": ([0-9]+)')
# One per drain-lane run, items not runs. Matches the final summary only, not
# the per-batch "progress: N written" lines, which restate the same items.
DONE = re.compile(r"\bdone: (\d+) written, (\d+) skipped\b")
# Drain lanes print no result JSON; their spend exists only as the audited SDK
# client's llm_call line (one per call, still via the llm_proxy). Never applied
# to the loop lanes, where the same line duplicates the result JSON.
LLM_CALL_COST = re.compile(
    r'INFO llm_call \{.*?"cost_usd": ([0-9.]+(?:[eE][+-]?[0-9]+)?)')


def journal(unit: str, since: str) -> str:
    result = subprocess.run(
        ["journalctl", "-u", unit, "--since", since, "--no-pager", "-o", "cat"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"WARNING: journalctl failed for {unit}: {result.stderr.strip()}",
              file=sys.stderr)
    return result.stdout


def tally_drain(unit: str, since: str) -> dict:
    text = journal(unit, since)
    runs = DONE.findall(text)
    failed = len(FAILED.findall(text))
    return {
        "applied": sum(int(written) for written, _ in runs),
        "skipped": sum(int(skipped) for _, skipped in runs),
        "failed": failed,
        "no_edit": 0,
        "fallback": 0,
        # A crashed run never prints its done line, so it is counted on top.
        "total": len(runs) + failed,
        "cost": sum(float(v) for v in LLM_CALL_COST.findall(text)),
        "rounds": [],
    }


def drain_problems(lane: str, stats: dict) -> list[str]:
    found = []
    if stats["total"] == 0:
        return [f"{lane}: no runs recorded"]
    if stats["failed"] / stats["total"] > MAX_FAILURE_RATE:
        found.append(
            f"{lane}: {stats['failed']}/{stats['total']} runs failed"
        )
    # Skipped rows stay in the queue and are re-selected next run, so a few
    # skips with writes alongside are a transient; skips with NO writes are
    # the same rows failing every twenty minutes.
    if stats["skipped"] and not stats["applied"]:
        found.append(
            f"{lane}: {stats['skipped']} skips and nothing written — "
            f"the same rows may be failing every run"
        )
    return found


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
        # Waste, not failure. Everything above counts runs that went wrong; a
        # run can also succeed and still spend most of itself on nothing —
        # walking checklist branches that cannot apply, re-fetching state the
        # commission already stated. Nothing in this digest saw that until a
        # six-step enrich checklist whose first four steps applied to 1 card in
        # 1,236 was found by hand on 2026-08-02. Rounds are the unit that costs
        # money, so rounds-per-applied-edit is the number to watch.
        "rounds": [int(v) for v in ROUNDS.findall(text)],
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


def tool_rejections(since: str) -> tuple[list[str], list[str]]:
    """Per-tool rejection counts from the audit log, plus threshold alerts.

    The lane tallies above only see whole runs; a run that succeeds after three
    rejected writes looks healthy there while quietly paying for four calls.
    Queried through `docker exec` like the journal is queried through
    journalctl: this unit carries no db_password credential, and the digest
    must not start needing one. Fail-soft either way.
    """
    match = re.fullmatch(r"-(\d+)h", since.strip())
    hours = int(match.group(1)) if match else 24
    # Grouped by status as well as tool: a bare count cannot tell a validation
    # rejection the model fixes next round from an unknown_tool deny (a typo'd
    # name — "fetfetch_url rejected 1/1" read as a broken tool on 2026-08-14
    # until someone opened the audit log).
    sql = (
        "SELECT tool_name, result_status, count(*)"
        "  FROM tool_audit_log"
        " WHERE agent_name = 'commulingo_curator'"
        f"  AND ts > now() - interval '{hours} hours'"
        " GROUP BY 1, 2"
    )
    result = subprocess.run(
        ["docker", "exec", "leninbot-pg", "psql", "-U", "postgres", "-d", "leninbot",
         "-t", "-A", "-F", "|", "-c", sql],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return [f"(tool rejection stats unavailable: {result.stderr.strip()[:200]})"], []
    by_tool: dict[str, dict[str, int]] = {}
    for raw in result.stdout.strip().splitlines():
        parts = raw.split("|")
        if len(parts) != 3:
            continue
        by_tool.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    lines, alerts = [], []
    for tool, statuses in sorted(
        by_tool.items(),
        key=lambda item: -sum(v for k, v in item[1].items() if k != "ok"),
    ):
        total = sum(statuses.values())
        rejected = total - statuses.get("ok", 0)
        if not rejected:
            continue
        rate = rejected / total
        why = ", ".join(
            f"{status} {count}"
            for status, count in sorted(statuses.items(), key=lambda kv: -kv[1])
            if status != "ok"
        )
        lines.append(f"{tool:28} rejected {rejected:4}/{total:<5} ({rate:.0%})  [{why}]")
        if total >= MIN_CALLS_FOR_REJECTION_ALERT and rate > MAX_REJECTION_RATE:
            alerts.append(
                f"tool {tool}: {rejected}/{total} calls rejected ({rate:.0%}, {why}) — "
                f"paid rounds are being burned on retries"
            )
    return lines, alerts


def bio_length_drift() -> tuple[list[str], list[str]]:
    """How close freshly written bios are sitting to their hard ceiling.

    Queried through `docker exec` for the same reason the rejection stats are:
    this unit carries no db_password credential and must not start needing one.
    Fail-soft — a digest that cannot reach the DB still reports the lanes.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from runtime_tools.commulingo_people import FIELD_LIMITS
        ceiling = FIELD_LIMITS["bio"][0]
    except Exception as exc:  # never let a stats line break the digest
        return [f"(bio length stats unavailable: {exc})"], []
    # The band is derived from the ceiling, never restated: FIELD_LIMITS is the
    # single source for these numbers and a second copy would drift from it.
    top_band_floor = ceiling * 9 // 10
    sql = (
        "WITH last_touch AS ("
        "  SELECT p.id, length(p.bio_ko) AS n"
        "    FROM commulingo_people p"
        "    JOIN commulingo_people_revisions r ON r.entity_id = p.id"
        "   WHERE r.changed_by LIKE 'commulingo-maintainer%'"
        "     AND COALESCE(p.bio_ko, '') <> ''"
        f"    AND r.created_at > now() - interval '{BIO_DRIFT_WINDOW_DAYS} days'"
        "   GROUP BY p.id, p.bio_ko)"
        " SELECT count(*),"
        "        COALESCE(percentile_disc(0.5) WITHIN GROUP (ORDER BY n), 0),"
        f"       count(*) FILTER (WHERE n > {top_band_floor})"
        "   FROM last_touch"
    )
    result = subprocess.run(
        ["docker", "exec", "leninbot-pg", "psql", "-U", "postgres", "-d", "leninbot",
         "-t", "-A", "-F", "|", "-c", sql],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return [f"(bio length stats unavailable: {result.stderr.strip()[:200]})"], []
    parts = result.stdout.strip().split("|")
    if len(parts) != 3:
        return [], []
    total, median, top_band = int(parts[0]), int(parts[1]), int(parts[2])
    if total == 0:
        return [], []
    rate = top_band / total
    lines = [
        f"bio length ({BIO_DRIFT_WINDOW_DAYS}d, n={total}): median {median} / "
        f"{ceiling}, {top_band} ({rate:.0%}) in the top band (>{top_band_floor})"
    ]
    alerts = []
    if total >= MIN_BIOS_FOR_DRIFT_ALERT and rate > MAX_BIO_TOP_BAND_RATE:
        alerts.append(
            f"bio length: {top_band}/{total} ({rate:.0%}) of bios written in the last "
            f"{BIO_DRIFT_WINDOW_DAYS} days sit above {top_band_floor} of {ceiling} — "
            f"the curator may be writing toward the ceiling instead of to the subject"
        )
    return lines, alerts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="-24h", help="journalctl --since value")
    parser.add_argument("--notify", action="store_true",
                        help="send Telegram only when a lane is unhealthy")
    args = parser.parse_args()

    lines, alerts, total_cost = [], [], 0.0
    waste_lines = []
    for lane, unit in LANES.items():
        drain = lane in DRAIN_LANES
        stats = tally_drain(unit, args.since) if drain else tally(unit, args.since)
        total_cost += stats["cost"]
        lines.append(
            f"{lane:7} applied {stats['applied']:4}  skipped {stats['skipped']:3}  "
            f"failed {stats['failed']:3}  no_edit {stats['no_edit']:3}  "
            f"fallback {stats['fallback']:3}  "
            f"${stats['cost']:.2f}"
            + (f"  ({stats['total']} runs)" if drain else "")
        )
        alerts.extend(drain_problems(lane, stats) if drain else problems(lane, stats))
        if stats["applied"] and stats["rounds"]:
            rounds = sorted(stats["rounds"])
            median = rounds[len(rounds) // 2] if rounds else 0
            waste_lines.append(
                f"{lane:7} {stats['cost'] / stats['applied']:.4f} $/edit  "
                f"{sum(rounds) / stats['applied']:5.1f} rounds/edit  "
                f"(median {median} per run)"
            )
    if waste_lines:
        lines.append("cost of what worked:")
        lines.extend(f"  {line}" for line in waste_lines)

    rejection_lines, rejection_alerts = tool_rejections(args.since)
    if rejection_lines:
        lines.append("tool rejections:")
        lines.extend(f"  {line}" for line in rejection_lines)
    alerts.extend(rejection_alerts)

    drift_lines, drift_alerts = bio_length_drift()
    lines.extend(drift_lines)
    alerts.extend(drift_alerts)

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
