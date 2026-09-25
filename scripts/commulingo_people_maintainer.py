#!/usr/bin/env python3
"""Run one bounded CommuLingo people-dictionary maintenance cycle.

The script deterministically selects one sparse existing person (or periodically asks for
one missing person), then gives only that task to the dedicated DeepSeek V4 Pro curator.
Each stage exposes only its read tools and the narrow terminal write tools it can use.

This file is the command line. The cycle itself lives in
commulingo/people_lane.py and the plumbing every lane shares in
commulingo/lane.py; both are re-exported here for older callers.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Distinguish unattended writes in revisions/suggestion provenance. This must be set before
# runtime_tools.registry imports commulingo.people.
os.environ.setdefault("COMMULINGO_SUGGESTED_BY", "commulingo-maintainer")
# The lane this process writes as. The new/enrich wrappers set the variable
# before importing this module, so the lock follows their lane without
# rebinding module globals.
SUGGESTED_BY = os.environ["COMMULINGO_SUGGESTED_BY"]

# Everything the people lane used to define here, re-exported so that
# `commulingo_people_maintainer.X` keeps resolving. Patching a name on this
# module does not reach code inside the library modules; patch it there.
from commulingo.lane import *  # noqa: E402,F401,F403
from commulingo.lane import _call_curator_stage  # noqa: E402,F401
from commulingo.people_lane import *  # noqa: E402,F401,F403
from commulingo.lane import load_config  # noqa: E402
from commulingo.people_lane import (  # noqa: E402
    pending_person_gap_count, run_once, select_sparse_person, stale_turn,
)
from scripts import commulingo_budget_guard as budget_guard  # noqa: E402

logger = logging.getLogger("commulingo_people_maintainer")

LOCK_PATH = Path(f"/tmp/leninbot-{SUGGESTED_BY}.lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one direct CommuLingo maintenance edit.")
    parser.add_argument("--mode", choices=["auto", "enrich", "new"], default="auto")
    parser.add_argument("--candidate", default="", help="Force an existing person id (enrich mode only).")
    parser.add_argument("--print-candidate", action="store_true", help="Print the selected candidate without calling the model.")
    parser.add_argument("--runs", type=int, default=0,
                        help="Edits to land in this invocation. 0 = config enrich_runs_per_batch "
                             "for an unforced enrich run, otherwise 1.")
    return parser.parse_args()


def planned_runs(args: argparse.Namespace, config: dict) -> int:
    """How many edits this invocation should land.

    Only an unforced `--mode enrich` invocation (the batch's enrich lane) gets
    the multi-run treatment; auto/new and a forced --candidate stay at one.
    """
    if args.runs > 0:
        return args.runs
    if args.mode != "enrich" or args.candidate:
        return 1
    runs = int(config["enrich_runs_per_batch"])
    extra = int(config["enrich_extra_runs_when_gap_empty"])
    if extra > 0:
        try:
            pending = pending_person_gap_count()
        except Exception as exc:  # noqa: BLE001 - a failed count must not cost the base runs
            logger.warning("could not count the person gap queue: %s", exc)
            pending = -1
        if pending == 0:
            logger.info("person gap queue is empty; adding %d enrich run(s)", extra)
            runs += extra
    return runs


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_config()
    lock_file = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("another maintainer run is active; exiting")
        return 0

    if args.print_candidate:
        print(json.dumps(select_sparse_person(
            config["recent_days"], args.candidate, config["incomplete_recent_days"],
            config["enrich_non_soviet_revolutionaries"],
            prefer_stale=stale_turn(config), stale_days=config["stale_priority_days"],
        ), ensure_ascii=False, default=str, indent=2))
        return 0

    runs = planned_runs(args, config)
    time_budget = int(config["enrich_batch_time_budget_sec"])
    started = time.monotonic()
    failures = 0
    for index in range(runs):
        if index > 0:
            elapsed = time.monotonic() - started
            if time_budget > 0 and elapsed > time_budget:
                logger.info("time budget spent (%.0fs > %ds); stopping after %d of %d run(s)",
                            elapsed, time_budget, index, runs)
                break
            # Each run's result JSON is what the guard sums, so re-check between
            # runs rather than trusting the unit's one ExecCondition at start.
            if budget_guard.main() != 0:
                logger.info("daily cap reached; stopping after %d of %d run(s)", index, runs)
                break
        try:
            result = asyncio.run(run_once(mode=args.mode, candidate_id=args.candidate, config=config))
        except Exception as exc:
            # Logged as a traceback (the lane digest counts those), then on to
            # the next run: the failed card is on cooldown, the next pick differs.
            # A single-run invocation keeps its old contract and raises.
            if getattr(exc, "summary", None):
                summary = exc.summary
                print(json.dumps({"status": summary["status"], "run_id": summary["run_id"],
                    "cost_usd": summary["cost_usd"], "rounds": summary["rounds_used"],
                    "error": str(exc)[:500]}, ensure_ascii=False, indent=2))
                if summary['status']=='budget_deferred':
                    break
                failures += 1
                continue
            if runs == 1:
                raise
            failures += 1
            logger.exception("run %d of %d failed", index + 1, runs)
            continue
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 1 if failures and failures == runs else 0


if __name__ == "__main__":
    raise SystemExit(main())
