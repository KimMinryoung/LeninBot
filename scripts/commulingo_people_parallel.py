#!/usr/bin/env python3
"""Run a CommuLingo maintainer lane independently from the legacy auto lane.

Each systemd unit supplies a distinct COMMULINGO_SUGGESTED_BY value.  The wrapper
keeps its own lock and edit-count provenance, allowing new-person discovery and
existing-person enrichment to research and write concurrently without mistaking
the other lane's successful edit for its own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUGGESTED_BY = os.getenv("COMMULINGO_SUGGESTED_BY", "").strip()
if SUGGESTED_BY not in {"commulingo-maintainer-enrich", "commulingo-maintainer-new"}:
    raise SystemExit("parallel lane requires a recognized COMMULINGO_SUGGESTED_BY")

# The edit tool reads this during the imported module graph's initialization.
os.environ["COMMULINGO_SUGGESTED_BY"] = SUGGESTED_BY

from scripts import commulingo_people_maintainer as maintainer  # noqa: E402


EDITORIAL_POLICY = """

EDITORIAL POLICY (MANDATORY):
- Do not default to a hostile or reductively anti-Soviet frame. Describe the subject
  in historically grounded terms and preserve factual complexity; criticism is allowed
  when it is relevant and sourced, but polemical anti-Soviet framing is not the voice of
  this dictionary.
- The core account of a person must prioritize achievements, political positions,
  theoretical contributions, institutional work, organizing, and historical significance.
- Do not choose death, execution, suicide, repression, imprisonment, or personal tragedy
  as the main topic merely because it is dramatic. Use such material only when genuinely
  indispensable to understanding the person's historical role, and never as a substitute
  for explaining what the person did and stood for.
- For an existing person, prefer the most important missing achievement, political debate,
  policy, body of work, or organizational contribution. For a new card, lead with historical
  contribution and political identity rather than manner of death.
"""


_base_build_task = maintainer.build_task
_base_build_discovery_task = maintainer.build_discovery_task
_base_build_new_person_task = maintainer.build_new_person_task


def build_task_with_policy(mode: str, candidate: dict | None) -> str:
    return _base_build_task(mode, candidate) + EDITORIAL_POLICY


def build_discovery_task_with_policy() -> str:
    return _base_build_discovery_task() + EDITORIAL_POLICY


def build_new_person_task_with_policy(candidate: dict) -> str:
    return _base_build_new_person_task(candidate) + EDITORIAL_POLICY


def completed_run_count() -> int:
    row = maintainer.db_query_one(
        """SELECT COUNT(*)::int AS n
             FROM commulingo_agent_suggestions
            WHERE suggested_by = %(suggested_by)s
              AND status = 'approved'""",
        {"suggested_by": SUGGESTED_BY},
    )
    return int((row or {}).get("n") or 0)


def latest_lane_edit() -> dict | None:
    return maintainer.db_query_one(
        """SELECT id, target_type, target_id, action, status, confidence, created_at
             FROM commulingo_agent_suggestions
            WHERE suggested_by = %(suggested_by)s
            ORDER BY id DESC LIMIT 1""",
        {"suggested_by": SUGGESTED_BY},
    )


maintainer.LOCK_PATH = Path(f"/tmp/leninbot-{SUGGESTED_BY}.lock")
maintainer.completed_run_count = completed_run_count
maintainer.latest_maintainer_edit = latest_lane_edit
maintainer.build_task = build_task_with_policy
maintainer.build_discovery_task = build_discovery_task_with_policy
maintainer.build_new_person_task = build_new_person_task_with_policy


if __name__ == "__main__":
    raise SystemExit(maintainer.main())
