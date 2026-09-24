#!/usr/bin/env python3
"""Run a CommuLingo maintainer lane independently from the legacy auto lane.

Each systemd unit supplies a distinct COMMULINGO_SUGGESTED_BY value.  The wrapper
keeps its own lock and edit-count provenance, allowing new-person discovery and
existing-person enrichment to research and write concurrently without mistaking
the other lane's successful edit for its own.
"""

from __future__ import annotations

import json
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


# The editorial policy used to be appended here, by monkey-patching the
# maintainer's three task builders from this wrapper. That made the policy a
# property of ONE entry point: `commulingo_people_maintainer.py --candidate X`,
# the documented way to force a card, produced tasks with no policy at all. The
# shared rule now lives in the curator agent's identity and the people-specific
# bullets in the maintainer's own builders, so every caller gets both and this
# wrapper is back to what its docstring says it is — locking and provenance.


if __name__ == "__main__":
    # Focus policy: the new-person lane stands down when new_lane_enabled is
    # false, so its timer keeps firing cheaply while every real maintenance
    # cycle goes to the enrich lane's standard-field work. Re-enable by setting
    # new_lane_enabled back to true in config/commulingo_maintainer.json.
    if SUGGESTED_BY == "commulingo-maintainer-new" and not maintainer.load_config()["new_lane_enabled"]:
        print(json.dumps({"status": "skipped", "reason": "new_lane_enabled=false"}, ensure_ascii=False))
        raise SystemExit(0)
    raise SystemExit(maintainer.main())
