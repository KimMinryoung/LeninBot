#!/usr/bin/env python3
"""Run one email bridge polling cycle for the systemd timer."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The shared .env sets EMAIL_POLLING_ENABLED=false to disable the legacy
# in-process Telegram poll loop. This dedicated worker must always poll, and
# a systemd Environment= override cannot win against EnvironmentFile=, so
# force it here before email_bridge builds CONFIG at import time.
os.environ["EMAIL_POLLING_ENABLED"] = "true"

from email_bridge import run_polling_cycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one email bridge polling cycle.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum unseen messages to process.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    result = run_polling_cycle(limit=max(1, min(args.limit, 50)))
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
