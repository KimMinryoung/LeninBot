#!/usr/bin/env python3
"""Restart LeninBot services only after restart_guard passes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.restart_guard import check_restart_allowed


UNITS = {
    "telegram": ["leninbot-telegram.service"],
    "api": ["leninbot-api.service"],
    "all": ["leninbot-api.service", "leninbot-telegram.service"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=sorted(UNITS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    allowed, blockers = check_restart_allowed(args.service)
    if not allowed and not args.force:
        print(f"Restart refused for {args.service}: active work is in progress.")
        print("\n".join(blockers))
        print("Use --force only if losing that in-flight work is acceptable.")
        return 2

    if blockers and args.force:
        print("Restart guard overridden with --force. Blockers were:")
        print("\n".join(blockers))

    for unit in UNITS[args.service]:
        proc = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", unit],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            print(proc.stdout.strip())
            return proc.returncode
        print(f"restarted {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
