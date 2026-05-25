#!/usr/bin/env python3
"""Compatibility runner for the autonomous project loop.

The production systemd unit runs venv/bin/python -m autonomous_project.
Keep this script as a safe manual entrypoint for older operator habits and
external references; it must not contain a separate autonomous workflow.
"""

from __future__ import annotations

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from autonomous_project import run_tick  # noqa: E402


def main() -> int:
    try:
        run_tick()
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
