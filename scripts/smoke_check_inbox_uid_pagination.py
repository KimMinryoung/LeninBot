#!/usr/bin/env python3
"""Run mailbox regression coverage against the explicitly selected test clone."""
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    if os.environ.get("MAIL_TEST_DATABASE") != "1" or os.environ.get("DB_NAME") != "leninbot_test":
        raise SystemExit("Set MAIL_TEST_DATABASE=1 DB_NAME=leninbot_test and test DB credentials.")
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", "tests/test_mail_briefing.py", "-q"], cwd=root))
