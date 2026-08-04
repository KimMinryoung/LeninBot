#!/usr/bin/env bash
# run_unit_tests.sh — run the tests/ unit suite (stdlib unittest, no pytest dep).
#
# Unlike scripts/run_smokes.sh these are hermetic: no API keys, no DB, no
# Redis. Fake clients + patched executors only. Fast enough to run on every
# change to the agent loops.
#
# Usage:
#   scripts/run_unit_tests.sh              # whole suite
#   scripts/run_unit_tests.sh claude_loop  # only test files matching the pattern
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Keep the suite hermetic: the LLM gateway's journald sink still logs, but no
# DB writer thread is spawned and no insert is attempted.
export LENINBOT_LLM_AUDIT_DB=0

PATTERN="${1:-}"
if [ -n "$PATTERN" ]; then
    exec venv/bin/python -m unittest discover tests -p "test_*${PATTERN}*.py" -v
fi
exec venv/bin/python -m unittest discover tests -v
