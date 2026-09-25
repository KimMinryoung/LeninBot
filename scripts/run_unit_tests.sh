#!/usr/bin/env bash
# run_unit_tests.sh — run the tests/ unit suite: stdlib unittest, plus pytest for
# the pytest-style modules unittest cannot run.
#
# Unlike scripts/run_smokes.sh these are hermetic: no API keys, no DB, no
# Redis. Fake clients + patched executors only. Fast enough to run on every
# change to the agent loops.
#
# Usage:
#   scripts/run_unit_tests.sh              # whole suite
#   scripts/run_unit_tests.sh claude_loop  # only test files matching the pattern
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Catch names referenced only inside rarely executed workers before deployment.
venv/bin/python scripts/check_python_names.py

# Keep the suite hermetic: the LLM gateway's journald sink still logs, but no
# DB writer thread is spawned and no insert is attempted.
export LENINBOT_LLM_AUDIT_DB=0

PATTERN="${1:-}"
# unittest imports pytest-style modules but never runs their module-level
# test functions; those files also run under pytest below.
mapfile -t PYTEST_FILES < <(grep -lE '^(import pytest|from pytest)' tests/test_*${PATTERN}*.py 2>/dev/null || true)

rc=0
venv/bin/python -m unittest discover tests -p "test_*${PATTERN}*.py" -v || {
    status=$?
    # 5 = no tests ran: fine when the pattern only matched pytest-style files.
    if [ "$status" -ne 5 ] || [ "${#PYTEST_FILES[@]}" -eq 0 ]; then rc=1; fi
}
if [ "${#PYTEST_FILES[@]}" -gt 0 ]; then
    venv/bin/python -m pytest -q -p no:cacheprovider "${PYTEST_FILES[@]}" || rc=1
fi
exit $rc
