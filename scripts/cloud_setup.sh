#!/usr/bin/env bash
# cloud_setup.sh — prepare a fresh checkout for development without this
# server: a Claude Code cloud session, or any lone clone.
#
# Creates venv/ (scripts/run_unit_tests.sh expects it) and replays the
# production venv from requirements.lock.txt with --no-deps, minus torch and
# the packages that exist only for it: the unit suite does not import them, and
# they add gigabytes.
# Idempotent: rerunning only installs what changed. No DB, Redis, Neo4j,
# secrets or frontend checkout is needed; see AGENTS.md "Cloud sessions".
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$(command -v python3.12 || command -v python3)"
[ -x venv/bin/python ] || "$PY" -m venv venv

# The SessionStart hook runs this on every session, including resumes; skip
# the install when the lock has not changed since the last one.
STAMP=venv/.requirements.lock.sha256
if [ -f "$STAMP" ] && sha256sum -c --status "$STAMP" 2>/dev/null; then
    echo "cloud_setup: venv up to date"
    exit 0
fi

REQ="$(mktemp)"
trap 'rm -f "$REQ"' EXIT
grep -vE '^(torch|triton|nvidia-[a-z0-9-]+|sentence-transformers|langchain-huggingface)==' \
    requirements.lock.txt > "$REQ"
venv/bin/pip install --quiet --no-deps -r "$REQ"
sha256sum requirements.lock.txt > "$STAMP"
echo "cloud_setup: venv ready ($(venv/bin/python --version))"
