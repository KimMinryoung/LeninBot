#!/bin/bash
# Wrapper: load db_password credential + run the writer regression suite.
# Usage:
#   sudo bash scripts/run_writer_tests.sh
# Full output is printed AND saved to temp_dev/test_writer_improvements_<date>.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRED=/etc/credstore.encrypted/db_password.cred
LOG="$ROOT/temp_dev/test_writer_improvements_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "$CRED" ]]; then
  echo "ERROR: $CRED not found" >&2
  exit 1
fi

systemd-run --pipe --wait --collect \
  -p User=grass \
  -p WorkingDirectory="$ROOT" \
  -p EnvironmentFile="$ROOT/.env" \
  -p LoadCredentialEncrypted=db_password:"$CRED" \
  "$ROOT/venv/bin/python" "$ROOT/temp_dev/test_writer_improvements.py" 2>&1 | tee "$LOG"

echo "log saved: $LOG"
