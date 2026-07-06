#!/bin/bash
# Run the writer regression suite with the DB credential.
# Usage:
#   bash scripts/run_writer_tests.sh        # no sudo needed if docker is available
#
# Credential sources, in order:
#   1. DB_PASSWORD already exported / present in .env
#   2. docker read of the decrypted credential mounted for the running
#      leninbot-api service (grass is in the docker group, so no password
#      prompt — docker group membership is root-equivalent by design)
#   3. sudo systemd-run with the encrypted credstore (interactive password)
# Full output is printed AND saved to temp_dev/test_writer_improvements_<date>.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRED=/etc/credstore.encrypted/db_password.cred
LOG="$ROOT/temp_dev/test_writer_improvements_$(date +%Y%m%d_%H%M%S).log"

# Source DB_* from .env without polluting the environment further.
set -a
source <(grep -E '^DB_' "$ROOT/.env" 2>/dev/null || true)
set +a

if [[ -z "${DB_PASSWORD:-}" ]] && docker info >/dev/null 2>&1; then
  DB_PASSWORD="$(docker run --rm -v /run/credentials/leninbot-api.service:/c:ro \
      --entrypoint sh redis -c "tr -d '\n' < /c/db_password" 2>/dev/null || true)"
fi

if [[ -n "${DB_PASSWORD:-}" ]]; then
  DB_PASSWORD="$DB_PASSWORD" "$ROOT/venv/bin/python" \
      "$ROOT/temp_dev/test_writer_improvements.py" 2>&1 | tee "$LOG"
else
  echo "No passwordless credential path worked; falling back to sudo systemd-run." >&2
  [[ -f "$CRED" ]] || { echo "ERROR: $CRED not found" >&2; exit 1; }
  sudo systemd-run --pipe --wait --collect \
    -p User=grass \
    -p WorkingDirectory="$ROOT" \
    -p EnvironmentFile="$ROOT/.env" \
    -p LoadCredentialEncrypted=db_password:"$CRED" \
    "$ROOT/venv/bin/python" "$ROOT/temp_dev/test_writer_improvements.py" 2>&1 | tee "$LOG"
fi

echo "log saved: $LOG"
