#!/bin/bash
# Wrapper: load neo4j_password credential + run fix_legacy_kg_edges.py
# Usage:
#   sudo bash scripts/run_fix_legacy_kg.sh          # dry-run
#   sudo bash scripts/run_fix_legacy_kg.sh --apply  # actually backfill
set -euo pipefail

FLAG="--dry-run"
if [[ "${1:-}" == "--apply" ]]; then
  FLAG=""
fi

CRED=/etc/credstore.encrypted/neo4j_password.cred
if [[ ! -f "$CRED" ]]; then
  echo "ERROR: $CRED not found" >&2
  exit 1
fi

exec systemd-run --pty --uid=grass \
  --working-directory=/home/grass/leninbot \
  --property=LoadCredentialEncrypted=neo4j_password:"$CRED" \
  --setenv=NEO4J_PASSWORD_FROM_CRED=1 \
  /home/grass/leninbot/venv/bin/python \
  /home/grass/leninbot/scripts/fix_legacy_kg_edges.py $FLAG
