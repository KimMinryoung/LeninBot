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

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CRED=/etc/credstore.encrypted/neo4j_password.cred
if [[ ! -f "$CRED" ]]; then
  echo "ERROR: $CRED not found" >&2
  exit 1
fi

exec systemd-run --pty --uid=grass \
  --working-directory="$ROOT" \
  --property=LoadCredentialEncrypted=neo4j_password:"$CRED" \
  --setenv=NEO4J_PASSWORD_FROM_CRED=1 \
  "$ROOT/venv/bin/python" \
  "$ROOT/scripts/fix_legacy_kg_edges.py" $FLAG
