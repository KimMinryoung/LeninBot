#!/usr/bin/env bash
# Apply the Tier A credentials drop-in to all leninbot services that need it,
# reload systemd, and restart the long-running ones.
#
# Run as root:
#   /home/grass/leninbot/scripts/apply_credentials_dropin.sh
#
# Safe to re-run (cp is idempotent, daemon-reload + restart are no-ops if
# already in place).

set -euo pipefail

DROPIN_SRC=/home/grass/leninbot/scripts/systemd-credentials.conf
if [[ ! -f "$DROPIN_SRC" ]]; then
  echo "ERROR: drop-in source not found at $DROPIN_SRC" >&2
  echo "Run scripts/migrate_secrets_to_credstore.py first." >&2
  exit 1
fi

SERVICES=(
  leninbot-api
  leninbot-telegram
  leninbot-browser
  leninbot-autonomous
  leninbot-experience
  leninbot-kg-backup
)

LONG_RUNNING=(
  leninbot-api
  leninbot-telegram
  leninbot-browser
)

echo "--- installing drop-in ---"
for svc in "${SERVICES[@]}"; do
  dir=/etc/systemd/system/${svc}.service.d
  mkdir -p "$dir"
  cp "$DROPIN_SRC" "$dir/credentials.conf"
  echo "  + $dir/credentials.conf"
done

echo "--- daemon-reload ---"
systemctl daemon-reload

echo "--- restarting long-running services ---"
systemctl restart "${LONG_RUNNING[@]}"

echo "--- status ---"
systemctl is-active "${LONG_RUNNING[@]}"

echo "--- recent warnings/errors (last 1 min) ---"
journalctl $(printf -- '-u %s ' "${LONG_RUNNING[@]}") \
  --since "1 minute ago" -p warning..err --no-pager | tail -20 || true

echo "done."
