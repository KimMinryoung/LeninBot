#!/usr/bin/env bash
# Install the weekly stale-secrets check timer.
#
# Run as root:
#   /home/grass/leninbot/scripts/install_stale_secrets_timer.sh
#
# Installs two systemd units (oneshot service + weekly timer), reloads the
# daemon, and enables + starts the timer. Idempotent.

set -euo pipefail

SRC=/home/grass/leninbot/scripts/systemd
DST=/etc/systemd/system

for unit in leninbot-stale-secrets.service leninbot-stale-secrets.timer; do
  cp "${SRC}/${unit}" "${DST}/${unit}"
  echo "installed ${DST}/${unit}"
done

systemctl daemon-reload
systemctl enable --now leninbot-stale-secrets.timer

echo "--- timer status ---"
systemctl list-timers leninbot-stale-secrets.timer --no-pager

echo
echo "To test the notification immediately:"
echo "  sudo systemctl start leninbot-stale-secrets.service"
echo "  sudo journalctl -u leninbot-stale-secrets --no-pager | tail -20"
