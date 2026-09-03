#!/usr/bin/env bash
# Install + enable the KG sync/report systemd units (needs root).
#   sudo scripts/install_kg_timers.sh
set -euo pipefail
SRC=/home/grass/leninbot/systemd
for u in leninbot-kg-sync.service leninbot-kg-sync.timer leninbot-kg-report.service leninbot-kg-report.timer; do
  install -m 644 "$SRC/$u" /etc/systemd/system/"$u"
done
systemctl daemon-reload
systemctl enable --now leninbot-kg-sync.timer leninbot-kg-report.timer
systemctl list-timers leninbot-kg-sync.timer leninbot-kg-report.timer --no-pager
