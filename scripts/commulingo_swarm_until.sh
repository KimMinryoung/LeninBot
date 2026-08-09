#!/usr/bin/env bash
# Run one curator lane or gap worker repeatedly until a wall-clock deadline.
#
# The loops this replaces were bounded by a run count (`seq 1 60`). A count is
# not what anyone actually wants — it ends at an unpredictable time, and when it
# ends nothing says so. Two person workers finished their sixty on 2026-08-09
# and sat dead for over an hour before anyone noticed. A deadline ends when it
# is meant to and writes a line saying it did.
#
# Usage: commulingo_swarm_until.sh <deadline-epoch> <log-tag> <script> [args...]
set -uo pipefail
cd /home/grass/leninbot
DEADLINE=$1; TAG=$2; shift 2
S=/home/grass/leninbot/logs/commulingo-swarm
STOP=/home/grass/leninbot/data/commulingo-swarm.stop
runs=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    [ -f "$STOP" ] && { echo "[$TAG] stop file after $runs run(s) $(date -Is)" >> "$S/$TAG.log"; exit 0; }
    runs=$((runs + 1))
    env CREDENTIALS_DIRECTORY=/run/credentials/leninbot-api.service \
        LENINBOT_ALLOW_WRITE=1 PYTHONUNBUFFERED=1 \
        timeout 1500 venv/bin/python "$@" >> "$S/$TAG.jsonl" 2>> "$S/$TAG.log"
    sleep 8
done
echo "[$TAG] deadline reached after $runs run(s) $(date -Is)" >> "$S/$TAG.log"
echo "$TAG $runs" >> "$S/_deadline-finished.log"
