#!/usr/bin/env bash
# move_env_secret_to_credstore.sh — move one secret from .env into the
# systemd credstore and mount it only on the service that uses it.
#
# Run as root on the main server:
#   /home/grass/leninbot/scripts/move_env_secret_to_credstore.sh ROLEPLAY_BOT_TOKEN leninbot-roleplay
#
# Prerequisite (in the repo): NAME is in TIER_A and in that service's
# SERVICE_CREDS (scripts/migrate_secrets_to_credstore.py), and
# scripts/dropins/<service>.conf mounts it.
#
# 1. encrypts the current .env value via scripts/manage_secrets.py add
#    (value goes over stdin, never printed or put on a command line)
# 2. installs the service's credential drop-in
# 3. deletes NAME from .env
# 4. restarts the service
# The value itself does not change; rotate it at its issuer if it was exposed.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: systemd-creds and /etc/systemd need it." >&2
    exit 1
fi
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 NAME SERVICE   (e.g. ROLEPLAY_BOT_TOKEN leninbot-roleplay)" >&2
    exit 2
fi

NAME="$1"
SERVICE="${2%.service}"
LOWER="$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]')"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"
DROPIN_SRC="$ROOT/scripts/dropins/$SERVICE.conf"
DROPIN_DST="/etc/systemd/system/$SERVICE.service.d/credentials.conf"
CRED="/etc/credstore.encrypted/$LOWER.cred"

grep -q "^LoadCredentialEncrypted=$LOWER:" "$DROPIN_SRC" || {
    echo "$DROPIN_SRC does not mount $LOWER; add NAME to SERVICE_CREDS and regenerate drop-ins first" >&2
    exit 1
}

if [ -e "$CRED" ]; then
    echo "= $CRED already exists; keeping it"
else
    grep -q "^$NAME=" "$ENV_FILE" || { echo "$NAME not found in $ENV_FILE" >&2; exit 1; }
    "$ROOT/venv/bin/python" - "$ENV_FILE" "$NAME" <<'PY' | "$ROOT/venv/bin/python" "$ROOT/scripts/manage_secrets.py" add "$NAME"
import sys
from dotenv import dotenv_values
value = dotenv_values(sys.argv[1]).get(sys.argv[2]) or ""
if not value:
    sys.exit(f"{sys.argv[2]} is empty in .env")
sys.stdout.write(value)
PY
fi

install -D -m 644 "$DROPIN_SRC" "$DROPIN_DST"
systemctl daemon-reload
echo "+ $DROPIN_DST"

sudo -u grass sed -i "/^$NAME=/d" "$ENV_FILE"
echo "+ $NAME removed from .env"

systemctl restart "$SERVICE.service"
sleep 5
systemctl is-active "$SERVICE.service"
journalctl -u "$SERVICE.service" --since "1 minute ago" -p warning..err --no-pager | tail -10 || true
echo "done."
