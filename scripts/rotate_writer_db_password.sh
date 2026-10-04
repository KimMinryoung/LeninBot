#!/usr/bin/env bash
# rotate_writer_db_password.sh — new password for the `writer` DB login,
# kept only in the systemd credstore (never in .env).
#
# Run as root on the main server:
#   /home/grass/leninbot/scripts/rotate_writer_db_password.sh
#
# 1. encrypts a fresh password to /etc/credstore.encrypted/writer_db_password.cred
#    (the previous one, if any, is kept as .bak)
# 2. installs the novel-writer-api credential drop-in (scripts/dropins/)
# 3. ALTER ROLE writer through the container's local socket (password on stdin)
# 4. deletes WRITER_DB_PASSWORD from .env
# 5. restarts novel-writer-api, the only service that logs in as `writer`
# Backups use `docker exec pg_dump` and need no password. Safe to rerun.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: systemd-creds and /etc/systemd need it." >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRED=/etc/credstore.encrypted/writer_db_password.cred
DROPIN_SRC="$ROOT/scripts/dropins/novel-writer-api.conf"
DROPIN_DST=/etc/systemd/system/novel-writer-api.service.d/credentials.conf
UNIT=novel-writer-api.service

grep -q '^LoadCredentialEncrypted=writer_db_password:' "$DROPIN_SRC" || {
    echo "$DROPIN_SRC does not mount writer_db_password; regenerate it with scripts/migrate_secrets_to_credstore.py --dropins-only" >&2
    exit 1
}

umask 077
PW="$("$ROOT/venv/bin/python" -c 'import secrets; print(secrets.token_hex(24))')"

[ -e "$CRED" ] && mv -f "$CRED" "$CRED.bak"
printf '%s' "$PW" | systemd-creds encrypt --name=writer_db_password - "$CRED"
chmod 400 "$CRED"
echo "+ $CRED"

install -D -m 644 "$DROPIN_SRC" "$DROPIN_DST"
systemctl daemon-reload
echo "+ $DROPIN_DST"

docker exec -i leninbot-pg psql -U postgres -d postgres -v ON_ERROR_STOP=1 -q <<SQL
ALTER ROLE writer PASSWORD '$PW';
SQL
unset PW
echo "+ ALTER ROLE writer"

sudo -u grass sed -i '/^WRITER_DB_PASSWORD=/d' "$ROOT/.env"
echo "+ WRITER_DB_PASSWORD removed from .env"

systemctl restart "$UNIT"
sleep 5
systemctl is-active "$UNIT"
journalctl -u "$UNIT" --since "1 minute ago" -p warning..err --no-pager | tail -10 || true
echo "done."
