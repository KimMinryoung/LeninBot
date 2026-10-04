#!/usr/bin/env bash
# setup_readonly_db_role.sh — create (or rotate) the leninbot_ro login.
#
# Processes outside a systemd service (operator scripts, restart_guard, the
# MCP gateway) have no DB_PASSWORD. db.py then logs in as leninbot_ro, which
# holds pg_read_all_data and defaults to read-only transactions, so the
# database itself refuses writes. Approved writes still use scripts/psql-main.
#
# Run as the operator account (grass) on the main server:
#   scripts/setup_readonly_db_role.sh
# Rerunning rotates the password. The role replicates to the standby with the
# rest of the cluster; copy the password file there only if it needs it.

set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
    echo "Run as grass, not root: the password file must be readable by grass." >&2
    echo "  sudo -u grass -H $0" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PW_FILE="${DB_RO_PASSWORD_FILE:-$HOME/.config/leninbot/db_ro_password}"

umask 077
mkdir -p "$(dirname "$PW_FILE")"
PW="$("$ROOT/venv/bin/python" -c 'import secrets; print(secrets.token_hex(24))')"

# The password goes over stdin, never on a command line.
"$ROOT/scripts/psql-main" -v ON_ERROR_STOP=1 -q <<SQL
SET client_min_messages = warning;
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'leninbot_ro') THEN
        CREATE ROLE leninbot_ro LOGIN;
    END IF;
END
\$\$;
ALTER ROLE leninbot_ro LOGIN PASSWORD '$PW';
GRANT pg_read_all_data TO leninbot_ro;
ALTER ROLE leninbot_ro SET default_transaction_read_only = on;
SQL

printf '%s\n' "$PW" > "$PW_FILE"
chmod 600 "$PW_FILE"
echo "leninbot_ro ready; password stored in $PW_FILE"
