#!/bin/bash
# Wrapper: resolve DB_PASSWORD the way scripts/psql-main does, then run
# register_state_socialism_term.py.
#
# Usage:
#   bash scripts/run_register_state_socialism.sh          # dry-run
#   bash scripts/run_register_state_socialism.sh --apply  # create the term
#
# The password comes from a running LeninBot service's credential mount when one
# is readable, so the common case needs no sudo. Falls back to the encrypted
# credstore via a transient systemd-run scope, which does.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"
CRED=/etc/credstore.encrypted/db_password.cred
FLAG="${1:-}"

set -a
# shellcheck disable=SC1090
source <(grep -E '^DB_(HOST|PORT|NAME|USER|PASSWORD)=' "$ENV_FILE" || true)
set +a

if [ -z "${DB_PASSWORD:-}" ] && [ -n "${CREDENTIALS_DIRECTORY:-}" ] \
        && [ -r "$CREDENTIALS_DIRECTORY/db_password" ]; then
    DB_PASSWORD="$(<"$CREDENTIALS_DIRECTORY/db_password")"
    export DB_PASSWORD
fi

if [ -z "${DB_PASSWORD:-}" ]; then
    for service in \
        leninbot-telegram.service \
        leninbot-api.service \
        leninbot-browser.service \
        leninbot-experience.service \
        leninbot-autonomous.service
    do
        cred="/run/credentials/${service}/db_password"
        if [ -r "$cred" ]; then
            DB_PASSWORD="$(<"$cred")"
            export DB_PASSWORD
            break
        fi
    done
fi

if [ -z "${DB_PASSWORD:-}" ]; then
    if [ ! -f "$CRED" ]; then
        echo "ERROR: no runtime credential readable and $CRED not found" >&2
        exit 1
    fi
    exec sudo systemd-run --pipe --wait --collect --uid="$(id -un)" \
        --working-directory="$ROOT" \
        --property=EnvironmentFile="$ENV_FILE" \
        --property=LoadCredentialEncrypted=db_password:"$CRED" \
        "$ROOT/venv/bin/python" \
        "$ROOT/scripts/register_state_socialism_term.py" $FLAG
fi

exec "$ROOT/venv/bin/python" "$ROOT/scripts/register_state_socialism_term.py" $FLAG
