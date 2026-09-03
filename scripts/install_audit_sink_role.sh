#!/usr/bin/env bash
# Switch the LLM proxy's audit sink to an INSERT-only Postgres role (needs root).
#   sudo scripts/install_audit_sink_role.sh
#
# Does, in order (each step idempotent):
#   1. generate a password and register it as the AUDIT_DB_PASSWORD credential
#      (skipped if the credential already exists — then it must be passed as
#      AUDIT_DB_PASSWORD=... so the role gets the same value)
#   2. AUDIT_DB_USER=leninbot_audit in .env
#   3. CREATE/ALTER ROLE + grants via schema_migrations --only audit-sink-role
#   4. enable the LoadCredentialEncrypted line in the proxy unit, install it,
#      daemon-reload, restart the proxy, print /health
# See dev_docs/llm_gateway.md "감사 싱크".
set -euo pipefail
ROOT=/home/grass/leninbot
ROLE=${AUDIT_DB_USER:-leninbot_audit}
CRED=/etc/credstore.encrypted/audit_db_password.cred
UNIT=leninbot-llm-proxy.service
cd "$ROOT"

if [ "$(id -u)" -ne 0 ]; then echo "run with sudo" >&2; exit 1; fi

# 1. credential
if [ -f "$CRED" ]; then
  : "${AUDIT_DB_PASSWORD:?credential $CRED already exists; rerun with AUDIT_DB_PASSWORD=<same value> so the DB role matches}"
  echo "[1/4] credential exists, reusing given AUDIT_DB_PASSWORD"
else
  AUDIT_DB_PASSWORD="$(openssl rand -base64 48 | tr -d '/+=\n' | cut -c1-40)"
  printf '%s' "$AUDIT_DB_PASSWORD" | venv/bin/python scripts/manage_secrets.py add AUDIT_DB_PASSWORD
  echo "[1/4] credential AUDIT_DB_PASSWORD registered"
fi

# 2. .env
if grep -q '^AUDIT_DB_USER=' .env; then
  sed -i "s|^AUDIT_DB_USER=.*|AUDIT_DB_USER=$ROLE|" .env
else
  printf '\nAUDIT_DB_USER=%s\n' "$ROLE" >> .env
fi
echo "[2/4] .env AUDIT_DB_USER=$ROLE"

# 3. role + grants (DB_USER=postgres from .env creates the role)
DB_PASSWORD="$(cat /run/credentials/leninbot-telegram.service/db_password)" \
AUDIT_DB_USER="$ROLE" AUDIT_DB_PASSWORD="$AUDIT_DB_PASSWORD" \
LENINBOT_ALLOW_WRITE=1 PYTHONDONTWRITEBYTECODE=1 \
  venv/bin/python scripts/schema_migrations.py --only audit-sink-role
echo "[3/4] role ready"

# 4. unit
sed -i 's|^#LoadCredentialEncrypted=audit_db_password|LoadCredentialEncrypted=audit_db_password|' "systemd/$UNIT"
install -m 644 "systemd/$UNIT" "/etc/systemd/system/$UNIT"
systemctl daemon-reload
systemctl restart "$UNIT"
sleep 4
echo "[4/4] proxy restarted:"
curl -s -m 5 http://127.0.0.1:8110/health; echo
echo 'expect "audit_sink":"ok (role)". Then: git -C /home/grass/leninbot add systemd/ && git commit'
