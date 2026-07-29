#!/usr/bin/env bash
# refresh_test_db.sh — (re)create schema-only test clones of the production
# databases inside the leninbot-pg container.
#
#   leninbot → leninbot_test    (main DB;  use with DB_NAME=leninbot_test)
#   writer   → writer_test      (writer DB; use with WRITER_DB_NAME=writer_test)
#
# Test scripts that need DB writes should point at these clones; db.py's
# read-only guard allows writes to any *_test database without extra flags.
# Schema only — no production data is copied. Rerun any time the schema
# changes or a test leaves the clone dirty.

set -euo pipefail

CONTAINER=leninbot-pg

for pair in "leninbot:leninbot_test" "writer:writer_test"; do
    src=${pair%%:*}
    dst=${pair##*:}
    docker exec "$CONTAINER" psql -q -U postgres -v ON_ERROR_STOP=1 \
        -c "DROP DATABASE IF EXISTS ${dst} WITH (FORCE)" \
        -c "CREATE DATABASE ${dst}"
    docker exec "$CONTAINER" sh -c \
        "pg_dump -U postgres --schema-only ${src} | psql -q -U postgres -v ON_ERROR_STOP=1 -d ${dst}"
    echo "refreshed ${dst} from ${src} (schema only)"
done
