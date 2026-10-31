# Schema Migrations

LeninBot historically used idempotent `ensure_*` functions during service
startup to create and alter PostgreSQL tables. New schema work should move
toward explicit migration entrypoints instead.

Current runner:

```bash
venv/bin/python scripts/schema_migrations.py --list
venv/bin/python scripts/schema_migrations.py
```

The runner currently applies Telegram, research document, publication record,
site publishing, experiential memory, autonomous project, and x402 ledger schema
blocks.

Services do not run startup DDL. Apply this runner before deploying code that
depends on new tables, columns, indexes, or constraints.

The runner needs a write-capable DB login. Outside a systemd service the
production server has no `DB_PASSWORD`: `db.py` falls back to the read-only
`leninbot_ro` login, and `LENINBOT_ALLOW_WRITE=1` alone then stops at the
missing password (`dev_docs/db_migration_plan.md`). In that shell, send the
migration's DDL through `scripts/psql-main`, which loads the service
credential, for example:

```bash
venv/bin/python -c "from telegram.tool_evidence import DDL; print(DDL)" | scripts/psql-main
```
