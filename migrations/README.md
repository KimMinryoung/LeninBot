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
