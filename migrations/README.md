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
site publishing, experiential memory, and x402 ledger schema blocks.

Existing service startup calls remain for compatibility. After the explicit
runner is part of the deploy flow, set `LENINBOT_SKIP_STARTUP_DDL=true` for
Telegram to skip the legacy startup DDL path and rely on this command instead.
