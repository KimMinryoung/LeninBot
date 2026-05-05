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
site publishing, and experiential memory schema blocks. Existing service
startup calls remain for compatibility until each service can be switched to
startup-time schema checks only.
