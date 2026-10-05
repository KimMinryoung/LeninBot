#!/usr/bin/env python3
"""Check leninbot's database login without changing data (dev_docs/db_app_role.md).

Logs in as the given role and asks the server, never by writing: it can read
and write every table leninbot uses, owns the tables leninbot runs DDL on,
reaches pgvector, and has no access to the frontend's CommuLingo tables.

  APP_DB_PASSWORD=... venv/bin/python scripts/check_app_role.py [--user leninbot_app]
Exit 1 on any failed check.
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg2

LENINBOT_CACHE = ("commulingo_pipeline_sources", "commulingo_pipeline_fetch_cache", "commulingo_pipeline_job_sources")
SHARED_DML = ("posts", "ai_diary", "users", "user_passkeys", "user_fingerprints")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="leninbot_app")
    args = parser.parse_args()
    conn = psycopg2.connect(host=os.getenv("DB_HOST", "127.0.0.1"), port=int(os.getenv("DB_PORT", "5434")),
                            dbname=os.getenv("DB_NAME", "leninbot"), user=args.user,
                            password=os.environ["APP_DB_PASSWORD"])
    conn.set_session(readonly=True)
    cur = conn.cursor()
    failures: list[str] = []

    cur.execute("SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user")
    user, superuser = cur.fetchone()
    if superuser:
        failures.append(f"{user} is a superuser")

    cur.execute("""SELECT tablename, tableowner = current_user,
                          has_table_privilege(current_user, quote_ident(tablename), 'SELECT,INSERT,UPDATE,DELETE')
                   FROM pg_tables WHERE schemaname = 'public' ORDER BY 1""")
    rows = cur.fetchall()
    owned = {t for t, own, _ in rows if own}
    writable = {t for t, _, dml in rows if dml}
    for table in LENINBOT_CACHE + ("research_documents", "chat_logs", "telegram_tasks", "tool_audit_log", "llm_audit_log"):
        if table not in owned:
            failures.append(f"does not own {table}")
    for table in SHARED_DML:
        if table not in writable:
            failures.append(f"cannot read/write {table}")
    leaked = sorted(t for t in writable | owned if t.startswith("commulingo_") and t not in LENINBOT_CACHE)
    if leaked:
        failures.append("has access to CommuLingo tables: " + ", ".join(leaked))
    cur.execute("SELECT has_table_privilege(current_user, 'commulingo_people', 'SELECT')")
    if cur.fetchone()[0]:
        failures.append("can read commulingo_people")

    cur.execute("SELECT '[1,2]'::vector <-> '[1,3]'::vector")
    cur.fetchone()

    print(f"{user}: owns {len(owned)} tables, reads/writes {len(writable)}")
    for failure in failures:
        print("FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
