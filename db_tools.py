"""db_tools.py — Direct SQL tool for the programmer agent.

Gives programmer a first-class DB handle instead of routing every query through
`execute_python + db.query(...)`. Benefits:

- Ergonomics: one tool call per query, SQL visible in the tool_use block rather
  than buried inside a Python snippet.
- Audit: every SQL statement shows up in tool call logs verbatim — much easier
  to review than scrolling through execute_python source.
- Least privilege: this tool can only touch Postgres. It does not import modules,
  read files, or make HTTP calls. `execute_python` still exists for genuine
  code/filesystem/network work.

Exposed to the programmer agent only. SELECT results are row-capped by default
to keep responses small. No operation-type allowlist — the programmer is trusted
at the `execute_python` level already, so adding an artificial DDL/DML gate here
would be theater.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime
from decimal import Decimal

from db import (
    query as db_query,
    execute_returning_rowcount as db_exec,
)

logger = logging.getLogger(__name__)

_READ_KEYWORDS = ("select", "with", "show", "explain", "values", "table")

# DROP cannot be undone inside a transaction boundary we control — once
# committed the object is gone. Even with backups, recovery is hours of
# downtime. Reserved for the operator (scripts/psql-supabase). The block
# applies to the leading keyword only — ALTER ... DROP COLUMN is still
# allowed (different threat level, and sometimes required for migrations).
_BLOCKED_LEADING_KEYWORDS = frozenset({"drop"})


def _classify_sql(sql: str) -> str:
    """Return the first SQL keyword in lowercase (empty string if none)."""
    stripped = sql.strip()
    if not stripped:
        return ""
    # Strip leading comments
    while stripped.startswith("--") or stripped.startswith("/*"):
        if stripped.startswith("--"):
            newline = stripped.find("\n")
            stripped = stripped[newline + 1 :] if newline >= 0 else ""
        else:
            close = stripped.find("*/")
            stripped = stripped[close + 2 :] if close >= 0 else ""
        stripped = stripped.lstrip()
    first = stripped.split(None, 1)[0] if stripped else ""
    return first.lower()


def _json_default(o):
    """JSON encoder fallback for types psycopg2 returns natively."""
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (bytes, memoryview)):
        return f"<binary {len(bytes(o))} bytes>"
    return str(o)


QUERY_DB_TOOL = {
    "name": "query_db",
    "description": (
        "Run a single SQL statement against the project's Postgres (Supabase). "
        "SELECT/WITH/SHOW/EXPLAIN → row list as JSON. INSERT/UPDATE/DELETE → "
        "affected row count. CREATE/ALTER → OK. Use `params` for "
        "parameterized queries (%s placeholders) — never interpolate user input "
        "into the SQL string. Each call runs in its own transaction; exceptions "
        "roll back. **DROP is blocked at the tool level** (irreversible; "
        "operator-only via scripts/psql-supabase). Other destructive ops "
        "(TRUNCATE, UPDATE/DELETE without WHERE) run as issued — check twice."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single SQL statement. Use %s placeholders for parameters.",
            },
            "params": {
                "type": "array",
                "items": {},
                "description": "Positional parameters for %s placeholders. Optional.",
            },
            "max_rows": {
                "type": "integer",
                "description": "Cap on SELECT rows in the response. Default 100, max 1000.",
            },
        },
        "required": ["sql"],
    },
}


async def _exec_query_db(
    sql: str,
    params: list | None = None,
    max_rows: int = 100,
) -> str:
    sql = (sql or "").strip()
    if not sql:
        return "Error: sql is required."
    try:
        max_rows = max(1, min(int(max_rows), 1000))
    except (TypeError, ValueError):
        max_rows = 100
    param_tuple = tuple(params or [])
    kind = _classify_sql(sql)

    if kind in _BLOCKED_LEADING_KEYWORDS:
        return (
            f"Error: {kind.upper()} is blocked on query_db — irreversible. "
            "Ask the operator to run this via scripts/psql-supabase after review."
        )

    try:
        if kind in _READ_KEYWORDS:
            rows = await asyncio.to_thread(db_query, sql, param_tuple)
            total = len(rows)
            truncated = total > max_rows
            shown = rows[:max_rows] if truncated else rows
            body = json.dumps(shown, default=_json_default, ensure_ascii=False, indent=2)
            header = f"{total} row(s){' — truncated to ' + str(max_rows) if truncated else ''}"
            # Cap overall response size too
            if len(body) > 20000:
                body = body[:20000] + "\n…(response truncated at 20000 chars)"
            return f"{header}\n{body}"
        else:
            affected = await asyncio.to_thread(db_exec, sql, param_tuple)
            verb = kind.upper() if kind else "STATEMENT"
            if affected is None or affected < 0:
                return f"OK. {verb} executed."
            return f"OK. {verb} executed; {affected} row(s) affected."
    except Exception as e:
        logger.warning("query_db error: %s\nSQL: %s", e, sql[:500])
        return f"Error: {type(e).__name__}: {e}"


DB_TOOLS = [QUERY_DB_TOOL]
DB_TOOL_HANDLERS = {"query_db": _exec_query_db}
