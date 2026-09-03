"""audit_sink.py — one writer for the append-only audit tables.

Both audit ledgers (``llm_audit_log`` from llm/gateway.py, ``tool_audit_log``
from security_gateway/audit.py) used to be inserted by every process that
produced a row, so every process needed the production DB password and every
ad-hoc script silently dropped its rows. Since 2026-09-04 the rows travel to
the LLM proxy (``leninbot-llm-proxy``, localhost-only, already the custodian
of the provider keys) over ``POST /audit/{llm|tool}`` and the proxy is the
only process that INSERTs — through a dedicated INSERT-only role when
``AUDIT_DB_USER`` / ``AUDIT_DB_PASSWORD`` are configured, else through the
main pool.

Three modes, resolved by ``mode()``:
  local  — this process IS the sink (llm_proxy.app calls set_local_sink());
           rows go straight to Postgres.
  proxy  — ``proxy_base`` is set in the gateway policy; rows are POSTed.
  db     — no proxy configured (tests, standalone tools); direct insert as
           before.

The journald JSON line each gateway emits stays the always-on first sink;
this module only replaces the Postgres leg. Nothing here may raise into a
caller: every public function returns a status instead.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger("audit_sink")

MAX_ROWS_PER_REQUEST = 200
MAX_BODY_BYTES = 1_000_000
POST_TIMEOUT_SECONDS = float(os.getenv("LENINBOT_AUDIT_SINK_TIMEOUT", "3"))
POST_ATTEMPTS = 2  # a second try after 0.5s absorbs a proxy restart blip

# Column whitelist per ledger: name → (kind, cap). Unknown columns are
# rejected (schema drift must surface in tests, not become silent nulls);
# over-long strings are truncated, never rejected.
_STR, _INT, _FLOAT, _BOOL = "str", "int", "float", "bool"
TABLES: dict[str, dict] = {
    "llm": {
        "table": "llm_audit_log",
        "required": ("surface", "status"),
        "columns": {
            "surface": (_STR, 40), "caller": (_STR, 200), "provider": (_STR, 40),
            "model": (_STR, 120), "label": (_STR, 200), "tokens_in": (_INT, None),
            "tokens_out": (_INT, None), "cache_read": (_INT, None), "cache_create": (_INT, None),
            "cost_usd": (_FLOAT, None), "latency_ms": (_INT, None), "status": (_STR, 40),
            "error_excerpt": (_STR, 1000),
        },
    },
    "tool": {
        "table": "tool_audit_log",
        "required": ("tool_name", "decision"),
        "columns": {
            "interface": (_STR, 40), "agent_name": (_STR, 120), "user_id": (_STR, 120),
            "is_owner": (_BOOL, None), "task_id": (_STR, 200), "session_id": (_STR, 200),
            "request_id": (_STR, 200), "parent_request_id": (_STR, 200), "scope_type": (_STR, 60),
            "scope_id": (_STR, 200), "chat_log_id": (_INT, None), "tool_name": (_STR, 120),
            "risk_class": (_STR, 40), "decision": (_STR, 40), "enforced": (_BOOL, None),
            "deny_reason": (_STR, 1000), "args_summary": (_STR, 2000), "result_status": (_STR, 40),
            "latency_ms": (_INT, None), "error_excerpt": (_STR, 1000),
        },
    },
}


def _insert_sql(kind: str) -> str:
    spec = TABLES[kind]
    cols = list(spec["columns"])
    return (
        f"INSERT INTO {spec['table']} ({', '.join(cols)}) VALUES ("
        + ", ".join(f"%({c})s" for c in cols) + ")"
    )


_INSERT_SQL = {kind: _insert_sql(kind) for kind in TABLES}


# ── Validation (shared by client and server) ──────────────────────────

def normalize_row(kind: str, row: dict) -> dict:
    """Coerce ``row`` to the ledger's columns. Raises ValueError on an unknown
    ledger, an unknown column, a non-object row or a missing required field."""
    spec = TABLES.get(kind)
    if spec is None:
        raise ValueError(f"unknown audit ledger {kind!r}")
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    unknown = sorted(set(row) - set(spec["columns"]))
    if unknown:
        raise ValueError(f"unknown column(s) for {kind}: {', '.join(unknown)}")
    out: dict = {}
    for col, (typ, cap) in spec["columns"].items():
        val = row.get(col)
        if val is None:
            out[col] = None
            continue
        if typ == _STR:
            val = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)
            out[col] = val if cap is None or len(val) <= cap else val[:cap] + "…"
        elif typ == _INT:
            out[col] = int(val)
        elif typ == _FLOAT:
            out[col] = float(val)
        else:
            out[col] = bool(val)
    for col in spec["required"]:
        if out.get(col) in (None, ""):
            raise ValueError(f"{kind} row missing required field {col!r}")
    return out


def normalize_rows(kind: str, rows) -> list[dict]:
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError("rows must be an object or a list of objects")
    if len(rows) > MAX_ROWS_PER_REQUEST:
        raise ValueError(f"at most {MAX_ROWS_PER_REQUEST} rows per request")
    return [normalize_row(kind, r) for r in rows]


# ── Mode ──────────────────────────────────────────────────────────────

_local_sink = False


def set_local_sink(enabled: bool = True) -> None:
    """Mark this process as the sink (the proxy). Rows are inserted directly
    instead of being POSTed back to ourselves."""
    global _local_sink
    _local_sink = enabled


def is_service_process() -> bool:
    """Same rule as db._writes_allowed minus the ad-hoc opt-in: systemd sets
    INVOCATION_ID for every unit; LENINBOT_SERVICE=1 marks other daemons."""
    return bool(os.getenv("INVOCATION_ID")) or os.getenv("LENINBOT_SERVICE") == "1"


def proxy_base() -> str | None:
    try:
        from llm.gateway import proxy_base as _pb
        return _pb()
    except Exception:
        return None


def mode() -> str:
    if _local_sink:
        return "local"
    forced = (os.getenv("LENINBOT_AUDIT_SINK") or "").strip().lower()
    if forced in ("db", "proxy", "local"):
        return forced
    return "proxy" if proxy_base() else "db"


# ── Server side: Postgres ─────────────────────────────────────────────

_sink_pool = None
_sink_pool_lock = threading.Lock()


def sink_role_configured() -> bool:
    from secrets_loader import get_secret
    return bool(os.getenv("AUDIT_DB_USER")) and bool(get_secret("AUDIT_DB_PASSWORD", ""))


def _sink_pool_get():
    """Connection pool for the INSERT-only audit role; None when the role is
    not configured (callers fall back to db.get_conn)."""
    global _sink_pool
    if not sink_role_configured():
        return None
    if _sink_pool is None:
        with _sink_pool_lock:
            if _sink_pool is None:
                from psycopg2 import pool
                from secrets_loader import get_secret
                _sink_pool = pool.ThreadedConnectionPool(
                    minconn=1, maxconn=int(os.getenv("AUDIT_DB_POOL_MAX", "3")),
                    host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT", "5432")),
                    dbname=os.getenv("DB_NAME", "postgres"), user=os.getenv("AUDIT_DB_USER"),
                    password=get_secret("AUDIT_DB_PASSWORD"), sslmode=os.getenv("DB_SSL", "prefer"),
                    application_name="leninbot-audit-sink",
                )
    return _sink_pool


class _SinkConn:
    """``with _SinkConn() as conn`` — dedicated role when configured, else the
    main pool (db.get_conn, which commits on clean exit)."""

    def __enter__(self):
        p = _sink_pool_get()
        if p is None:
            from db import get_conn
            self._cm = get_conn()
            return self._cm.__enter__()
        self._pool, self._cm = p, None
        self._conn = p.getconn()
        if self._conn.closed:
            p.putconn(self._conn, close=True)
            self._conn = p.getconn()
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        if self._cm is not None:
            return self._cm.__exit__(exc_type, exc, tb)
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._pool.putconn(self._conn)
        return False


def insert_rows(kind: str, rows: list[dict]) -> int:
    """INSERT already-normalized rows in one transaction. Raises on failure."""
    if not rows:
        return 0
    with _SinkConn() as conn:
        with conn.cursor() as cur:
            cur.executemany(_INSERT_SQL[kind], rows)
    return len(rows)


_SPEND_SQL = """
SELECT COALESCE(provider, '?') AS provider, COALESCE(SUM(cost_usd), 0)::float AS spend
  FROM llm_audit_log
 WHERE ts >= date_trunc('day', now() AT TIME ZONE 'utc') AND status IN ('ok', 'error')
 GROUP BY 1
"""


def today_spend() -> dict[str, float]:
    """Today's (UTC) LLM cost by provider, read on the sink connection."""
    with _SinkConn() as conn:
        with conn.cursor() as cur:
            cur.execute(_SPEND_SQL)
            return {str(p): float(s) for p, s in cur.fetchall()}


def sink_health() -> str:
    try:
        with _SinkConn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return "ok" + (" (role)" if sink_role_configured() else " (main pool)")
    except Exception as e:
        return f"error: {e.__class__.__name__}"


ROLE_DDL = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %(role)s) THEN
        EXECUTE format('CREATE ROLE %%I LOGIN PASSWORD %%L', %(role)s, %(password)s);
    ELSE
        EXECUTE format('ALTER ROLE %%I WITH LOGIN PASSWORD %%L', %(role)s, %(password)s);
    END IF;
END $$;
"""
ROLE_GRANTS = (
    "GRANT CONNECT ON DATABASE {db} TO {role}",
    "GRANT USAGE ON SCHEMA public TO {role}",
    "GRANT INSERT ON llm_audit_log, tool_audit_log TO {role}",
    "GRANT USAGE, SELECT ON SEQUENCE llm_audit_log_id_seq, tool_audit_log_id_seq TO {role}",
    "GRANT SELECT ON llm_audit_log TO {role}",  # budget policy reads today's spend
)


def ensure_audit_role() -> str:
    """Create/refresh the INSERT-only sink role from AUDIT_DB_USER +
    AUDIT_DB_PASSWORD. Returns a status line; skips (no error) when the
    credential is absent so schema_migrations can run everywhere."""
    from secrets_loader import get_secret
    role = os.getenv("AUDIT_DB_USER") or ""
    password = get_secret("AUDIT_DB_PASSWORD", "") or ""
    if not role or not password:
        return "audit-sink-role skipped: AUDIT_DB_USER / AUDIT_DB_PASSWORD not configured"
    from db import get_conn
    db = os.getenv("DB_NAME", "postgres")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ROLE_DDL, {"role": role, "password": password})
            for stmt in ROLE_GRANTS:
                cur.execute(stmt.format(db=db, role=role))
    return f"audit-sink-role ok: {role} (INSERT llm_audit_log, tool_audit_log; SELECT llm_audit_log)"


# ── Client side: HTTP to the proxy ────────────────────────────────────

_post_failed_before = False


def post_rows(kind: str, rows: list[dict]) -> bool:
    """POST rows to the proxy sink. Returns False (after logging) on failure;
    the first failure warns, repeats log at DEBUG."""
    global _post_failed_before
    base = proxy_base()
    if not base:
        logger.debug("audit sink: no proxy_base; %d %s row(s) dropped", len(rows), kind)
        return False
    body = json.dumps(rows, ensure_ascii=False, default=str).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/audit/{kind}", data=body, method="POST",
        headers={"content-type": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(POST_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=POST_TIMEOUT_SECONDS) as resp:
                if 200 <= resp.status < 300:
                    _post_failed_before = False
                    return True
                last = RuntimeError(f"HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            last = RuntimeError(f"HTTP {e.code}: {e.read(200)!r}")
            break  # 4xx = our payload is wrong; retrying will not help
        except Exception as e:
            last = e
        if attempt + 1 < POST_ATTEMPTS:
            time.sleep(0.5)
    log = logger.debug if _post_failed_before else logger.warning
    log("audit sink POST failed (%d %s row(s) dropped): %s", len(rows), kind, last)
    _post_failed_before = True
    return False


def fetch_today_spend() -> dict[str, float] | None:
    base = proxy_base()
    if not base:
        return None
    try:
        with urllib.request.urlopen(f"{base}/audit/spend/today", timeout=POST_TIMEOUT_SECONDS) as resp:
            data = json.load(resp)
        spend = data.get("spend") if isinstance(data, dict) else None
        return {str(k): float(v) for k, v in (spend or {}).items()}
    except Exception as e:
        logger.warning("audit sink spend lookup failed (fail-open): %s", e)
        return None
