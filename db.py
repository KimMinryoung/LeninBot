"""db.py — Direct PostgreSQL connection pool for AIChatBot.

Replaces Supabase REST API (anon key) with direct pg connection,
enabling proper per-service access control via PostgreSQL roles.
"""

import os
import json
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from secrets_loader import get_secret

_pool: pool.ThreadedConnectionPool | None = None


def _application_name() -> str:
    return os.getenv("DB_APPLICATION_NAME", "leninbot-api")


def _pool_maxconn() -> int:
    return int(os.getenv("DB_POOL_MAX", "10"))


def _tag_conn(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT set_config(%s, %s, false)", ("application_name", _application_name()))


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        host = os.getenv("DB_HOST")
        port = int(os.getenv("DB_PORT", "5432"))
        dbname = os.getenv("DB_NAME", "postgres")
        user = os.getenv("DB_USER")
        password = get_secret("DB_PASSWORD")
        missing = [
            name for name, value in (
                ("DB_HOST", host),
                ("DB_USER", user),
                ("DB_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing database configuration: "
                + ", ".join(missing)
                + ". For local psql use scripts/psql-supabase; production services "
                "load DB_PASSWORD via systemd LoadCredentialEncrypted."
            )
        _pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=_pool_maxconn(),
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode="require",
            application_name=_application_name(),
        )
    return _pool


@contextmanager
def get_conn():
    """Get a connection from the pool. Auto-returns on exit.

    Detects stale connections (closed by server after idle timeout)
    and transparently replaces them with fresh ones.
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        # Detect stale connections (Supabase/cloud DB closes idle connections)
        if conn.closed:
            p.putconn(conn, close=True)
            conn = p.getconn()
            _tag_conn(conn)
        else:
            try:
                _tag_conn(conn)
            except psycopg2.Error:
                try:
                    conn.close()
                except Exception:
                    pass
                p.putconn(conn, close=True)
                conn = p.getconn()
                _tag_conn(conn)
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        p.putconn(conn)


def query(sql: str, params: tuple | list = None) -> list[dict]:
    """Execute a SELECT and return list of dicts."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def execute(sql: str, params: tuple | list = None) -> None:
    """Execute an INSERT/UPDATE/DELETE (no return value)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def execute_returning_rowcount(sql: str, params: tuple | list = None) -> int:
    """Execute a DML/DDL statement and return affected row count.

    -1 for statements where rowcount is not meaningful (most DDL).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount


def query_one(sql: str, params: tuple | list = None) -> dict | None:
    """Execute SQL and return a single row dict, or None."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
