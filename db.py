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
from dotenv import load_dotenv

load_dotenv()

_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require",
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
        else:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                try:
                    conn.close()
                except Exception:
                    pass
                p.putconn(conn, close=True)
                conn = p.getconn()
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


def query_one(sql: str, params: tuple | list = None) -> dict | None:
    """Execute SQL and return a single row dict, or None."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
