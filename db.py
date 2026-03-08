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
    """Get a connection from the pool. Auto-returns on exit."""
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
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
