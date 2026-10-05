"""Bounded SQL reads shared by operator collectors. Never runs schema setup."""


def query(sql, params=()):
    from db import get_conn
    from psycopg2.extras import RealDictCursor
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            cur.execute("SET LOCAL statement_timeout = '5s'")
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
