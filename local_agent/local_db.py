"""SQLite database for local task queue, crawl cache, and conversation log."""

import os
import sqlite3

_DB_DIR = os.path.join(os.path.dirname(__file__), "data")
_DB_PATH = os.path.join(_DB_DIR, "local.db")

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(_DB_DIR, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            content         TEXT NOT NULL,
            status          TEXT DEFAULT 'pending',
            result          TEXT,
            parent_task_id  INTEGER,
            scratchpad      TEXT,
            depth           INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now', 'localtime')),
            completed_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS crawl_cache (
            url         TEXT PRIMARY KEY,
            content     TEXT,
            title       TEXT,
            crawled_at  TEXT DEFAULT (datetime('now', 'localtime')),
            expires_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS chat_summaries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_start_id  INTEGER NOT NULL,
            chunk_end_id    INTEGER NOT NULL,
            summary         TEXT NOT NULL,
            msg_count       INTEGER NOT NULL,
            created_at      TEXT DEFAULT (datetime('now', 'localtime'))
        );
    """)
    conn.commit()

    # Migrate: add new columns to tasks table if missing
    _migrate_tasks_table(conn)

    # One-time backfill: populate crawl_cache with URLs from existing docs/*.txt
    count = conn.execute("SELECT COUNT(*) FROM crawl_cache WHERE content = '[backfilled from docs/]'").fetchone()[0]
    if count == 0:
        n = backfill_crawl_cache_from_docs()
        if n:
            print(f"  Backfilled {n} URLs from docs/ into crawl_cache.")


def _migrate_tasks_table(conn):
    """Add new columns to tasks table if they don't exist (idempotent)."""
    cursor = conn.execute("PRAGMA table_info(tasks)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    migrations = [
        ("parent_task_id", "INTEGER"),
        ("scratchpad", "TEXT"),
        ("depth", "INTEGER DEFAULT 0"),
    ]
    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_type}")
    conn.commit()


def query(sql: str, params: tuple | list = ()) -> list[dict]:
    conn = _get_conn()
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def execute(sql: str, params: tuple | list = ()) -> int:
    conn = _get_conn()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.lastrowid


def backfill_crawl_cache_from_docs():
    """One-time: parse Source: URLs from docs/*.txt and populate crawl_cache.

    This ensures crawl_site won't re-crawl articles already ingested via
    the old pipeline (crawler_*.py + update_knowledge.py).
    """
    import glob as _glob

    conn = _get_conn()
    docs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    if not os.path.isdir(docs_root):
        return 0

    existing = {r[0] for r in conn.execute("SELECT url FROM crawl_cache").fetchall()}
    inserted = 0

    for txt_path in _glob.glob(os.path.join(docs_root, "**", "*.txt"), recursive=True):
        url, title = None, None
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("Source:"):
                        url = stripped[7:].strip()
                    elif stripped.startswith("Title:"):
                        title = stripped[6:].strip()
                    elif not stripped.startswith(("Author:", "Authors:", "Year:", "")):
                        break
        except Exception:
            continue

        if url and url.startswith("http") and url not in existing:
            conn.execute(
                "INSERT OR IGNORE INTO crawl_cache (url, title, content) VALUES (?, ?, ?)",
                (url, title or os.path.basename(txt_path), "[backfilled from docs/]"),
            )
            existing.add(url)
            inserted += 1

    conn.commit()
    return inserted
