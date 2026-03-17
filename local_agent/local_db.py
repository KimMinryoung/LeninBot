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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT NOT NULL,
            status      TEXT DEFAULT 'pending',
            result      TEXT,
            created_at  TEXT DEFAULT (datetime('now', 'localtime')),
            completed_at TEXT
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
    """)
    conn.commit()

    # One-time backfill: populate crawl_cache with URLs from existing docs/*.txt
    count = conn.execute("SELECT COUNT(*) FROM crawl_cache WHERE content = '[backfilled from docs/]'").fetchone()[0]
    if count == 0:
        n = backfill_crawl_cache_from_docs()
        if n:
            print(f"  Backfilled {n} URLs from docs/ into crawl_cache.")


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
