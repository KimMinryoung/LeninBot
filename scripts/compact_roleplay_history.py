#!/usr/bin/env python3
"""Compact existing roleplay state_history rows in place (one-time, idempotent).

New rows are written compact by roleplay.memory.compact_audit; this rewrites
rows written before that change, then VACUUMs the file so the space returns.
Safe while leninbot-roleplay runs: each batch is one BEGIN IMMEDIATE write.

Usage:
    scripts/compact_roleplay_history.py            # compact and vacuum
    scripts/compact_roleplay_history.py --dry-run  # report sizes only
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roleplay.memory import MEMORY_PATH, compact_audit  # noqa: E402

BATCH = 50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--path", type=Path, default=MEMORY_PATH)
    args = parser.parse_args()
    conn = sqlite3.connect(args.path, timeout=30, isolation_level=None)
    try:
        rows = conn.execute("SELECT id, payload FROM state_history").fetchall()
        before_bytes = sum(len(p) for _, p in rows)
        updates = []
        for row_id, payload in rows:
            record = json.loads(payload)
            if "before" not in record and "staged_records" not in record:
                continue
            updates.append((json.dumps(compact_audit(record), ensure_ascii=False), row_id))
        after_bytes = before_bytes - sum(len(dict(rows)[i]) - len(p) for p, i in updates)
        print(f"rows={len(rows)} to_compact={len(updates)} payload {before_bytes / 1e6:.1f} MB -> {after_bytes / 1e6:.1f} MB")
        if args.dry_run or not updates:
            return 0
        for start in range(0, len(updates), BATCH):
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany("UPDATE state_history SET payload = ? WHERE id = ?", updates[start:start + BATCH])
            conn.execute("COMMIT")
        conn.execute("VACUUM")
        print(f"file size now {args.path.stat().st_size / 1e6:.1f} MB")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
