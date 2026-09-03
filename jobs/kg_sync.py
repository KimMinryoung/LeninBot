"""KG sync entry point — mirror other knowledge stores into the graph.

    python -m jobs.kg_sync --source commulingo[,documents] [--full] [--limit N]
                           [--dry-run] [--notify-on-error]

Each source module exposes ``run(*, since, full, limit, dry_run) -> stats``.
State (watermark per source, last run, last stats) lives in the Postgres
table ``kg_sync_state`` so incremental runs only touch rows changed since the
previous run. A full reconciliation pass (``--full``, or automatically when
the last full pass is older than FULL_EVERY_DAYS) also expires edges whose
source rows disappeared.

Runs under ``systemd/leninbot-kg-sync.service`` nightly; ad-hoc runs need
``LENINBOT_ALLOW_WRITE=1`` for the state table (the graph itself is not
covered by the Postgres write guard — be deliberate).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from db import execute as db_execute, query_one as db_query_one

logger = logging.getLogger(__name__)

SOURCES = ("commulingo", "documents")
FULL_EVERY_DAYS = 7

_table_ensured = False


def _ensure_table() -> None:
    global _table_ensured
    if _table_ensured:
        return
    db_execute("""
        CREATE TABLE IF NOT EXISTS kg_sync_state (
            source       TEXT PRIMARY KEY,
            watermark    TIMESTAMPTZ,
            last_run_at  TIMESTAMPTZ,
            last_full_at TIMESTAMPTZ,
            stats        JSONB
        )
    """)
    _table_ensured = True


def get_state(source: str) -> dict:
    _ensure_table()
    row = db_query_one("SELECT * FROM kg_sync_state WHERE source = %s", (source,))
    return dict(row) if row else {"source": source, "watermark": None, "last_run_at": None,
                                  "last_full_at": None, "stats": None}


def set_state(source: str, *, watermark: datetime, full: bool, stats: dict) -> None:
    _ensure_table()
    db_execute(
        """
        INSERT INTO kg_sync_state (source, watermark, last_run_at, last_full_at, stats)
        VALUES (%s, %s, NOW(), CASE WHEN %s THEN NOW() ELSE NULL END, %s)
        ON CONFLICT (source) DO UPDATE SET
            watermark    = EXCLUDED.watermark,
            last_run_at  = NOW(),
            last_full_at = CASE WHEN %s THEN NOW() ELSE kg_sync_state.last_full_at END,
            stats        = EXCLUDED.stats
        """,
        (source, watermark, full, json.dumps(stats, ensure_ascii=False, default=str), full),
    )


def _load_source(name: str):
    if name == "commulingo":
        from jobs import kg_sync_commulingo as mod
    elif name == "documents":
        from jobs import kg_sync_documents as mod
    else:
        raise ValueError(f"unknown sync source: {name}")
    return mod


def run_source(name: str, *, full: bool = False, limit: int | None = None,
               dry_run: bool = False, since: datetime | None = None, force: bool = False) -> dict:
    """Run one source; returns its stats dict (also persisted unless dry-run)."""
    mod = _load_source(name)
    state = get_state(name)
    started = datetime.now(timezone.utc)

    if not full:
        last_full = state.get("last_full_at")
        if state.get("watermark") is None:
            full = True
            reason = "no watermark"
        elif last_full is None or (started - last_full) > timedelta(days=FULL_EVERY_DAYS):
            full = True
            reason = "periodic full pass"
        else:
            reason = "incremental"
    else:
        reason = "requested"
    since = None if full else (since or state.get("watermark"))

    logger.info("[kg-sync] %s: %s (since=%s, limit=%s, dry_run=%s)", name, reason, since, limit, dry_run)
    t0 = time.monotonic()
    extra = {"force": True} if (force and name == "documents") else {}
    stats = mod.run(since=since, full=full, limit=limit, dry_run=dry_run, **extra)
    stats = dict(stats or {})
    stats.update({"mode": "full" if full else "incremental", "reason": reason,
                  "elapsed_s": round(time.monotonic() - t0, 1), "dry_run": dry_run})
    if not dry_run and not stats.get("error"):
        set_state(name, watermark=started, full=full, stats=stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="commulingo",
                        help="comma-separated: " + ",".join(SOURCES))
    parser.add_argument("--full", action="store_true", help="full reconciliation pass")
    parser.add_argument("--limit", type=int, default=None, help="cap items processed (documents: docs; commulingo: facts)")
    parser.add_argument("--dry-run", action="store_true", help="compute facts, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="documents: re-extract even when the content hash is unchanged (LLM backfill)")
    parser.add_argument("--notify-on-error", action="store_true", help="Telegram notify on failure")
    parser.add_argument("--json", action="store_true", help="print stats as JSON only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    results: dict[str, dict] = {}
    failed = False
    for name in [s.strip() for s in args.source.split(",") if s.strip()]:
        try:
            results[name] = run_source(name, full=args.full, limit=args.limit, dry_run=args.dry_run, force=args.force)
            if results[name].get("error"):
                failed = True
        except Exception as exc:
            logger.exception("[kg-sync] %s failed", name)
            results[name] = {"error": str(exc)}
            failed = True

    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    if failed and args.notify_on_error:
        try:
            sys.path.insert(0, "scripts")
            from _notify import notify_telegram
            lines = [f"⚠️ KG sync 실패 ({datetime.now().strftime('%m-%d %H:%M')})"]
            for name, st in results.items():
                if st.get("error"):
                    lines.append(f"- {name}: {str(st['error'])[:300]}")
            notify_telegram("\n".join(lines))
        except Exception as exc:  # notification is best-effort
            logger.warning("[kg-sync] notify failed: %s", exc)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
