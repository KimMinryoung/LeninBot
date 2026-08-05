#!/usr/bin/env python3
"""Repoint internal report links whose target slug was never published.

Reports cross-reference each other by slug in their '선행 보고서' line and in
'관련 보고서' lines. Some of those slugs never existed: the imperialism series was
published under a 20260418_ prefix and under bare slugs, but later installments
linked back to it with a 20260419_ prefix and with subtitles in place of the
published slug stem, so twelve references pointed at nothing.

The frontend renders an unresolvable report link as plain text rather than a
404 (utils/markdown.js), so nothing was broken for readers; this restores the
links themselves. The publish gate in jobs/autonomous_publication_controls.py now
rejects new references to slugs that do not exist.

Usage:
  python scripts/fix_broken_report_links.py            # report what would change
  python scripts/fix_broken_report_links.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import redis

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import execute as db_execute, query as db_query

# Each entry is a wrong target and the published report it meant. The series has
# exactly one installment per episode number, and both the wrong slug and the
# link label carry that number, so the mapping is positional and unambiguous:
# -01-financialization and -01-intro both mean episode 1, which is
# 20260418_imperialism-reconfig-2026-01-intro.
REPLACEMENTS = {
    "/reports/research/20260419_imperialism-reconfig-2026-01-intro.md":
        "/reports/research/20260418_imperialism-reconfig-2026-01-intro.md",
    "/reports/research/20260419_imperialism-reconfig-2026-02-monopoly.md":
        "/reports/research/20260418_imperialism-reconfig-2026-02-monopoly.md",
    "/reports/research/20260419_imperialism-reconfig-2026-03-finance.md":
        "/reports/research/20260418_imperialism-reconfig-2026-03-finance.md",
    "/reports/research/20260419_imperialism-reconfig-2026-01-financialization.md":
        "/reports/research/20260418_imperialism-reconfig-2026-01-intro.md",
    "/reports/research/20260419_imperialism-reconfig-2026-02-monopoly-capital.md":
        "/reports/research/20260418_imperialism-reconfig-2026-02-monopoly.md",
    "/reports/research/20260419_imperialism-reconfig-2026-03-finance-oligarchy.md":
        "/reports/research/20260418_imperialism-reconfig-2026-03-finance.md",
}

FIELDS = ("markdown", "markdown_en")


def clear_frontend_research_cache() -> None:
    """Drop the frontend's rendered-report cache.

    routes/reports.js caches rendered research permanently, so a corrected
    markdown row is invisible until those keys go. Same patterns the translation
    job clears after it writes.
    """
    redis_url = os.getenv("REDIS_URL") or "redis://127.0.0.1:6379"
    try:
        client = redis.Redis.from_url(redis_url)
        keys: list[bytes] = []
        for pattern in ("report:research_list:*", "research:*"):
            keys.extend(client.scan_iter(match=pattern))
        if keys:
            print(f"cleared {client.delete(*keys)} frontend cache key(s)")
        else:
            print("no frontend cache keys to clear")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not clear the frontend cache: {exc}", file=sys.stderr)


def published_targets() -> set[str]:
    rows = db_query("SELECT slug, filename FROM research_documents WHERE status = 'public'")
    known: set[str] = set()
    for row in rows:
        for value in (row.get("slug"), row.get("filename")):
            if value:
                known.add(str(value).removesuffix(".md"))
    return known


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    known = published_targets()
    unknown_destinations = [
        dest for dest in set(REPLACEMENTS.values())
        if dest.rsplit("/", 1)[-1].removesuffix(".md") not in known
    ]
    if unknown_destinations:
        print("refusing to run: these replacement targets are not published:")
        for dest in sorted(unknown_destinations):
            print(f"  {dest}")
        return 2

    rows = db_query(
        "SELECT id, slug, markdown, markdown_en FROM research_documents "
        "WHERE markdown LIKE '%%/reports/research/20260419_imperialism%%' "
        "   OR markdown_en LIKE '%%/reports/research/20260419_imperialism%%'"
    )
    total = 0
    for row in rows:
        updates: dict[str, str] = {}
        for field in FIELDS:
            text = row.get(field) or ""
            if not text:
                continue
            fixed = text
            counts: list[str] = []
            for wrong, right in REPLACEMENTS.items():
                hits = fixed.count(wrong)
                if hits:
                    fixed = fixed.replace(wrong, right)
                    counts.append(f"{hits}x {wrong.rsplit('/', 1)[-1]}")
            if fixed != text:
                updates[field] = fixed
                total += sum(int(c.split("x")[0]) for c in counts)
                print(f"{row['slug']} [{field}]: " + ", ".join(counts))
        if updates and args.apply:
            sets = ", ".join(f"{field} = %s" for field in updates)
            db_execute(
                f"UPDATE research_documents SET {sets}, updated_at = NOW() WHERE id = %s",
                (*updates.values(), row["id"]),
            )

    print(f"\n{total} reference(s) {'updated' if args.apply else 'would be updated'}")
    if not args.apply:
        print("re-run with --apply to write")
        return 0
    if total:
        clear_frontend_research_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
