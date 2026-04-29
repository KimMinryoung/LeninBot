#!/usr/bin/env python3
"""Translate untranslated public research_documents rows into English.

This is the scheduled DB-backed variant of translate_research_markdown.py.
It reuses the same DeepSeek prompt and validation, but stores translations in
research_documents.markdown_en/title_en/summary_en instead of writing files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import redis

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research_store
from db import execute as db_execute, query as db_query
from scripts.translate_research_markdown import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    _call_deepseek,
    _validate_translation,
)


def _select_rows(*, limit: int, max_chars: int, force: bool) -> list[dict[str, Any]]:
    research_store.ensure_research_table()
    where = "status = 'public'"
    if not force:
        where += " AND NULLIF(BTRIM(COALESCE(markdown_en, '')), '') IS NULL"
    params: list[Any] = []
    limit_sql = ""
    if max_chars > 0:
        where += " AND LENGTH(markdown) <= %s"
        params.append(max_chars)
    if limit > 0:
        limit_sql = "LIMIT %s"
        params.append(limit)
    return db_query(
        f"""
        SELECT id, slug, filename, title, summary, markdown, updated_at
          FROM research_documents
         WHERE {where}
         ORDER BY updated_at DESC, id DESC
         {limit_sql}
        """,
        tuple(params),
    )


def _update_translation(row: dict[str, Any], translated_markdown: str) -> None:
    title_en = research_store.extract_title(translated_markdown, row.get("title") or row["slug"])
    summary_en = research_store.extract_excerpt(translated_markdown)
    db_execute(
        """
        UPDATE research_documents
           SET markdown_en = %s,
               title_en = %s,
               summary_en = %s
         WHERE id = %s
        """,
        (translated_markdown, title_en, summary_en, row["id"]),
    )


def _clear_frontend_research_cache() -> None:
    redis_url = os.getenv("REDIS_URL") or "redis://127.0.0.1:6379"
    try:
        client = redis.Redis.from_url(redis_url)
        keys: list[bytes] = []
        for pattern in ("report:research_list:*", "research:*"):
            keys.extend(client.scan_iter(match=pattern))
        if keys:
            deleted = client.delete(*keys)
            print(f"cleared redis cache keys: {deleted}")
    except Exception as exc:
        print(f"warning: could not clear redis cache: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate missing research_documents English columns.")
    parser.add_argument("--limit", type=int, default=2, help="Rows to translate per run. Use 0 for no limit.")
    parser.add_argument("--max-chars", type=int, default=60_000, help="Skip rows longer than this many source chars. Use 0 for no cap.")
    parser.add_argument("--force", action="store_true", help="Retranslate even when markdown_en already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only list selected rows; do not call the translation API.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-hangul-ratio", type=float, default=0.03)
    args = parser.parse_args()

    rows = _select_rows(limit=args.limit, max_chars=args.max_chars, force=args.force)
    print(f"selected {len(rows)} research document(s)")
    if args.dry_run:
        for row in rows:
            print(f"would translate {row['slug']} ({len(row['markdown']):,} chars)")
        return 0

    failures = 0
    changed = 0
    for row in rows:
        slug = row["slug"]
        markdown = row["markdown"]
        try:
            print(f"translating {slug} ({len(markdown):,} chars) with {args.model}")
            translated = _call_deepseek(
                markdown,
                model=args.model,
                base_url=args.base_url.rstrip("/"),
                max_tokens=args.max_tokens,
            )
            _validate_translation(markdown, translated, max_hangul_ratio=args.max_hangul_ratio)
            _update_translation(row, translated)
            changed += 1
            print(f"updated {slug}: {len(translated):,} English chars")
        except Exception as exc:
            failures += 1
            print(f"failed {slug}: {exc}", file=sys.stderr)

    if changed:
        _clear_frontend_research_cache()
    print(f"done: updated {changed}, failures {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
