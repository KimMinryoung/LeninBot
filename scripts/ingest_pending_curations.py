#!/usr/bin/env python3
"""ingest_pending_curations.py — Embed queued hub curations into lenin_corpus.

Hub curations are published with just the source body stashed in
`hub_curations.source_content`. This job runs after publish (daily, alongside
experience_writer) to chunk + embed those bodies into the modern_analysis
corpus layer and mark the row as ingested.

Idempotent per row: once `ingested_at` is set, the curation is skipped. Pass
--reingest to drop existing chunks (by source_title) and re-embed. Individual
failures are logged but don't abort the batch.

Usage:
    python scripts/ingest_pending_curations.py
    python scripts/ingest_pending_curations.py --limit 5
    python scripts/ingest_pending_curations.py --reingest --slug some-slug
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from db import query as db_query, execute as db_execute
from shared import delete_corpus_source, ingest_to_corpus

logger = logging.getLogger(__name__)


def _fetch_pending(limit: int | None, slug: str | None) -> list[dict]:
    base = """
        SELECT id, slug, title, source_url, source_title, source_author,
               source_published_at, source_content
          FROM hub_curations
         WHERE source_content IS NOT NULL
           AND ingested_at IS NULL
    """
    params: list = []
    if slug:
        base += " AND slug = %s"
        params.append(slug)
    base += " ORDER BY published_at ASC"
    if limit:
        base += f" LIMIT {int(limit)}"
    return db_query(base, tuple(params) or None)


def _fetch_reingest(slug: str | None) -> list[dict]:
    base = """
        SELECT id, slug, title, source_url, source_title, source_author,
               source_published_at, source_content
          FROM hub_curations
         WHERE source_content IS NOT NULL
    """
    params: list = []
    if slug:
        base += " AND slug = %s"
        params.append(slug)
    base += " ORDER BY published_at DESC"
    return db_query(base, tuple(params) or None)


def _ingest_row(row: dict, reingest: bool) -> int:
    source_title = row["source_title"] or row["title"]
    year = None
    if row.get("source_published_at"):
        try:
            year = int(str(row["source_published_at"])[:4])
        except (TypeError, ValueError):
            year = None

    if reingest:
        deleted = delete_corpus_source(source_title, layer="modern_analysis")
        if deleted:
            logger.info("  reingest: dropped %d prior chunks for %r", deleted, source_title)

    n = ingest_to_corpus(
        row["source_content"],
        source=source_title,
        layer="modern_analysis",
        author=row.get("source_author"),
        year=year,
        extra_metadata={
            "source_url": row.get("source_url"),
            "curation_slug": row.get("slug"),
            "curation_id": row.get("id"),
        },
    )
    db_execute(
        "UPDATE hub_curations SET ingested_at = now() WHERE id = %s",
        (row["id"],),
    )
    return n


def run(limit: int | None = None, slug: str | None = None,
        reingest: bool = False) -> dict:
    """Called by CLI and by experience_writer. Returns tallies."""
    rows = _fetch_reingest(slug) if reingest else _fetch_pending(limit, slug)
    if not rows:
        logger.info("curation ingest: nothing pending")
        return {"curations": 0, "chunks": 0, "failures": 0}

    logger.info("curation ingest: %d row(s) to process (reingest=%s)", len(rows), reingest)
    total_chunks = 0
    failures = 0
    for row in rows:
        try:
            n = _ingest_row(row, reingest=reingest)
            total_chunks += n
            logger.info("  ✓ %s → %d chunks", row["slug"], n)
        except Exception as e:
            failures += 1
            logger.error("  ✗ %s failed: %s", row["slug"], e)
    logger.info(
        "curation ingest: done (%d rows, %d chunks, %d failures)",
        len(rows), total_chunks, failures,
    )
    return {"curations": len(rows), "chunks": total_chunks, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--limit", type=int, help="cap number of rows processed")
    p.add_argument("--slug", help="only process this slug")
    p.add_argument("--reingest", action="store_true",
                   help="drop existing chunks and re-embed (ignores ingested_at)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(limit=args.limit, slug=args.slug, reingest=args.reingest)
    return 0 if result["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
