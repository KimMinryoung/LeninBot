"""Embed queued hub curations into lenin_corpus.

Hub curations are published with just the source body stashed in
`hub_curations.source_content`. This job runs after publish (daily, alongside
experience_writer) to chunk + embed those bodies into the modern_analysis
corpus layer and mark the row as ingested.

Idempotent per row: once `ingested_at` is set, the curation is skipped. With
reingest=True, existing chunks (by source_title) are dropped and re-embedded.
Individual failures are logged but don't abort the batch.

CLI: scripts/ingest_pending_curations.py
"""
from __future__ import annotations

import logging

from db import query as db_query, execute as db_execute
from corpus.store import delete_corpus_source, ingest_to_corpus

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
        # --reingest is an explicit "redo" signal: bypass the URL-dedup guard so
        # the caller's delete_corpus_source above actually lands fresh chunks.
        skip_if_source_url_exists=not reingest,
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
