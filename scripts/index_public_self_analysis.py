"""Index mature public Cyber-Lenin reports into self_produced_analysis.

This script deliberately does not run from publish_research. Public documents
often need corrections shortly after publication, so indexing is a delayed,
explicit operation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research_store
from db import query as db_query
from shared import index_public_self_analysis


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join((data or "").split())
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self.parts)


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html or "")
    return parser.get_text()


def _public_research_url(slug: str) -> str:
    return f"https://cyber-lenin.com/reports/research/{slug}"


def _public_static_page_url(slug: str) -> str:
    return f"https://cyber-lenin.com/p/{slug}"


def _safe_interval(hours: int) -> str:
    return f"{max(0, int(hours))} hours"


def select_research_documents(*, min_age_hours: int, slug: str | None, limit: int) -> list[dict]:
    research_store.ensure_research_table()
    clauses = ["status = 'public'"]
    params: list = []
    if slug:
        raw = slug.strip()
        filename = raw if raw.endswith(".md") else f"{raw}.md"
        bare_slug = raw[:-3] if raw.endswith(".md") else raw
        clauses.append("(filename = %s OR slug = %s)")
        params.extend([filename, bare_slug])
    else:
        clauses.append("updated_at <= NOW() - %s::interval")
        params.append(_safe_interval(min_age_hours))
    params.append(max(1, min(int(limit), 1000)))
    return db_query(
        f"""
        SELECT id, filename, slug, title, summary, markdown, content_sha256,
               published_at, updated_at
          FROM research_documents
         WHERE {' AND '.join(clauses)}
         ORDER BY updated_at DESC, id DESC
         LIMIT %s
        """,
        tuple(params),
    )


def select_static_pages(*, min_age_hours: int, slug: str | None, limit: int) -> list[dict]:
    clauses = []
    params: list = []
    if slug:
        clauses.append("slug = %s")
        params.append(slug.strip().lower())
    else:
        clauses.append("updated_at <= NOW() - %s::interval")
        params.append(_safe_interval(min_age_hours))
    params.append(max(1, min(int(limit), 1000)))
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return db_query(
        f"""
        SELECT id, slug, title, summary, html_body, html_body_en, updated_at
          FROM static_pages
          {where}
         ORDER BY updated_at DESC, id DESC
         LIMIT %s
        """,
        tuple(params),
    )


def index_research(
    rows: list[dict],
    *,
    dry_run: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    ok = 0
    failed = 0
    for row in rows:
        slug = row.get("slug") or str(row.get("filename") or "").removesuffix(".md")
        title = row.get("title") or slug
        print(f"research {slug}: {title}")
        if dry_run:
            continue
        result = index_public_self_analysis(
            kind="research",
            slug=slug,
            title=title,
            content=row.get("markdown") or "",
            public_url=_public_research_url(slug),
            summary=row.get("summary"),
            content_sha256=row.get("content_sha256"),
            extra_metadata={
                "updated_at": _iso(row.get("updated_at")),
            },
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if result.get("ok"):
            ok += 1
            print(f"  indexed {result.get('chunks', 0)} chunks")
        else:
            failed += 1
            print(f"  failed: {result.get('error', 'unknown error')}")
    return ok, failed


def index_static_pages(
    rows: list[dict],
    *,
    dry_run: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    ok = 0
    failed = 0
    for row in rows:
        slug = row.get("slug") or ""
        title = row.get("title") or slug
        body_text = _html_to_text(row.get("html_body") or "")
        print(f"static_page {slug}: {title}")
        if dry_run:
            continue
        result = index_public_self_analysis(
            kind="static_page",
            slug=slug,
            title=title,
            content=body_text,
            public_url=_public_static_page_url(slug),
            summary=row.get("summary"),
            extra_metadata={
                "updated_at": _iso(row.get("updated_at")),
            },
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if result.get("ok"):
            ok += 1
            print(f"  indexed {result.get('chunks', 0)} chunks")
        else:
            failed += 1
            print(f"  failed: {result.get('error', 'unknown error')}")
    return ok, failed


def _iso(value) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return str(value or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delayed/manual indexing of public Cyber-Lenin outputs into self_produced_analysis."
    )
    parser.add_argument("--kind", choices=["research", "static_pages", "all"], default="research")
    parser.add_argument("--slug", help="Index one slug immediately, ignoring min-age.")
    parser.add_argument("--min-age-hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=3200)
    parser.add_argument("--chunk-overlap", type=int, default=240)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_ok = 0
    total_failed = 0

    if args.kind in ("research", "all"):
        rows = select_research_documents(
            min_age_hours=args.min_age_hours,
            slug=args.slug,
            limit=args.limit,
        )
        ok, failed = index_research(
            rows,
            dry_run=args.dry_run,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        total_ok += ok
        total_failed += failed

    if args.kind in ("static_pages", "all"):
        rows = select_static_pages(
            min_age_hours=args.min_age_hours,
            slug=args.slug,
            limit=args.limit,
        )
        ok, failed = index_static_pages(
            rows,
            dry_run=args.dry_run,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        total_ok += ok
        total_failed += failed

    if args.dry_run:
        print("dry run complete")
        return 0
    print(f"complete: indexed={total_ok} failed={total_failed}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
