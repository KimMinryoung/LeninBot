#!/usr/bin/env python3
"""ingest_literature.py — Ingest files from literature/corpus/ into lenin_corpus.

Drop .md / .txt / .pdf / .docx / .html files into literature/corpus/, optionally
group them into subdirectories, and run this script. Each successfully ingested
file gets a sibling `.ingested` marker so repeated runs skip it. Pass --reingest
to force re-processing (drops existing chunks by source title and re-inserts).

Metadata conventions:
  layer  = "modern_analysis"
  source = file stem (e.g. "piketty_capital_2024")
  author = inferred from YAML front-matter `author:` if the file starts with one,
           otherwise left empty — pass --author on the CLI to override for a run
  year   = from --year or from a 4-digit prefix in the filename

Usage:
    python scripts/ingest_literature.py                      # ingest new files
    python scripts/ingest_literature.py --dir literature/corpus/political
    python scripts/ingest_literature.py --file path/to/doc.pdf --author "Piketty"
    python scripts/ingest_literature.py --reingest path/to/doc.md
    python scripts/ingest_literature.py --list                # show what would run
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from shared import convert_document, delete_corpus_source, ingest_to_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_ROOT = ROOT / "literature" / "corpus"
SUPPORTED_EXT = {".md", ".txt", ".pdf", ".docx", ".html"}
_YEAR_RE = re.compile(r"(?:^|[_-])(19|20)(\d{2})(?:[_-]|$)")
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() in (".md", ".txt", ".html"):
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")
    # PDF/DOCX via markitdown
    return convert_document(str(path), max_chars=0)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    body_start = m.end()
    meta_raw = m.group(1)
    meta: dict = {}
    for line in meta_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip('"\'')
    return meta, text[body_start:]


def _year_from_name(name: str) -> int | None:
    m = _YEAR_RE.search(name)
    if m:
        return int(m.group(1) + m.group(2))
    return None


def _source_from_name(stem: str) -> str:
    # strip a leading date prefix like "20260401_" so the source reads as the title
    stripped = re.sub(r"^\d{4,8}[_-]", "", stem)
    return stripped.replace("_", " ").strip() or stem


def _iter_candidates(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    return sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def _process(path: Path, cli_author: str | None, cli_year: int | None,
             reingest: bool, dry_run: bool) -> tuple[str, int]:
    marker = path.with_suffix(path.suffix + ".ingested")
    if marker.exists() and not reingest:
        return ("skip", 0)

    text = _read_text(path)
    if not text or len(text.strip()) < 200:
        return ("empty", 0)

    fm, body = _parse_front_matter(text)
    author = cli_author or fm.get("author")
    year = cli_year or (int(fm["year"]) if fm.get("year", "").isdigit() else None) or _year_from_name(path.stem)
    source = fm.get("title") or _source_from_name(path.stem)

    if dry_run:
        logger.info("[dry] %s → source=%r author=%r year=%s len=%d",
                    path.name, source, author, year, len(body))
        return ("dry", 0)

    if reingest:
        deleted = delete_corpus_source(source, layer="modern_analysis")
        if deleted:
            logger.info("re-ingest: deleted %d existing chunks for %r", deleted, source)

    try:
        fp_meta = str(path.relative_to(ROOT))
    except ValueError:
        fp_meta = str(path)
    n = ingest_to_corpus(
        body, source=source, layer="modern_analysis",
        author=author, year=year,
        extra_metadata={"filepath": fp_meta},
    )
    marker.write_text(f"chunks={n}\nsource={source}\n", encoding="utf-8")
    return ("ok", n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--dir", type=Path, default=CORPUS_ROOT,
                   help=f"directory to scan (default: {CORPUS_ROOT.relative_to(ROOT)})")
    p.add_argument("--file", type=Path, help="single file to ingest (overrides --dir)")
    p.add_argument("--author", help="override author for every file in this run")
    p.add_argument("--year", type=int, help="override year for every file in this run")
    p.add_argument("--reingest", action="store_true",
                   help="drop existing chunks for each source before re-inserting")
    p.add_argument("--list", action="store_true", dest="dry_run",
                   help="show what would be processed without inserting")
    args = p.parse_args(argv)

    target = args.file if args.file else args.dir
    if not target.exists():
        print(f"ERROR: target does not exist: {target}", file=sys.stderr)
        if target == CORPUS_ROOT:
            print(f"HINT: create {CORPUS_ROOT.relative_to(ROOT)}/ and drop files in", file=sys.stderr)
        return 2

    candidates = _iter_candidates(target)
    if not candidates:
        print("No ingestible files found.", file=sys.stderr)
        return 0

    totals = {"ok": 0, "skip": 0, "empty": 0, "dry": 0, "chunks": 0}
    for path in candidates:
        try:
            status, n = _process(path, args.author, args.year, args.reingest, args.dry_run)
        except Exception as e:
            logger.error("%s → FAILED: %s", path.name, e)
            continue
        totals[status] = totals.get(status, 0) + 1
        totals["chunks"] += n
        if status == "ok":
            logger.info("%s → %d chunks", path.name, n)
        elif status == "skip":
            logger.debug("%s → already ingested", path.name)
        elif status == "empty":
            logger.warning("%s → empty or unreadable", path.name)

    print(
        f"\nDone: {totals['ok']} ingested, {totals['skip']} skipped, "
        f"{totals['empty']} empty/unreadable, {totals['dry']} dry-run "
        f"({totals['chunks']} total chunks)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
