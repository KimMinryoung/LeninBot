#!/usr/bin/env python3
"""Import legacy research markdown files into research_documents.

Primary source precedence:
  1. research/*.md
  2. output/research/*.md when the filename is not already present

Optional English translations are read from research/en/{filename}.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research_store

RESEARCH_DIR = ROOT / "research"
LEGACY_RESEARCH_DIR = ROOT / "output" / "research"
TRANSLATIONS_DIR = RESEARCH_DIR / "en"

_DATE_RE = re.compile(r"\*\*작성일:\*\*\s*(\d{4}-\d{2}-\d{2})")


def _published_at(markdown: str, path: Path) -> datetime:
    for line in markdown.splitlines()[:20]:
        m = _DATE_RE.match(line.strip())
        if m:
            try:
                return datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _iter_sources() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for directory in (LEGACY_RESEARCH_DIR, RESEARCH_DIR):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.md"):
            files[path.name] = path
    return files


def main() -> int:
    imported = 0
    updated = 0
    skipped = 0

    for filename, path in sorted(_iter_sources().items()):
        try:
            markdown = research_store.load_markdown_file(path)
            title = research_store.extract_title(markdown, research_store.public_slug(filename).replace("_", " "))
            translation_path = TRANSLATIONS_DIR / filename
            markdown_en = None
            title_en = None
            summary_en = None
            if translation_path.is_file():
                markdown_en = research_store.load_markdown_file(translation_path)
                title_en = research_store.extract_title(markdown_en, title)
                summary_en = research_store.extract_excerpt(markdown_en)

            _row, existed = research_store.upsert_document(
                filename=filename,
                title=title,
                markdown=markdown,
                summary=research_store.extract_excerpt(markdown),
                markdown_en=markdown_en,
                title_en=title_en,
                summary_en=summary_en,
                status="public",
                published_at=_published_at(markdown, path),
            )
            if existed:
                updated += 1
            else:
                imported += 1
        except Exception as e:
            skipped += 1
            print(f"skip {filename}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"research import complete: imported={imported} updated={updated} skipped={skipped}")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
