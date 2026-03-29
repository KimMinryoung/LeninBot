#!/usr/bin/env python3
"""Publish human-facing markdown outputs into the public research directory.

Convention going forward:
- author/edit polished public deliverables under `research/`
- or publish an existing file from `output/research/` into `research/`
- API serves `/research/{filename}` from `research/` and falls back to legacy `output/research/`
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_RESEARCH_DIR = ROOT / "research"
LEGACY_OUTPUT_RESEARCH_DIR = ROOT / "output" / "research"


def publish_markdown(source: Path, *, copy_only: bool = False) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.suffix.lower() != ".md":
        raise ValueError(f"Only .md files can be published: {source}")

    PUBLIC_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    destination = PUBLIC_RESEARCH_DIR / source.name

    if source.resolve() == destination.resolve():
        return destination

    if copy_only:
        shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a markdown file into the public research directory.")
    parser.add_argument("source", help="Path to the markdown file to publish (relative to project root or absolute).")
    parser.add_argument("--copy-only", action="store_true", help="Copy the file without any extra behavior (currently default).")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source

    published = publish_markdown(source, copy_only=args.copy_only)
    print(str(published))
    print(f"Public URL path: /research/{published.name}")


if __name__ == "__main__":
    main()
