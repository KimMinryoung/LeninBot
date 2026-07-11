#!/usr/bin/env python3
"""One-off cleanup: remove duplicated H1/author/date header blocks in research_documents.

Before 2026-07-11, research_document stage_public/publish_public prepended the
canonical header without stripping an agent-supplied one, so most documents
carry the title/author/date block twice. This script removes the inner
duplicate in-place (Korean markdown and markdown_en independently).

Usage:
  scripts/psql-supabase -c "\\copy (SELECT id, slug, title, status, markdown,
      coalesce(markdown_en,'') FROM research_documents ORDER BY id)
      TO 'docs.csv' WITH (FORMAT csv)"
  python scripts/dedupe_research_headers.py docs.csv fixed.csv backup.json
  # review the printed report, then load fixed.csv (id, markdown,
  # markdown_en, content_sha256) into a scratch table and UPDATE-join.
"""
import csv
import hashlib
import json
import re
import sys

H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$")
META_RE = re.compile(
    r"^\s*(?:\*\*)?(?:작성자|작성일|Author|Date)(?::)?(?:\*\*)?\s*:?.*$", re.IGNORECASE
)
HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_header(markdown: str) -> str:
    """Remove a second H1 (+author/date meta, + its own HR) that duplicates the first H1."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return markdown
    m = H1_RE.match(lines[idx])
    if not m:
        return markdown
    title = _norm(m.group(1))
    idx += 1
    # Canonical header: author/date meta lines, then a horizontal rule.
    while idx < len(lines) and (not lines[idx].strip() or META_RE.match(lines[idx])):
        idx += 1
    if idx >= len(lines) or not HR_RE.match(lines[idx].strip()):
        return markdown
    idx += 1
    # Look for the duplicated H1 right after the canonical header.
    probe = idx
    while probe < len(lines) and not lines[probe].strip():
        probe += 1
    if probe >= len(lines):
        return markdown
    dup = H1_RE.match(lines[probe])
    if not dup or _norm(dup.group(1)) != title:
        return markdown
    # Remove the duplicate H1 and its author/date meta lines.
    end = probe + 1
    while end < len(lines) and (not lines[end].strip() or META_RE.match(lines[end])):
        end += 1
    # If only meta/blank lines followed and the block closes with its own HR,
    # remove that HR too (the canonical header already supplies one).
    if end < len(lines) and HR_RE.match(lines[end].strip()):
        end += 1
    while end < len(lines) and not lines[end].strip():
        end += 1
    return "\n".join(lines[:idx] + [""] + lines[end:])


def main() -> None:
    src, dst, backup = sys.argv[1], sys.argv[2], sys.argv[3]
    csv.field_size_limit(10_000_000)
    changed, backups = [], []
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    for doc_id, slug, title, status, md, md_en in rows:
        new_md = dedupe_header(md)
        new_en = dedupe_header(md_en) if md_en else md_en
        if new_md == md and new_en == md_en:
            continue
        backups.append({"id": doc_id, "slug": slug, "markdown": md, "markdown_en": md_en})
        sha = hashlib.sha256(new_md.encode("utf-8")).hexdigest()
        changed.append((doc_id, new_md, new_en, sha))
        print(
            f"id={doc_id} status={status} slug={slug}\n"
            f"  ko: {len(md)} -> {len(new_md)} chars"
            f"{' (unchanged)' if new_md == md else ''}\n"
            f"  en: {len(md_en)} -> {len(new_en)} chars"
            f"{' (unchanged)' if new_en == md_en else ''}"
        )
    with open(backup, "w", encoding="utf-8") as fh:
        json.dump(backups, fh, ensure_ascii=False)
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(changed)
    print(f"\n{len(changed)} of {len(rows)} documents need dedupe; fixes -> {dst}, backup -> {backup}")


if __name__ == "__main__":
    main()
