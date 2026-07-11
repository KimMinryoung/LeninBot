#!/usr/bin/env python3
"""One-off cleanup: demote body H1 headings to H2 in research_documents.

The fixed report layout (2026-07-11) allows exactly one H1 — the document
title. Older documents used `# ...` section headings in the body, which the
site renders as extra <h1> elements. This demotes every H1 after the first
(the title) to `## ...`, in both markdown and markdown_en, skipping fenced
code blocks.

Same CSV round-trip flow as dedupe_research_headers.py.
"""
import csv
import hashlib
import json
import re
import sys

H1_RE = re.compile(r"^(\s*)#(\s+\S)")


def demote_body_h1(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").split("\n")
    out = []
    seen_title = False
    in_code = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if not in_code and H1_RE.match(line):
            if not seen_title:
                seen_title = True
                out.append(line)
            else:
                out.append(H1_RE.sub(r"\1##\2", line))
            continue
        out.append(line)
    return "\n".join(out)


def main() -> None:
    src, dst, backup = sys.argv[1], sys.argv[2], sys.argv[3]
    csv.field_size_limit(10_000_000)
    changed, backups = [], []
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    for doc_id, slug, status, md, md_en in rows:
        new_md = demote_body_h1(md)
        new_en = demote_body_h1(md_en) if md_en else md_en
        if new_md == md and new_en == md_en:
            continue
        backups.append({"id": doc_id, "slug": slug, "markdown": md, "markdown_en": md_en})
        sha = hashlib.sha256(new_md.encode("utf-8")).hexdigest()
        changed.append((doc_id, new_md, new_en, sha))
        n_ko = sum(1 for a, b in zip(md.split("\n"), new_md.split("\n")) if a != b)
        n_en = sum(1 for a, b in zip(md_en.split("\n"), new_en.split("\n")) if a != b) if md_en else 0
        print(f"id={doc_id} status={status} slug={slug}: demoted ko={n_ko} en={n_en} heading(s)")
    with open(backup, "w", encoding="utf-8") as fh:
        json.dump(backups, fh, ensure_ascii=False)
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(changed)
    print(f"\n{len(changed)} of {len(rows)} documents changed; fixes -> {dst}, backup -> {backup}")


if __name__ == "__main__":
    main()
