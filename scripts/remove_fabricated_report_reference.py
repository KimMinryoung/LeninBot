#!/usr/bin/env python3
"""Remove a citation to a report that was never published.

kospi-bear-market-leverage-etf-iran-july-2026 cites
'삼중 압착: 제조업 고용·건설·청년 동반 위기 (2026-06-17)' at
/reports/research/korea-triple-crush-employment-construction-youth-may-2026.
No report carries that title and none was published on that date, so the
citation is invented rather than mistyped and there is nothing to repoint it to
(scripts/fix_broken_report_links.py handles the ones that are repointable).

It appears three times per language field and all three have to go together:
the entry in the '관련 보고서' line, the [^triple_crush] footnote definition, and
the inline [^triple_crush] marker in the prose. Removing only the definition
would leave the marker rendering as literal text.

The sentence carrying the marker is left as written. Its claim about
manufacturing employment stays; only the fabricated source backing it goes.

Usage:
  python scripts/remove_fabricated_report_reference.py            # report only
  python scripts/remove_fabricated_report_reference.py --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import execute as db_execute, query as db_query
from scripts.fix_broken_report_links import clear_frontend_research_cache

SLUG = "korea-triple-crush-employment-construction-youth-may-2026"
FOOTNOTE = "triple_crush"
FIELDS = ("markdown", "markdown_en")

# A list entry: optional leading separator, the markdown link, and the trailing
# '(YYYY-MM-DD)' the related-report lines carry.
ENTRY_RE = re.compile(
    r"(?:\s*[,·]\s*)?\[[^\]]*\]\(/reports/research/" + re.escape(SLUG) + r"\)"
    r"(?:\s*\(\d{4}-\d{2}-\d{2}\))?"
)
DEFINITION_RE = re.compile(r"(?m)^\[\^" + re.escape(FOOTNOTE) + r"\]:.*\n?")
MARKER_RE = re.compile(r"\[\^" + re.escape(FOOTNOTE) + r"\]")


def clean(text: str) -> tuple[str, dict[str, int]]:
    counts = {
        "list entry": len(ENTRY_RE.findall(text)),
        "footnote definition": len(DEFINITION_RE.findall(text)),
    }
    out = ENTRY_RE.sub("", text)
    out = DEFINITION_RE.sub("", out)
    # Markers last, so the definition line is not counted as one.
    counts["inline marker"] = len(MARKER_RE.findall(out))
    out = MARKER_RE.sub("", out)
    # A removed first entry can leave the label followed by a separator.
    out = re.sub(r"(?m)^(\*\*[^*]+:\*\*)\s*[,·]\s*", r"\1 ", out)
    # And a removed only-entry leaves a bare label line.
    out = re.sub(r"(?m)^\*\*(?:관련 보고서|Related Reports?):\*\*\s*$\n?", "", out)
    return out, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = db_query(
        "SELECT id, slug, markdown, markdown_en FROM research_documents "
        "WHERE markdown LIKE %s OR markdown_en LIKE %s",
        (f"%{SLUG}%", f"%{SLUG}%"),
    )
    if not rows:
        print(f"no document references {SLUG}")
        return 0

    changed = 0
    for row in rows:
        updates: dict[str, str] = {}
        for field in FIELDS:
            text = row.get(field) or ""
            if SLUG not in text and f"[^{FOOTNOTE}]" not in text:
                continue
            fixed, counts = clean(text)
            if fixed == text:
                continue
            updates[field] = fixed
            summary = ", ".join(f"{n}x {what}" for what, n in counts.items() if n)
            print(f"{row['slug']} [{field}]: removed {summary}")
            for line in fixed.splitlines():
                if "관련 보고서" in line or "Related Report" in line:
                    print(f"    line now: {line}")
        if updates:
            changed += 1
            if args.apply:
                sets = ", ".join(f"{field} = %s" for field in updates)
                db_execute(
                    f"UPDATE research_documents SET {sets}, updated_at = NOW() WHERE id = %s",
                    (*updates.values(), row["id"]),
                )

    if not args.apply:
        print(f"\n{changed} document(s) would change; re-run with --apply")
        return 0
    print(f"\n{changed} document(s) updated")
    if changed:
        clear_frontend_research_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
