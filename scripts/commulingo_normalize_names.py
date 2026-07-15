#!/usr/bin/env python3
"""Bulk-normalize person-name spelling variants in CommuLingo text columns.

Reads variant -> canonical pairs from config/commulingo_name_normalization.json
and rewrites every Korean/English text column that uses a variant OUTSIDE
direct quotation marks (quoted spans keep their original spelling). Writes via
the sanctioned scripts/psql-supabase helper, like the nationality backfill.

Excluded on purpose:
  - commulingo_person_aliases: aliases intentionally hold variants (linkify).
  - commulingo_people.name_ko / name_en and patronymics: the canonical
    registry itself — a variant there is reported, never auto-rewritten.

Usage:
  scripts/commulingo_normalize_names.py --dry-run   # full diff report, no writes
  scripts/commulingo_normalize_names.py             # apply, printing every change
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PSQL = ROOT / "scripts" / "psql-supabase"
CONFIG = ROOT / "config" / "commulingo_name_normalization.json"

QUOTED_SPAN_RE = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|《[^》]*》')

# (table, column, lang) targets are discovered from information_schema; these
# are the exceptions that must never be auto-rewritten.
SKIP_TABLES = {"commulingo_person_aliases", "commulingo_person_patronymics"}
REPORT_ONLY_COLUMNS = {("commulingo_people", "name_ko"), ("commulingo_people", "name_en")}


def run_psql(args: list[str], stdin: str | None = None) -> str:
    result = subprocess.run([str(PSQL), *args], input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"psql-supabase failed: {result.stderr.strip()}")
    return result.stdout


def query_json(sql: str) -> list:
    out = run_psql(["-t", "-A"], stdin=f"SELECT COALESCE(json_agg(t), '[]'::json) FROM ({sql}) t;")
    return json.loads(out.strip() or "[]")


def load_map() -> dict:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked = data.get("blocked") or {}
    return {
        "ko": dict(data.get("ko") or {}),
        "en": dict(data.get("en") or {}),
        "blocked": {lang: list(blocked.get(lang) or []) for lang in ("ko", "en")},
    }


def protected_span_re(blocked: list[str]) -> re.Pattern:
    """Quoted spans plus blocked container-strings (시베리아 ⊃ 베리아), longest first."""
    parts = [QUOTED_SPAN_RE.pattern]
    parts.extend(re.escape(s) for s in sorted(blocked, key=len, reverse=True))
    return re.compile("|".join(parts))


def normalize_unprotected(text: str, table: dict[str, str], protect: re.Pattern) -> str:
    """Apply variant->canonical replacements outside quotes and blocked strings."""
    parts: list[str] = []
    last = 0
    for m in protect.finditer(text):
        chunk = text[last:m.start()]
        for variant, canonical in table.items():
            chunk = chunk.replace(variant, canonical)
        parts.append(chunk)
        parts.append(m.group(0))  # protected span untouched
        last = m.end()
    tail = text[last:]
    for variant, canonical in table.items():
        tail = tail.replace(variant, canonical)
    parts.append(tail)
    return "".join(parts)


def discover_targets() -> list[dict]:
    """Text columns ending _ko/_en in commulingo tables, with their PK columns."""
    cols = query_json(
        """SELECT c.table_name, c.column_name,
                  RIGHT(c.column_name, 2) AS lang
             FROM information_schema.columns c
            WHERE c.table_schema = 'public'
              AND c.table_name LIKE 'commulingo%'
              AND (c.column_name LIKE '%\\_ko' OR c.column_name LIKE '%\\_en')
              AND c.data_type IN ('text', 'character varying')
            ORDER BY c.table_name, c.column_name"""
    )
    pks = query_json(
        """SELECT tc.table_name, kcu.column_name, kcu.ordinal_position
             FROM information_schema.table_constraints tc
             JOIN information_schema.key_column_usage kcu
               ON kcu.constraint_name = tc.constraint_name
              AND kcu.table_schema = tc.table_schema
            WHERE tc.table_schema = 'public'
              AND tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_name LIKE 'commulingo%'
            ORDER BY tc.table_name, kcu.ordinal_position"""
    )
    pk_by_table: dict[str, list[str]] = {}
    for row in pks:
        pk_by_table.setdefault(row["table_name"], []).append(row["column_name"])
    targets = []
    for row in cols:
        table = row["table_name"]
        if table in SKIP_TABLES or table not in pk_by_table:
            continue
        targets.append({
            "table": table,
            "column": row["column_name"],
            "lang": row["lang"],
            "pk": pk_by_table[table],
        })
    return targets


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def context(old: str, masked: str, variant: str, width: int = 28) -> str:
    i = masked.find(variant)
    if i < 0:
        return old[:width * 2]
    return old[max(0, i - width):i + len(variant) + width].replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    norm = load_map()
    if not any(norm.values()):
        print("normalization map is empty; nothing to do")
        return 0

    targets = discover_targets()
    updates: list[str] = []
    changed = 0

    for t in targets:
        table_map = norm.get(t["lang"]) or {}
        if not table_map:
            continue
        protect = protected_span_re(norm["blocked"].get(t["lang"]) or [])
        like = " OR ".join(
            f"{t['column']} LIKE '%' || {sql_literal(v)} || '%'" for v in table_map
        )
        pk_cols = ", ".join(t["pk"])
        rows = query_json(
            f"SELECT {pk_cols}, {t['column']} AS txt FROM {t['table']} WHERE {like}"
        )
        for row in rows:
            old = row["txt"] or ""
            new = normalize_unprotected(old, table_map, protect)
            key = "/".join(str(row[c]) for c in t["pk"])
            if new == old:
                continue  # protected-only match (quotes / blocked containers)
            masked = protect.sub(lambda m: " " * len(m.group(0)), old)
            hit_variants = [v for v in table_map if v in masked]
            if (t["table"], t["column"]) in REPORT_ONLY_COLUMNS:
                print(f"[REPORT-ONLY] {t['table']}.{t['column']} pk={key}: canonical "
                      f"registry uses variant(s) {hit_variants} — fix by hand")
                continue
            changed += 1
            for v in hit_variants:
                print(f"  {t['table']}.{t['column']} pk={key}: …{context(old, masked, v)}…")
            where = " AND ".join(
                f"{c} = {sql_literal(str(row[c]))}" for c in t["pk"]
            )
            updates.append(
                f"UPDATE {t['table']} SET {t['column']} = {sql_literal(new)} WHERE {where};"
            )

    print(f"\n[normalize] {changed} row-column updates across "
          f"{len(targets)} scanned columns")
    if args.dry_run:
        print("[normalize] DRY RUN — nothing written")
        return 0
    if not updates:
        print("[normalize] nothing to write")
        return 0
    script = "BEGIN;\n" + "\n".join(updates) + "\nCOMMIT;\n"
    run_psql(["-v", "ON_ERROR_STOP=1"], stdin=script)
    print(f"[normalize] applied {len(updates)} updates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
