#!/usr/bin/env python3
"""Hermetic regression checks for CommuLingo nationality backfill semantics."""

from commulingo_backfill_nationality import build_update, decide_nationality


born_in_lithuania = {
    "cit": {"Q15180"},  # Soviet Union
    "bp": {"Q37"},     # Lithuania
}
citizenship, national_origin, unmapped = decide_nationality(born_in_lithuania)
assert citizenship == "soviet"
assert national_origin == "", "birthplace must not become nationalOrigin"
assert unmapped == []

sql = build_update("example", "soviet")
assert "citizenship_code='soviet'" in sql
assert "origin_code" not in sql, "citizenship backfill must preserve curated nationalOrigin"

print("commulingo nationality backfill smoke ok")
