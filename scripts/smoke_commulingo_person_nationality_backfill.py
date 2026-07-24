#!/usr/bin/env python3
"""Regression checks for the deterministic nationality-gap backfill."""

import importlib.util
from pathlib import Path

PATH = Path(__file__).with_name("commulingo_backfill_person_nationality.py")
spec = importlib.util.spec_from_file_location("nationality_backfill", PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)

rows = [
    {"id": "existing-foreign", "name": "A", "citizenship": "france", "origin": ""},
    {"id": "victor-serge", "name": "B", "citizenship": "soviet", "origin": ""},
    {"id": "bogdan-knunyants", "name": "C", "citizenship": "", "origin": ""},
]
changes = mod.plan(rows)
by_id = {row["id"]: row for row in changes}
assert by_id["existing-foreign"]["new_origin"] == "france"
assert by_id["victor-serge"]["new_origin"] == "russia"
assert by_id["bogdan-knunyants"]["new_citizenship"] == "russia"
assert by_id["bogdan-knunyants"]["new_origin"] == "armenia"

sql = mod.build_sql(changes)
assert "COALESCE(citizenship_code,'')=''" in sql
assert "COALESCE(origin_code,'')=''" in sql
assert "origin_code='armenia'" in sql
assert "BEGIN;" in sql and "COMMIT;" in sql

try:
    mod.plan([{"id": "new-unknown", "name": "D", "citizenship": "", "origin": ""}])
except RuntimeError as exc:
    assert "no reviewed override" in str(exc)
else:
    raise AssertionError("unknown missing citizenship must fail closed")

print("commulingo person nationality backfill smoke: ok")
