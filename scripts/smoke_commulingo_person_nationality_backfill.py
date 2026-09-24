#!/usr/bin/env python3
"""Regression checks for the deterministic nationality-gap backfill."""

import importlib.util
from pathlib import Path

PATH = Path(__file__).with_name("commulingo_backfill_person_nationality.py")
spec = importlib.util.spec_from_file_location("nationality_backfill", PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)

# Since 83733c2 national origin is never inferred from citizenship: it must
# already be set or come from a documented ORIGIN_OVERRIDES entry.
rows = [
    {"id": "existing-foreign", "name": "A", "citizenship": "france", "origin": "france"},
    {"id": "victor-serge", "name": "B", "citizenship": "soviet", "origin": ""},
    {"id": "pak-hon-yong", "name": "E", "citizenship": "north-korea", "origin": ""},
    {"id": "bogdan-knunyants", "name": "C", "citizenship": "", "origin": ""},
]
changes = mod.plan(rows)
by_id = {row["id"]: row for row in changes}
assert by_id["existing-foreign"]["new_origin"] == "france"
assert by_id["existing-foreign"]["origin_reason"] == "existing"
assert by_id["victor-serge"]["origin_reason"] == "documented national/family identity"
assert by_id["victor-serge"]["new_origin"] == "russia"
assert by_id["pak-hon-yong"]["new_origin"] == "korea"
assert by_id["bogdan-knunyants"]["new_citizenship"] == "russia"
assert by_id["bogdan-knunyants"]["new_origin"] == "armenia"

assert not hasattr(mod, 'build_sql'), 'report must not generate direct SQL writes'

try:
    mod.plan([{"id": "new-unknown", "name": "D", "citizenship": "", "origin": ""}])
except RuntimeError as exc:
    assert "no reviewed override" in str(exc)
else:
    raise AssertionError("unknown missing citizenship must fail closed")

for citizenship in ("france", "soviet"):
    try:
        mod.plan([{"id": "no-origin-override", "name": "F", "citizenship": citizenship, "origin": ""}])
    except RuntimeError as exc:
        assert "non-citizenship-only override" in str(exc)
    else:
        raise AssertionError(f"origin must not be inferred from citizenship {citizenship!r}")

print("commulingo person nationality backfill smoke: ok")
