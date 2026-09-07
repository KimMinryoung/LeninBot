#!/usr/bin/env python3
"""Hermetic regression checks for CommuLingo nationality backfill semantics."""

from commulingo_backfill_nationality import decide_nationality


born_in_lithuania = {
    "cit": {"Q15180"},  # Soviet Union
    "bp": {"Q37"},     # Lithuania
}
citizenship, national_origin, unmapped = decide_nationality(born_in_lithuania)
assert citizenship == "soviet"
assert national_origin == "", "birthplace must not become nationalOrigin"
assert unmapped == []


print("commulingo nationality backfill smoke ok")
