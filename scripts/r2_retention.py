#!/usr/bin/env python3
"""Shared R2 retention sweep for the daily backup jobs.

Each backup job used to expire exactly one dated key per run (`today - N`), so
any run that failed or was skipped left its object on R2 permanently — nothing
ever revisited that date. The leak was real: under a 2-day window the KG bucket
still held kg-backup objects from 2026-04-22, 2026-04-23 and 2026-06-30.

This sweeps the whole prefix against a cutoff instead, so a missed run
self-heals on the next successful one.

Safety properties:
  - Only keys matching `<prefix>-YYYY-MM-DD<suffix>` exactly are considered.
    Anything undated, differently named, or under another prefix is invisible
    here and can never be deleted (e.g. `frontend-archives/...`).
  - A listing failure deletes nothing; it warns and returns.
  - If a sweep would leave fewer than `min_keep` objects it deletes nothing.
    A bad cutoff can then only make us keep too much, never wipe the bucket.
"""

import re
from datetime import date, datetime

from _r2_backup import r2_delete, r2_list_keys


def prune_r2_prefix(
    bucket: str,
    key_prefix: str,
    suffix: str,
    cutoff: date,
    *,
    dry_run: bool = False,
    min_keep: int = 2,
) -> list[str]:
    """Delete every `<key_prefix>-YYYY-MM-DD<suffix>` object dated before cutoff.

    Returns the keys deleted (or, when dry_run, the keys that would be).
    Never raises on a listing problem — the caller's backup has already
    succeeded by this point and a failed sweep must not fail the unit.
    """
    dated = re.compile(rf"^{re.escape(key_prefix)}-(\d{{4}}-\d{{2}}-\d{{2}}){re.escape(suffix)}$")

    try:
        keys = r2_list_keys(bucket, key_prefix)
    except Exception as exc:  # network, auth, malformed payload
        print(f"WARNING: R2 sweep skipped for {key_prefix}: {exc}")
        return []

    expired, kept = [], []
    for key in keys:
        match = dated.match(key)
        if not match:
            continue  # undated or foreign key — out of scope, never touched
        try:
            key_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        (expired if key_date < cutoff else kept).append(key)

    if not expired:
        return []

    if len(kept) < min_keep:
        print(
            f"WARNING: R2 sweep skipped for {key_prefix}: would leave "
            f"{len(kept)} object(s), below min_keep={min_keep}. "
            f"Cutoff {cutoff} looks wrong — deleting nothing."
        )
        return []

    deleted = []
    for key in sorted(expired):
        if dry_run:
            print(f"[dry-run] would delete expired R2 backup: {key}")
            deleted.append(key)
        else:
            r2_delete(bucket, key)
            print(f"Deleted expired R2 backup: {key}")
            deleted.append(key)
    return deleted
