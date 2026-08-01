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

import os
import re
from datetime import date, datetime

import requests

from secrets_loader import require_secret

_API = "https://api.cloudflare.com/client/v4"
_LIST_PAGE = 1000


def _headers() -> dict:
    return {"Authorization": f"Bearer {require_secret('R2_CF_API_TOKEN')}"}


def _bucket_url(bucket: str) -> str:
    account = os.environ["R2_CF_ACCOUNT_ID"]
    return f"{_API}/accounts/{account}/r2/buckets/{bucket}"


def _list_keys(bucket: str, key_prefix: str) -> list[str]:
    """Return every object key under key_prefix, following pagination."""
    keys: list[str] = []
    cursor = None
    while True:
        params = {"prefix": key_prefix, "per_page": _LIST_PAGE}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(
            f"{_bucket_url(bucket)}/objects",
            headers=_headers(),
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"R2 list failed: {payload.get('errors')}")
        keys.extend(item["key"] for item in payload.get("result") or [])
        cursor = (payload.get("result_info") or {}).get("cursor")
        if not cursor:
            return keys


def _delete(bucket: str, key: str) -> bool:
    response = requests.delete(
        f"{_bucket_url(bucket)}/objects/{key}", headers=_headers(), timeout=60
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


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
        keys = _list_keys(bucket, key_prefix)
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
        elif _delete(bucket, key):
            print(f"Deleted expired R2 backup: {key}")
            deleted.append(key)
    return deleted
