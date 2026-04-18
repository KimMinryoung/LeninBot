#!/usr/bin/env python3
"""Daily KG backup to Cloudflare R2.

Dumps Neo4j entities/edges/mentions via the existing backup_kg.py logic,
bundles them into kg-backup-YYYY-MM-DD.tar.gz, uploads to the
cyber-lenin-backups R2 bucket, and deletes the backup from 2 days ago
(rolling 2-day retention: keep today + yesterday).

Scheduled by leninbot-kg-backup.timer (daily 03:00 KST).
"""
import os
import sys
import tarfile
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "skills" / "kg-maintenance" / "scripts"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import requests
from backup_kg import backup as _dump_kg

BUCKET = "cyber-lenin-backups"
KST = timezone(timedelta(hours=9))


def _r2_url(key: str) -> str:
    acct = os.environ["R2_CF_ACCOUNT_ID"]
    return f"https://api.cloudflare.com/client/v4/accounts/{acct}/r2/buckets/{BUCKET}/objects/{key}"


def _r2_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['R2_CF_API_TOKEN']}"}


def _r2_put(key: str, path: str) -> None:
    with open(path, "rb") as f:
        data = f.read()
    resp = requests.put(
        _r2_url(key),
        headers={**_r2_headers(), "Content-Type": "application/gzip"},
        data=data,
        timeout=300,
    )
    resp.raise_for_status()


def _r2_delete(key: str) -> bool:
    resp = requests.delete(_r2_url(key), headers=_r2_headers(), timeout=60)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True


def main() -> int:
    ts = _dump_kg(include_embeddings=True)

    backup_dir = ROOT / "data" / "kg_backups"
    dump_files = [
        backup_dir / f"entities_{ts}.json",
        backup_dir / f"edges_{ts}.json",
        backup_dir / f"mentions_{ts}.json",
    ]

    today = datetime.now(KST)
    archive_key = f"kg-backup-{today.strftime('%Y-%m-%d')}.tar.gz"

    tmp_path = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False).name
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for f in dump_files:
                tar.add(f, arcname=f.name)
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        print(f"Archive built: {archive_key} ({size_mb:.1f} MB)")

        _r2_put(archive_key, tmp_path)
        print(f"Uploaded to R2: {BUCKET}/{archive_key}")
    finally:
        os.unlink(tmp_path)

    two_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    old_key = f"kg-backup-{two_days_ago}.tar.gz"
    if _r2_delete(old_key):
        print(f"Deleted old backup: {old_key}")
    else:
        print(f"No old backup to delete: {old_key}")

    for f in dump_files:
        f.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
