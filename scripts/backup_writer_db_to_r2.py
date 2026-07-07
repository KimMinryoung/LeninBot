#!/usr/bin/env python3
"""Daily writer DB backup to Cloudflare R2.

The fiction workspace tables live in the local Docker Postgres
(leninbot-writer-pg) since the 2026-07-07 migration off Supabase, so this
box is the only copy — this job is the durability story (mirrors
backup_kg_to_r2.py).

Dumps via `docker exec pg_dump -Fc` (no credentials needed: in-container
socket auth), sanity-checks the archive with `pg_restore --list`, uploads
writer-db-backup-YYYY-MM-DD.dump to the cyber-lenin-backups R2 bucket
(rolling R2_RETENTION_DAYS), and keeps a rolling LOCAL_RETENTION_DAYS copy
under data/writer_db_backups/.

Restore (into a fresh or wiped container):
  docker exec -i leninbot-writer-pg pg_restore -U writer -d writer \
      --clean --if-exists --no-owner --no-privileges < <backup.dump>

Scheduled by leninbot-writer-backup.timer (daily 03:20 KST).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _promote_systemd_credentials() -> None:
    """Expose LoadCredentialEncrypted secrets to legacy env-based helpers."""
    cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if not cred_dir:
        return
    path = Path(cred_dir) / "r2_cf_api_token"
    if path.is_file() and not os.environ.get("R2_CF_API_TOKEN"):
        os.environ["R2_CF_API_TOKEN"] = path.read_text().rstrip("\n")


_promote_systemd_credentials()

import requests
from secrets_loader import require_secret

BUCKET = "cyber-lenin-backups"
CONTAINER = "leninbot-writer-pg"
KST = timezone(timedelta(hours=9))
R2_RETENTION_DAYS = 7
LOCAL_RETENTION_DAYS = 3
_LOCAL_KEY_RE = re.compile(r"^writer-db-backup-(\d{4}-\d{2}-\d{2})\.dump$")


def _r2_url(key: str) -> str:
    acct = os.environ["R2_CF_ACCOUNT_ID"]
    return f"https://api.cloudflare.com/client/v4/accounts/{acct}/r2/buckets/{BUCKET}/objects/{key}"


def _r2_headers() -> dict:
    return {"Authorization": f"Bearer {require_secret('R2_CF_API_TOKEN')}"}


def _r2_put(key: str, path: str) -> None:
    with open(path, "rb") as f:
        data = f.read()
    resp = requests.put(
        _r2_url(key),
        headers={**_r2_headers(), "Content-Type": "application/octet-stream"},
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


def _dump(path: str) -> None:
    """pg_dump the writer DB through the container (version-matched tools)."""
    with open(path, "wb") as out:
        subprocess.run(
            ["docker", "exec", CONTAINER, "pg_dump", "-U", "writer", "-d", "writer", "-Fc"],
            stdout=out,
            check=True,
            timeout=600,
        )


def _verify(path: str) -> int:
    """pg_restore --list as an archive integrity check; returns entry count."""
    with open(path, "rb") as f:
        listed = subprocess.run(
            ["docker", "exec", "-i", CONTAINER, "pg_restore", "--list"],
            stdin=f,
            capture_output=True,
            check=True,
            timeout=120,
        )
    entries = [l for l in listed.stdout.decode().splitlines() if l and not l.startswith(";")]
    if not any("writer_manuscripts" in l for l in entries):
        raise RuntimeError("archive verification failed: writer_manuscripts missing from TOC")
    return len(entries)


def main() -> int:
    today = datetime.now(KST)
    archive_key = f"writer-db-backup-{today.strftime('%Y-%m-%d')}.dump"
    backup_dir = ROOT / "data" / "writer_db_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = tempfile.NamedTemporaryFile(suffix=".dump", delete=False).name
    try:
        _dump(tmp_path)
        entries = _verify(tmp_path)
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        print(f"Dump built and verified: {archive_key} ({size_mb:.1f} MB, {entries} TOC entries)")

        _r2_put(archive_key, tmp_path)
        print(f"Uploaded to R2: {BUCKET}/{archive_key}")

        # Local copy after successful upload (fast restore without R2 roundtrip).
        shutil.copyfile(tmp_path, backup_dir / archive_key)
        print(f"Saved local copy: {backup_dir / archive_key}")
    finally:
        os.unlink(tmp_path)

    expired = (today - timedelta(days=R2_RETENTION_DAYS + 1)).strftime("%Y-%m-%d")
    old_key = f"writer-db-backup-{expired}.dump"
    if _r2_delete(old_key):
        print(f"Deleted old R2 backup: {old_key}")

    cutoff = (today - timedelta(days=LOCAL_RETENTION_DAYS - 1)).date()
    for p in backup_dir.glob("writer-db-backup-*.dump"):
        m = _LOCAL_KEY_RE.match(p.name)
        if m and datetime.strptime(m.group(1), "%Y-%m-%d").date() < cutoff:
            p.unlink()
            print(f"Pruned local copy: {p.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
