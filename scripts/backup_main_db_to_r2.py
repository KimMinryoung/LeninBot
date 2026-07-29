#!/usr/bin/env python3
"""Daily main and legacy-game DB backups to Cloudflare R2.

The active main database and the read-only legacy game archive live in the
local Docker Postgres (`leninbot-pg`, databases `leninbot` and
`legacy_game`). This job dumps both databases with version-matched tools,
verifies each archive TOC, uploads dated objects to R2, and keeps short local
retention copies for fast recovery.

Scheduled by leninbot-main-backup.timer (daily 03:40 KST).
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
CONTAINER = "leninbot-pg"
KST = timezone(timedelta(hours=9))
R2_RETENTION_DAYS = 7
LOCAL_RETENTION_DAYS = 3
BACKUP_SPECS = (
    (
        "leninbot",
        "main-db-backup",
        ROOT / "data" / "main_db_backups",
        "lenin_corpus",
    ),
    (
        "legacy_game",
        "legacy-game-db-backup",
        ROOT / "data" / "legacy_game_db_backups",
        "story_scenes",
    ),
)


def _r2_url(key: str) -> str:
    account = os.environ["R2_CF_ACCOUNT_ID"]
    return (
        f"https://api.cloudflare.com/client/v4/accounts/{account}"
        f"/r2/buckets/{BUCKET}/objects/{key}"
    )


def _r2_headers() -> dict:
    return {"Authorization": f"Bearer {require_secret('R2_CF_API_TOKEN')}"}


def _r2_put(key: str, path: str) -> None:
    with open(path, "rb") as source:
        data = source.read()
    response = requests.put(
        _r2_url(key),
        headers={**_r2_headers(), "Content-Type": "application/octet-stream"},
        data=data,
        timeout=300,
    )
    response.raise_for_status()


def _r2_delete(key: str) -> bool:
    response = requests.delete(_r2_url(key), headers=_r2_headers(), timeout=60)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def _dump(path: str, database: str) -> None:
    """Dump one DB through the container's version-matched pg_dump."""
    with open(path, "wb") as output:
        subprocess.run(
            [
                "docker",
                "exec",
                CONTAINER,
                "pg_dump",
                "-U",
                "postgres",
                "-d",
                database,
                "-Fc",
            ],
            stdout=output,
            check=True,
            timeout=600,
        )


def _verify(path: str, marker: str) -> int:
    """Verify archive readability and a scope-specific TOC marker."""
    with open(path, "rb") as archive:
        listed = subprocess.run(
            ["docker", "exec", "-i", CONTAINER, "pg_restore", "--list"],
            stdin=archive,
            capture_output=True,
            check=True,
            timeout=120,
        )
    entries = [
        line
        for line in listed.stdout.decode().splitlines()
        if line and not line.startswith(";")
    ]
    if not any(marker in line for line in entries):
        raise RuntimeError(f"archive verification failed: {marker} missing from TOC")
    return len(entries)


def _build_upload_and_save(
    database: str,
    key_prefix: str,
    backup_dir: Path,
    marker: str,
    date: str,
) -> None:
    archive_key = f"{key_prefix}-{date}.dump"
    backup_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tempfile.NamedTemporaryFile(suffix=".dump", delete=False).name
    try:
        _dump(tmp_path, database)
        entries = _verify(tmp_path, marker)
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        print(
            f"Dump built and verified: {archive_key} "
            f"({size_mb:.1f} MB, {entries} TOC entries)"
        )
        _r2_put(archive_key, tmp_path)
        print(f"Uploaded to R2: {BUCKET}/{archive_key}")
        shutil.copyfile(tmp_path, backup_dir / archive_key)
        print(f"Saved local copy: {backup_dir / archive_key}")
    finally:
        os.unlink(tmp_path)


def _prune_local(backup_dir: Path, key_prefix: str, cutoff) -> None:
    key_re = re.compile(
        rf"^{re.escape(key_prefix)}-(\d{{4}}-\d{{2}}-\d{{2}})\.dump$"
    )
    for path in backup_dir.glob(f"{key_prefix}-*.dump"):
        match = key_re.match(path.name)
        if match and datetime.strptime(match.group(1), "%Y-%m-%d").date() < cutoff:
            path.unlink()
            print(f"Pruned local copy: {path.name}")


def main() -> int:
    today = datetime.now(KST)
    date = today.strftime("%Y-%m-%d")

    for spec in BACKUP_SPECS:
        _build_upload_and_save(*spec, date)

    expired = (today - timedelta(days=R2_RETENTION_DAYS + 1)).strftime("%Y-%m-%d")
    for _database, key_prefix, _backup_dir, _marker in BACKUP_SPECS:
        old_key = f"{key_prefix}-{expired}.dump"
        if _r2_delete(old_key):
            print(f"Deleted old R2 backup: {old_key}")

    cutoff = (today - timedelta(days=LOCAL_RETENTION_DAYS - 1)).date()
    for _database, key_prefix, backup_dir, _marker in BACKUP_SPECS:
        _prune_local(backup_dir, key_prefix, cutoff)

    return 0


if __name__ == "__main__":
    sys.exit(main())
