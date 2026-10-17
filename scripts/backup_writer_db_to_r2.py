#!/usr/bin/env python3
"""Daily writer DB backup to Cloudflare R2.

The fiction workspace tables live in the local Docker Postgres
(leninbot-pg, `writer` database — consolidated from leninbot-writer-pg on
2026-07-28, originally migrated off Supabase 2026-07-07), so this box is
the only copy — this job is the durability story (mirrors
backup_kg_to_r2.py).

Dumps via `docker exec pg_dump -Fc` (no credentials needed: in-container
socket auth), sanity-checks the archive with `pg_restore --list`, uploads
writer-db-backup-YYYY-MM-DD.dump to the cyber-lenin-backups R2 bucket
(rolling R2_RETENTION_DAYS), and keeps a rolling LOCAL_RETENTION_DAYS copy
under data/writer_db_backups/.

The writer DB often goes days without a write, and an unchanged dump is the
same 79 MB every day. When today's dump matches the previous manifest's
content digest and that dump is still on R2, only today's manifest is
uploaded, pointing at the earlier dump; restore_db.py follows the pointer.

Restore (into a fresh or wiped container):
  docker exec -i leninbot-pg pg_restore -U writer -d writer \
      --clean --if-exists --no-owner --no-privileges < <backup.dump>

Scheduled by leninbot-writer-backup.timer (daily 03:20 KST).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _r2_backup import (
    MANIFEST_SUFFIX,
    disk_warning,
    dump_digest,
    promote_systemd_credentials,
    prune_local_backups,
    publish_manifest,
    r2_put,
    try_count_rows,
    unchanged_archive,
)

promote_systemd_credentials()

from r2_retention import prune_r2_prefix

BUCKET = "cyber-lenin-backups"
# The standby runs these same jobs against its hot-standby replica.
CONTAINER = os.environ.get("BACKUP_PG_CONTAINER", "leninbot-pg")
KST = timezone(timedelta(hours=9))
R2_RETENTION_DAYS = 15
LOCAL_RETENTION_DAYS = 3


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
    disk = disk_warning(ROOT)

    tmp_path = tempfile.NamedTemporaryFile(suffix=".dump", delete=False).name
    try:
        rows_before = try_count_rows(CONTAINER, "writer")
        _dump(tmp_path)
        entries = _verify(tmp_path)
        rows_after = try_count_rows(CONTAINER, "writer")
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        print(f"Dump built and verified: {archive_key} ({size_mb:.1f} MB, {entries} TOC entries)")

        # Local copy first, so a failed upload does not also cost the local one.
        shutil.copyfile(tmp_path, backup_dir / archive_key)
        print(f"Saved local copy: {backup_dir / archive_key}")

        digest = dump_digest(tmp_path)
        data_archive = unchanged_archive(BUCKET, backup_dir, "writer-db-backup", archive_key, digest)
        # Skip only when today's manifest can carry the pointer.
        if data_archive and rows_before is not None and rows_after is not None:
            print(f"Unchanged since {data_archive}; R2 dump upload skipped")
        else:
            data_archive = archive_key
            r2_put(BUCKET, archive_key, tmp_path)
            print(f"Uploaded to R2: {BUCKET}/{archive_key}")
    finally:
        os.unlink(tmp_path)

    problems = []
    shrunk = []
    if rows_before is None or rows_after is None:
        problems.append("dump uploaded, but row counting failed; no manifest")
    else:
        shrunk = publish_manifest(
            BUCKET, backup_dir, "writer-db-backup", archive_key, "writer",
            rows_before, rows_after,
            data_archive=data_archive, content_sha256=digest,
        )

    # After a shrink, keep every older dump: the oldest may be the last copy
    # from before the loss.
    if not shrunk:
        r2_cutoff = (today - timedelta(days=R2_RETENTION_DAYS)).date()
        for suffix in (".dump", MANIFEST_SUFFIX):
            prune_r2_prefix(
                BUCKET, "writer-db-backup", suffix, r2_cutoff,
                protect=frozenset({data_archive}),
            )

    cutoff = (today - timedelta(days=LOCAL_RETENTION_DAYS - 1)).date()
    for suffix in (".dump", MANIFEST_SUFFIX):
        prune_local_backups(backup_dir, "writer-db-backup", suffix, cutoff)

    if shrunk:
        problems.append("tables shrank since the previous backup: " + "; ".join(shrunk))
    if disk:
        problems.append(disk)
    if problems:
        # Fails the unit after the upload, so the watchdog ping is withheld.
        raise SystemExit("WARNING: " + " | ".join(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
