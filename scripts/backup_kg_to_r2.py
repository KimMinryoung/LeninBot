#!/usr/bin/env python3
"""Daily KG backup to Cloudflare R2.

Dumps Neo4j entities/edges/mentions via the existing backup_kg.py logic,
bundles them into kg-backup-YYYY-MM-DD.tar.gz, uploads to the
cyber-lenin-backups R2 bucket, and deletes the backup that fell out of the
rolling R2_RETENTION_DAYS window.

Also keeps a rolling 3-day local copy under data/kg_backups/ for fast
restore without R2 roundtrip. Raw JSON dumps are deleted after upload
(only the tar.gz is retained locally).

Scheduled by leninbot-kg-backup.timer (daily 03:00 KST).
"""
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "skills" / "kg-maintenance" / "scripts"))


from _r2_backup import promote_systemd_credentials, prune_local_backups, r2_put

promote_systemd_credentials((
    ("neo4j_password", "NEO4J_PASSWORD"),
    ("r2_cf_api_token", "R2_CF_API_TOKEN"),
))

from backup_kg import backup as _dump_kg
from r2_retention import prune_r2_prefix

BUCKET = "cyber-lenin-backups"
KST = timezone(timedelta(hours=9))
R2_RETENTION_DAYS = 14  # keep today + the 14 previous days on R2
LOCAL_RETENTION_DAYS = 3  # keep today + yesterday + day-before under data/kg_backups/


def main() -> int:
    # Text only: with embeddings the archive passed 435 MB on 2026-09-03 and the
    # Cloudflare REST upload failed (413). Re-embed after a restore with
    # skills/kg-maintenance/scripts/embed_missing_facts.py.
    ts = _dump_kg(include_embeddings=False)

    backup_dir = ROOT / "data" / "kg_backups"
    dump_files = [
        backup_dir / f"entities_{ts}.json",
        backup_dir / f"edges_{ts}.json",
        backup_dir / f"mentions_{ts}.json",
    ]

    today = datetime.now(KST)
    archive_key = f"kg-backup-{today.strftime('%Y-%m-%d')}.tar.gz"
    local_archive = backup_dir / archive_key

    tmp_path = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False).name
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for f in dump_files:
                tar.add(f, arcname=f.name)
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        print(f"Archive built: {archive_key} ({size_mb:.1f} MB)")

        r2_put(BUCKET, archive_key, tmp_path, content_type="application/gzip")
        print(f"Uploaded to R2: {BUCKET}/{archive_key}")

        # Keep a local copy for fast restore (rolling LOCAL_RETENTION_DAYS).
        # Copy after upload succeeds so a failed upload doesn't leave a stale local copy.
        shutil.copyfile(tmp_path, local_archive)
        print(f"Saved local copy: {local_archive}")
    finally:
        os.unlink(tmp_path)

    r2_cutoff = (today - timedelta(days=R2_RETENTION_DAYS)).date()
    prune_r2_prefix(BUCKET, "kg-backup", ".tar.gz", r2_cutoff)

    local_cutoff = (today - timedelta(days=LOCAL_RETENTION_DAYS - 1)).date()
    prune_local_backups(backup_dir, "kg-backup", ".tar.gz", local_cutoff)

    for f in dump_files:
        f.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
