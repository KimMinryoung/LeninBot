"""Shared plumbing for the three R2 backup scripts (main/writer/kg).

Entry points stay separate — each has its own systemd timer and watchdog
ping — this module only holds the identical helpers they all forked.

IMPORTANT: promote_systemd_credentials() must run before importing
requests/secrets_loader in the calling script, exactly where the old
per-script copy ran.
"""
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


R2_CREDENTIAL_PAIRS = (
    ("r2_cf_api_token", "R2_CF_API_TOKEN"),
    ("r2_s3_access_key_id", "R2_S3_ACCESS_KEY_ID"),
    ("r2_s3_secret_access_key", "R2_S3_SECRET_ACCESS_KEY"),
)


def promote_systemd_credentials(pairs: tuple[tuple[str, str], ...] = R2_CREDENTIAL_PAIRS) -> None:
    """Expose LoadCredentialEncrypted secrets to legacy env-based helpers."""
    cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if not cred_dir:
        return
    for cred_name, env_name in pairs:
        if os.environ.get(env_name):
            continue
        path = Path(cred_dir) / cred_name
        if path.is_file():
            os.environ[env_name] = path.read_text().rstrip("\n")


_TRANSFER_PART = 64 * 1024 * 1024


def r2_client():
    """S3 client for R2, authenticated by the bucket-scoped key pair.

    Every backup-side R2 call (upload, list, download, delete) goes through
    this one key, so a host that only runs backups (the standby) needs no
    account-wide Cloudflare token.
    """
    import boto3
    from botocore.config import Config
    from secrets_loader import require_secret

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_CF_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=require_secret("R2_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=require_secret("R2_S3_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(multipart_threshold=_TRANSFER_PART, multipart_chunksize=_TRANSFER_PART)


def r2_put(bucket: str, key: str, path: str, content_type: str = "application/octet-stream") -> None:
    """Upload through R2's S3 API, which switches to multipart for large files.

    The Cloudflare REST object endpoint takes the whole body in one request and
    rejects it with 413 above roughly 300 MiB; the main DB dump crossed that on
    2026-09-20 and every upload failed until this moved to the S3 API.
    """
    r2_client().upload_file(
        path, bucket, key, ExtraArgs={"ContentType": content_type}, Config=_transfer_config()
    )


def r2_get(bucket: str, key: str, path: str) -> None:
    r2_client().download_file(bucket, key, path, Config=_transfer_config())


def r2_list_keys(bucket: str, key_prefix: str) -> list[str]:
    """Return every object key under key_prefix, following pagination."""
    keys: list[str] = []
    paginator = r2_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return keys


def r2_delete(bucket: str, key: str) -> None:
    r2_client().delete_object(Bucket=bucket, Key=key)


def prune_local_backups(backup_dir: Path, key_prefix: str, suffix: str, cutoff) -> None:
    """Delete local {key_prefix}-YYYY-MM-DD[_HHMMSS]{suffix} files dated before cutoff."""
    import re
    key_re = re.compile(rf"^{re.escape(key_prefix)}-(\d{{4}}-\d{{2}}-\d{{2}})(?:_\d{{6}})?{re.escape(suffix)}$")
    for p in backup_dir.glob(f"{key_prefix}-*{suffix}"):
        m = key_re.match(p.name)
        if not m:
            continue
        try:
            file_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff:
            p.unlink(missing_ok=True)
            print(f"Pruned local copy: {p.name}")


# Per-table exact counts without temp tables, so it also runs on the
# read-only hot standby the backups dump from.
ROW_COUNT_SQL = """
SELECT tablename || '|' || (xpath('/row/c/text()', query_to_xml(
  format('SELECT count(*) AS c FROM public.%I', tablename), false, true, '')))[1]::text
FROM pg_tables WHERE schemaname='public' ORDER BY tablename;
"""

MANIFEST_SUFFIX = ".manifest.json"
# A table that held at least SHRINK_MIN_ROWS rows and lost more than
# SHRINK_MAX_DROP of them since the previous backup is reported. The dump is
# still uploaded; the job fails afterwards so the watchdog ping is withheld.
SHRINK_MIN_ROWS = 1000
SHRINK_MAX_DROP = 0.5
DISK_WARN_FRACTION = 0.85


def count_rows(container: str, database: str) -> dict[str, int]:
    out = subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1",
         "-U", "postgres", "-d", database, "-At"],
        input=ROW_COUNT_SQL,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    counts = {}
    for line in out.splitlines():
        name, _, value = line.rpartition("|")
        if name:
            counts[name] = int(value)
    return counts


def try_count_rows(container: str, database: str) -> dict[str, int] | None:
    """count_rows for the manifest; a counting failure must not cost the dump."""
    try:
        return count_rows(container, database)
    except Exception as exc:
        print(f"Row count failed for {database}: {exc!r}")
        return None


def manifest_name(archive_key: str) -> str:
    return archive_key.removesuffix(".dump") + MANIFEST_SUFFIX


def shrunk_tables(previous: dict[str, int], current: dict[str, int]) -> list[str]:
    """Tables that were large in `previous` and are gone or sharply smaller now."""
    findings = []
    for table, before in sorted(previous.items()):
        if before < SHRINK_MIN_ROWS:
            continue
        now = current.get(table)
        if now is None:
            findings.append(f"{table}: {before} -> missing")
        elif now < before * (1 - SHRINK_MAX_DROP):
            findings.append(f"{table}: {before} -> {now}")
    return findings


def _previous_manifest(backup_dir: Path, key_prefix: str, name: str) -> dict | None:
    older = sorted(
        p for p in backup_dir.glob(f"{key_prefix}-*{MANIFEST_SUFFIX}") if p.name < name
    )
    if not older:
        return None
    try:
        return json.loads(older[-1].read_text())
    except (OSError, ValueError) as exc:
        print(f"WARNING: unreadable previous manifest {older[-1].name}: {exc}")
        return None


def publish_manifest(
    bucket: str,
    backup_dir: Path,
    key_prefix: str,
    archive_key: str,
    database: str,
    rows_before: dict[str, int],
    rows_after: dict[str, int],
) -> list[str]:
    """Save and upload the dump's row-count manifest; return shrink findings.

    rows_before/rows_after bracket the dump, which ran on a live replica, so
    a restore drill checks each table against that range.
    """
    name = manifest_name(archive_key)
    previous = _previous_manifest(backup_dir, key_prefix, name)
    manifest = {
        "database": database,
        "archive": archive_key,
        "taken_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows_before": rows_before,
        "rows_after": rows_after,
    }
    with tempfile.NamedTemporaryFile("w", suffix=MANIFEST_SUFFIX, delete=False) as tmp:
        json.dump(manifest, tmp, indent=1, sort_keys=True)
    try:
        shutil.copyfile(tmp.name, backup_dir / name)
        r2_put(bucket, name, tmp.name, content_type="application/json")
        print(f"Uploaded manifest: {bucket}/{name} ({len(rows_before)} tables)")
    finally:
        os.unlink(tmp.name)
    if previous is None:
        print(f"No earlier manifest for {key_prefix}; shrink check starts tomorrow")
        return []
    return shrunk_tables(previous.get("rows_after", {}), rows_before)


def disk_warning(path: Path) -> str | None:
    usage = shutil.disk_usage(path)
    used = 1 - usage.free / usage.total
    if used < DISK_WARN_FRACTION:
        return None
    return f"disk {used:.0%} used ({usage.free / 1024**3:.1f} GiB free) at {path}"
