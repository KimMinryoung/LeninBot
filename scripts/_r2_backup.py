"""Shared plumbing for the three R2 backup scripts (main/writer/kg).

Entry points stay separate — each has its own systemd timer and watchdog
ping — this module only holds the identical helpers they all forked.

IMPORTANT: promote_systemd_credentials() must run before importing
requests/secrets_loader in the calling script, exactly where the old
per-script copy ran.
"""
import os
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
    """Delete local {key_prefix}-YYYY-MM-DD{suffix} files dated before cutoff."""
    import re
    key_re = re.compile(rf"^{re.escape(key_prefix)}-(\d{{4}}-\d{{2}}-\d{{2}}){re.escape(suffix)}$")
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
