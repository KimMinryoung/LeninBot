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


def promote_systemd_credentials(pairs: tuple[tuple[str, str], ...] = (("r2_cf_api_token", "R2_CF_API_TOKEN"),)) -> None:
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


def r2_url(bucket: str, key: str) -> str:
    acct = os.environ["R2_CF_ACCOUNT_ID"]
    return f"https://api.cloudflare.com/client/v4/accounts/{acct}/r2/buckets/{bucket}/objects/{key}"


def r2_headers() -> dict:
    from secrets_loader import require_secret
    return {"Authorization": f"Bearer {require_secret('R2_CF_API_TOKEN')}"}


def r2_put(bucket: str, key: str, path: str, content_type: str = "application/octet-stream") -> None:
    import requests
    with open(path, "rb") as f:
        data = f.read()
    resp = requests.put(
        r2_url(bucket, key),
        headers={**r2_headers(), "Content-Type": content_type},
        data=data,
        timeout=300,
    )
    resp.raise_for_status()


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
