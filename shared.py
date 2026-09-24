"""shared.py — Small helpers shared across web_chat, the Telegram bot, and agents.

Lightweight module — no heavy dependencies. Domain code lives in its own
package (memory_store, kg_runtime, corpus, content_fetch, ops, …); import
from there rather than growing this module.
"""

import logging
import os
from datetime import timezone, timedelta

from secrets_loader import get_secret

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))


def upload_to_r2(local_path: str, key: str | None = None, content_type: str | None = None) -> str | None:
    """Upload a file to Cloudflare R2 and return its public URL.

    Args:
        local_path: Path to local file.
        key: Object key in R2 bucket. Defaults to filename.
        content_type: MIME type. Auto-detected from extension if not given.

    Returns public URL string, or None on failure.
    """
    import mimetypes
    import requests as _req

    cf_token = (get_secret("R2_CF_API_TOKEN", "") or "").strip()
    account_id = os.getenv("R2_CF_ACCOUNT_ID", "").strip()
    bucket = os.getenv("R2_BUCKET_NAME", "").strip()
    public_url = os.getenv("R2_PUBLIC_URL", "").strip().rstrip("/")

    if not all([cf_token, account_id, bucket, public_url]):
        logger.warning("[shared] R2 upload skipped: missing env config")
        return None

    from pathlib import Path as _Path
    path = _Path(local_path)
    if not path.is_file():
        logger.warning("[shared] R2 upload skipped: file not found: %s", local_path)
        return None

    if key is None:
        key = path.name
    if content_type is None:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    try:
        with open(path, "rb") as f:
            data = f.read()
        resp = _req.put(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/buckets/{bucket}/objects/{key}",
            headers={"Authorization": f"Bearer {cf_token}", "Content-Type": content_type},
            data=data,
            timeout=60,
        )
        resp.raise_for_status()
        url = f"{public_url}/{key}"
        logger.info("[shared] R2 uploaded: %s -> %s", local_path, url)
        return url
    except Exception as e:
        logger.error("[shared] R2 upload failed for %s: %s", local_path, e)
        return None


# Module architecture description — static, for bot self-awareness
MODULE_ARCHITECTURE = """\
## Architecture
Modules: telegram/bot.py (multi-agent orchestrator), telegram/tasks.py (background task worker), \
agents/ (AgentSpec registry), runtime_tools/ (tool registry), services/api.py (FastAPI), \
services/web_chat.py (web chat pipeline), kg_runtime/ + graph_memory/ (Neo4j KG), llm/ (provider loops and gateway).
Data: PostgreSQL (local Docker, leninbot-pg), Neo4j (local Docker), Redis (live state).
## Infrastructure
Server: Hetzner VPS (Ubuntu 24.04, 16 GB RAM), HTTPS via Nginx + Cloudflare Origin Certificate (cyber-lenin.com). \
Deploy: git pull + systemctl restart, triggered by Telegram /deploy command."""
