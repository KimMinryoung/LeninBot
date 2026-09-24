"""Cloudflare cache purge through the frontend's ``scripts/cloudflare-purge.js``.

Failure is reported to the caller but never raises: the database update and
Redis invalidation are the source-of-truth changes, the purge only shortens
how long the edge serves the old page.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from ops import paths as _paths

logger = logging.getLogger(__name__)

FRONTEND_DIR = str(_paths.FRONTEND_DIR)
CF_PURGE_SCRIPT = os.getenv(
    "CF_PURGE_SCRIPT",
    os.path.join(FRONTEND_DIR, "scripts", "cloudflare-purge.js"),
)


def purge_paths(paths: list[str], label: str) -> dict[str, Any]:
    """Purge ``paths`` (deduplicated, order kept); ``label`` names the target in logs."""
    paths = list(dict.fromkeys(paths))
    if not paths:
        return {"ok": True, "purged": 0, "urls": []}
    if not os.path.isfile(CF_PURGE_SCRIPT):
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": f"script_missing: {CF_PURGE_SCRIPT}",
        }

    try:
        proc = subprocess.run(
            ["node", CF_PURGE_SCRIPT, *paths],
            cwd=FRONTEND_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except Exception as e:
        logger.warning("Cloudflare purge failed before execution (%s): %s", label, e)
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": f"{type(e).__name__}: {e}",
        }

    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if proc.returncode != 0:
        logger.warning("Cloudflare purge failed (%s, exit=%s): %s", label, proc.returncode, output)
        return {
            "ok": False,
            "purged": 0,
            "urls": paths,
            "reason": output or f"exit_{proc.returncode}",
        }
    return {"ok": True, "purged": len(paths), "urls": paths, "output": output}
