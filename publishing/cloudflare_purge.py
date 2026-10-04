"""Cloudflare cache purge through the frontend's ``scripts/cloudflare-purge.js``.

Off unless ``LENINBOT_CLOUDFLARE_PURGE=1``: the frontend caches public HTML at
the edge for only 60 s, and an agent verifies a change through the origin (a
unique ``?verify=<timestamp>`` query or ``http://127.0.0.1:3000``) instead of
waiting for a purge. Turn it on if the edge TTL is ever made long.

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


# Paths the frontend serves in one language only; every other public path
# also has an English copy under /en/ (frontend isLanguageSpecificPublicPath),
# which Cloudflare caches separately.
_SINGLE_LANGUAGE_PATHS = {"/sitemap.xml", "/robots.txt"}


def purge_enabled() -> bool:
    return os.getenv("LENINBOT_CLOUDFLARE_PURGE", "0") == "1"


# Shown to the agent in place of a purge count when purging is off.
SKIPPED_NOTE = (
    "Cloudflare purge off (edge copies refresh within 60 s; verify with "
    "?verify=<timestamp> on the public URL)"
)


def with_english_paths(paths: list[str]) -> list[str]:
    """Add the /en/ copy of each language-specific path ("/" becomes "/en/")."""
    out: list[str] = []
    for path in paths:
        out.append(path)
        if path in _SINGLE_LANGUAGE_PATHS or not path.startswith("/") or path.startswith("/en/"):
            continue
        out.append("/en/" if path == "/" else f"/en{path}")
    return out


def purge_paths(paths: list[str], label: str) -> dict[str, Any]:
    """Purge ``paths`` and their /en/ copies (deduplicated, order kept);
    ``label`` names the target in logs."""
    paths = list(dict.fromkeys(with_english_paths(paths)))
    if not paths:
        return {"ok": True, "purged": 0, "urls": []}
    if not purge_enabled():
        return {"ok": True, "purged": 0, "urls": paths, "skipped": True}
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
