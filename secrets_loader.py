"""Unified secret access with systemd→.env fallback.

Priority:
  1. $CREDENTIALS_DIRECTORY/<name_lower>  (systemd LoadCredentialEncrypted)
  2. os.environ[<NAME>]                   (env var, loaded from .env via dotenv)

Production services mount secrets at $CREDENTIALS_DIRECTORY (tmpfs, 0400,
service-private) via systemd's credential system. Development uses .env.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Derive PROJECT_ROOT from this file's location if nothing already set it (e.g.
# via .env or systemd Environment=). Machines that clone the repo don't need to
# configure a path — the code finds itself.
os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parent))


def _cred_dir() -> Path | None:
    d = os.environ.get("CREDENTIALS_DIRECTORY")
    return Path(d) if d else None


def _cred_filename(name: str) -> str:
    return name.lower()


@lru_cache(maxsize=None)
def get_secret(name: str, default: str | None = None) -> str | None:
    """Return secret by UPPER_SNAKE name. Credential file first, env var fallback.

    Credential filename is the name lowercased (GEMINI_API_KEY → gemini_api_key).
    Cached for the lifetime of the process; restart the service to pick up
    rotated values.

    When a value is read from the systemd credential mount, it is also
    published to ``os.environ`` (via ``setdefault``) so third-party libraries
    that read env vars directly (e.g. the openai / anthropic SDKs used inside
    graphiti-core) still pick it up without knowing about credstore.
    """
    cred = _cred_dir()
    if cred is not None:
        path = cred / _cred_filename(name)
        if path.is_file():
            try:
                value = path.read_text(encoding="utf-8").rstrip("\n")
                os.environ.setdefault(name, value)
                return value
            except OSError:
                pass
    val = os.environ.get(name)
    if val is not None:
        return val
    return default


def require_secret(name: str) -> str:
    val = get_secret(name)
    if not val:
        raise RuntimeError(f"Required secret missing: {name}")
    return val
