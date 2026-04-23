"""Unified secret access with systemd→.env fallback.

Priority:
  1. $CREDENTIALS_DIRECTORY/<name_lower>  (systemd LoadCredentialEncrypted)
  2. os.environ[<NAME>]                   (env var, loaded from .env via dotenv)

Production services mount secrets at $CREDENTIALS_DIRECTORY (tmpfs, 0400,
service-private) via systemd's credential system. Development uses .env.
"""

from __future__ import annotations

import os
import re
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


# Filename must be a valid lowercase env-var shape (letters, digits, underscore).
# Excludes things like "eth.privkey" that aren't env vars.
_ENV_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _bridge_credentials_to_env() -> None:
    """Populate os.environ from every credential in CREDENTIALS_DIRECTORY.

    Third-party SDKs (openai, anthropic, langchain_tavily, graphiti-core's
    cross_encoder, etc.) read API keys directly from os.environ at client
    construction — they don't know about credstore. Running this at import
    time ensures those reads succeed even when the value only lives in the
    systemd credential mount.

    Uses ``setdefault``: shell exports and .env values (loaded just above)
    win over credstore, so a developer can override for local testing.
    Files whose name isn't a valid UPPER_SNAKE env var (e.g. "eth.privkey")
    are skipped.
    """
    cred = _cred_dir()
    if cred is None:
        return
    try:
        entries = list(cred.iterdir())
    except OSError:
        return
    for path in entries:
        if not path.is_file():
            continue
        if not _ENV_NAME_RE.match(path.name):
            continue
        name = path.name.upper()
        if name in os.environ:
            continue
        try:
            value = path.read_text(encoding="utf-8").rstrip("\n")
        except OSError:
            continue
        os.environ[name] = value


_bridge_credentials_to_env()


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
