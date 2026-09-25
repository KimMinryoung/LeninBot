"""Filesystem roots for this checkout and the separate frontend repository.

The frontend (Node/Express, CommuLingo data, report caches) is a separate
repository, so its location cannot be derived from this file; ``FRONTEND_DIR``
overrides the production default. Per-file ``COMMULINGO_*`` variables still
override individual data files.

The frontend owns the CommuLingo contract JSON files that both repositories
validate against. This checkout keeps copies in ``config/commulingo_contracts``
so it can import and test without the frontend (a lone clone in a cloud
session). The copies are used only when ``FRONTEND_DIR`` does not exist at all;
a production host with the frontend present always reads the originals.
``scripts/sync_commulingo_contracts.py`` refreshes the copies.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "/home/grass/frontend"))
COMMULINGO_DATA_DIR = FRONTEND_DIR / "data" / "commulingo"

VENDORED_CONTRACTS_DIR = PROJECT_ROOT / "config" / "commulingo_contracts"
CONTRACT_FILES = (
    "person-editorial-contract.json",
    "nationality-policy.json",
    "activity-schema.json",
    "activity-catalog.json",
)

_warned_vendored = False

# Checked-in configs name files under these roots as ``${NAME}/...`` so they
# stay valid on any host; ``expand_path_tokens`` resolves them at load time.
_PATH_TOKENS = {"${FRONTEND_DIR}": FRONTEND_DIR, "${PROJECT_ROOT}": PROJECT_ROOT}


def expand_path_tokens(value):
    """Resolve a leading ``${FRONTEND_DIR}``/``${PROJECT_ROOT}`` in every string of
    ``value`` (recursing through dicts and lists)."""
    if isinstance(value, str):
        for token, root in _PATH_TOKENS.items():
            if value == token or value.startswith(token + "/"):
                return str(root) + value[len(token):]
        return value
    if isinstance(value, dict):
        return {k: expand_path_tokens(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_path_tokens(v) for v in value]
    return value


def commulingo_data_file(name: str, env_var: str | None = None) -> Path:
    """Path of a frontend-owned CommuLingo data file, honoring ``env_var``."""
    global _warned_vendored
    override = os.environ.get(env_var) if env_var else None
    if override:
        return Path(override)
    if name in CONTRACT_FILES and not FRONTEND_DIR.exists():
        if not _warned_vendored:
            logger.warning("FRONTEND_DIR %s is missing; using vendored CommuLingo contracts in %s",
                           FRONTEND_DIR, VENDORED_CONTRACTS_DIR)
            _warned_vendored = True
        return VENDORED_CONTRACTS_DIR / name
    return COMMULINGO_DATA_DIR / name
