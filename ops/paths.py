"""Filesystem roots for this checkout and the separate frontend repository.

The frontend (Node/Express, CommuLingo data, report caches) is a separate
repository, so its location cannot be derived from this file; ``FRONTEND_DIR``
overrides the production default. Per-file ``COMMULINGO_*`` variables still
override individual data files, which is how tests point at fixtures.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "/home/grass/frontend"))
COMMULINGO_DATA_DIR = FRONTEND_DIR / "data" / "commulingo"


def commulingo_data_file(name: str, env_var: str | None = None) -> Path:
    """Path of a frontend-owned CommuLingo data file, honoring ``env_var``."""
    override = os.environ.get(env_var) if env_var else None
    return Path(override) if override else COMMULINGO_DATA_DIR / name
