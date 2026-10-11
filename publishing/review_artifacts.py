"""Save publication-review reports supplied as final deliverables by the runtime."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ops.paths import PROJECT_ROOT


def save_review_report(relative_path: str, report: str) -> None:
    """Write atomically under the fixed review directory, with no model file probes."""
    root = (Path(PROJECT_ROOT) / "temp_dev" / "stasova_reviews").resolve()
    path = (Path(PROJECT_ROOT) / relative_path).resolve()
    if path.parent != root or path.suffix != ".md":
        raise ValueError("Review artifact path is outside the review directory")
    if not str(report or "").strip():
        raise ValueError("Review report is empty")
    root.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         delete=False) as stream:
            tmp_path = stream.name
            stream.write(report.strip() + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
