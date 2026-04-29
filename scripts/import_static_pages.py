#!/usr/bin/env python3
"""Import legacy static_pages/*.json payloads into the static_pages DB table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from site_publishing import STATIC_PAGES_DIR, _upsert_static_page_payload  # noqa: E402


def main() -> int:
    imported = 0
    updated = 0
    skipped = 0
    for path in sorted(STATIC_PAGES_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                skipped += 1
                continue
            payload.setdefault("slug", path.stem)
            if not payload.get("title") or not payload.get("html_body"):
                skipped += 1
                continue
            _row, existed = _upsert_static_page_payload(payload)
            if existed:
                updated += 1
            else:
                imported += 1
        except Exception as e:
            print(f"skip {path.name}: {type(e).__name__}: {e}", file=sys.stderr)
            skipped += 1
    print(f"static page import complete: imported={imported} updated={updated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
