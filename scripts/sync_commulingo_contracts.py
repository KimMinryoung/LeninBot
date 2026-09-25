#!/usr/bin/env python3
"""Copy the frontend's CommuLingo contract JSON files into this checkout.

The frontend repository owns these files; ``config/commulingo_contracts`` holds
copies so this checkout imports and tests without the frontend (see
``ops/paths.py``). Run after the frontend changes a contract and commit the
result.

Usage:
    scripts/sync_commulingo_contracts.py          # copy changed files
    scripts/sync_commulingo_contracts.py --check  # exit 1 if any copy differs
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.paths import COMMULINGO_DATA_DIR, CONTRACT_FILES, FRONTEND_DIR, VENDORED_CONTRACTS_DIR


def drifted() -> list[str]:
    """Contract file names whose vendored copy differs from the frontend original."""
    return [name for name in CONTRACT_FILES
            if not (VENDORED_CONTRACTS_DIR / name).exists()
            or (COMMULINGO_DATA_DIR / name).read_bytes() != (VENDORED_CONTRACTS_DIR / name).read_bytes()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report drift without copying")
    args = parser.parse_args()
    if not FRONTEND_DIR.exists():
        print(f"FRONTEND_DIR {FRONTEND_DIR} not found; nothing to compare", file=sys.stderr)
        return 2
    names = drifted()
    if args.check:
        for name in names:
            print(f"drift: {name}")
        return 1 if names else 0
    VENDORED_CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        (VENDORED_CONTRACTS_DIR / name).write_bytes((COMMULINGO_DATA_DIR / name).read_bytes())
        print(f"copied: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
