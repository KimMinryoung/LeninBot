#!/usr/bin/env python3
"""Read-only paid search/extraction usage, grouped by service or task."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_gateway.client import usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--by", choices=("service", "task"), default="service")
    args = parser.parse_args()
    if not 1 <= args.days <= 366:
        parser.error("--days must be between 1 and 366")
    print(json.dumps(usage(args.days, args.by), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
