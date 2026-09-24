#!/usr/bin/env python3
"""Compose a 4-panel political comic page for cyber-lenin.com.

CLI wrapper around ``runtime_tools.comic_composer``, which holds the layout,
content rules and payload schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools.comic_composer import (  # noqa: E402
    build_html_body,
    build_page_payload,
    validate_payload,
)
from site_publishing import _exec_publish_static_page  # type: ignore  # noqa: E402


async def publish_payload(payload: dict[str, Any]) -> str:
    page = build_page_payload(payload)
    return await _exec_publish_static_page(**page)


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or publish a 4-panel political comic static page "
        "from agent-authored scene SVGs.",
    )
    parser.add_argument("payload", help="Path to JSON payload.")
    parser.add_argument(
        "--print-html",
        action="store_true",
        help="Print html_body instead of the JSON page payload.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish directly into static_pages via site_publishing.",
    )
    args = parser.parse_args()

    payload = load_payload(Path(args.payload))
    if args.publish:
        import asyncio
        print(asyncio.run(publish_payload(payload)))
        return

    page_payload = build_page_payload(payload)
    if args.print_html:
        print(page_payload["html_body"])
    else:
        print(json.dumps(page_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
