#!/usr/bin/env python3
"""Smoke checks for edit_content static_page support."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools.post_edit import EDIT_CONTENT_TOOL, _exec_edit_content


async def main() -> None:
    enum = EDIT_CONTENT_TOOL["input_schema"]["properties"]["content_type"]["enum"]
    assert "static_page" in enum

    result = await _exec_edit_content(
        content_type="static_page",
        slug="test-page",
        html_body="<script>alert(1)</script>",
    )
    assert "unsafe HTML" in result and "<script" in result, result

    result = await _exec_edit_content(
        content_type="static_page",
        slug="test-page",
        content="wrong field",
    )
    assert "not editable" in result and "html_body" in result, result

    result = await _exec_edit_content(content_type="static_page")
    assert "slug is required" in result, result


if __name__ == "__main__":
    asyncio.run(main())
