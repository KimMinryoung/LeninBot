#!/usr/bin/env python3
"""Smoke checks for full static_page reads through read_self."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import site_publishing
from self_runtime.tools import _exec_read_self
from tool_loop_common import execute_tool


LONG_BODY = "<p>" + ("가" * 60_000) + "</p>"
EN_BODY = "<p>" + ("A" * 5_000) + "</p>"


async def main() -> None:
    original_get_static_page = site_publishing.get_static_page

    def fake_get_static_page(slug: str) -> dict:
        return {
            "slug": slug,
            "title": "긴 정적 페이지",
            "summary": "본문 전문 반환 검사",
            "html_body": LONG_BODY,
            "title_en": "Long static page",
            "summary_en": "Full body smoke",
            "html_body_en": EN_BODY,
            "updated_at": "2026-05-10T00:00:00+00:00",
        }

    try:
        site_publishing.get_static_page = fake_get_static_page
        direct = await _exec_read_self(content_type="static_page", slug="test-full-page")
        assert LONG_BODY in direct, "direct read_self static_page detail truncated html_body"
        assert "title_en" not in direct
        assert "summary_en" not in direct
        assert "html_body_en" not in direct
        assert EN_BODY not in direct
        assert "... [truncated]" not in direct
        assert "preview" not in direct

        paged = await _exec_read_self(
            content_type="static_page",
            slug="test-full-page",
            max_chars=120,
        )
        assert LONG_BODY not in paged
        assert "truncated=True" in paged
        assert "next: read_self(content_type='static_page'" in paged
        assert "next_en:" not in paged
        assert "html_body_en" not in paged

        tool_result, is_error = await execute_tool(
            "read_self",
            {"content_type": "static_page", "slug": "test-full-page"},
            {"read_self": _exec_read_self},
        )
        assert not is_error, tool_result
        assert LONG_BODY in tool_result, "tool-loop safety cap truncated static_page detail"
        assert EN_BODY not in tool_result
        assert "html_body_en" not in tool_result
        assert "... [truncated]" not in tool_result
    finally:
        site_publishing.get_static_page = original_get_static_page


if __name__ == "__main__":
    asyncio.run(main())
