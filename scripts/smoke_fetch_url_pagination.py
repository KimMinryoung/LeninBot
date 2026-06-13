#!/usr/bin/env python3
"""Smoke tests for fetch_url pagination and tool-result caps."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_tools.fetch import _exec_fetch_url
from tool_loop_common import execute_tool
import content_fetch.urls as url_fetch
import self_runtime.tools as self_tools
import mcp_gateway.tools as mcp_tools
from runtime_tools.registry import _parse_email_message


SOURCE = "A" * 1000 + "B" * 1000 + "C" * 1000


async def _fake_fetch_url_content_async(url: str, max_chars: int = 10000):
    return SOURCE[:max_chars]


async def _main() -> None:
    original = url_fetch.fetch_url_content_async
    url_fetch.fetch_url_content_async = _fake_fetch_url_content_async
    try:
        first = await _exec_fetch_url("https://example.test/long", max_chars=1000)
        assert "chars 0:1000" in first, first
        assert "offset=1000" in first, first
        assert "A" * 80 in first, first
        assert "B" * 80 not in first, first

        second = await _exec_fetch_url("https://example.test/long", max_chars=1000, offset=1000)
        assert "chars 1000:2000" in second, second
        assert "offset=2000" in second, second
        assert "B" * 80 in second, second
        assert "A" * 80 not in second, second

        third = await _exec_fetch_url("https://example.test/long", max_chars=1000, offset=2000)
        assert "chars 2000:3000" in third, third
        assert "truncated=False" in third, third
        assert "offset=3000" not in third, third
        assert "C" * 80 in third, third
    finally:
        url_fetch.fetch_url_content_async = original

    import memory_store.queries as memory_queries

    original_fetch_diaries = memory_queries.fetch_diaries
    memory_queries.fetch_diaries = lambda limit, keyword, target_id: [{
        "id": 7,
        "created_at": None,
        "updated_at": None,
        "title": "slice diary",
        "content": SOURCE,
    }]
    try:
        diary = await self_tools._exec_read_self(
            content_type="diary", id=7, max_chars=1000, offset=1000
        )
        assert "returned_chars=1000:2000" in diary, diary
        assert "B" * 80 in diary, diary
        assert "A" * 80 not in diary, diary
    finally:
        memory_queries.fetch_diaries = original_fetch_diaries

    original_db = sys.modules.get("db")
    fake_db = types.SimpleNamespace(
        query=lambda sql, params: [{
            "id": 9,
            "user_id": 1,
            "status": "done",
            "priority": 0,
            "agent_type": "analyst",
            "mission_id": None,
            "parent_task_id": None,
            "depth": 0,
            "plan_id": None,
            "plan_role": None,
            "created_at": None,
            "updated_at": None,
            "available_at": None,
            "metadata": {},
            "content": "request",
            "result": SOURCE,
            "tool_log": "log",
        }]
    )
    sys.modules["db"] = fake_db
    try:
        task_status = await mcp_tools.get_task_status(
            9, include_result=True, field="result", max_chars=1000, offset=1000
        )
        task_status_json = json.loads(task_status)
        assert task_status_json["selected_field"] == "result", task_status
        assert task_status_json["returned_chars"] == [1000, 2000], task_status
        assert "B" * 80 in task_status_json["result"], task_status
        assert "A" * 80 not in task_status_json["result"], task_status
    finally:
        if original_db is None:
            sys.modules.pop("db", None)
        else:
            sys.modules["db"] = original_db

    raw_email = (
        "From: sender@example.test\n"
        "Subject: Long body\n"
        "Date: Mon, 01 Jan 2024 00:00:00 +0000\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        + SOURCE
    ).encode("utf-8")
    email_page = _parse_email_message(
        raw_email, include_body=True, body_max_chars=1000, body_offset=1000
    )
    assert email_page["body_start"] == 1000, email_page
    assert email_page["body_end"] == 2000, email_page
    assert email_page["body_truncated"] is True, email_page
    assert "B" * 80 in email_page["body"], email_page
    assert "A" * 80 not in email_page["body"], email_page

    async def long_result():
        return "x" * 80000

    async def very_long_result():
        return "y" * 60000

    read_result, read_error = await execute_tool("read_file", {}, {"read_file": long_result})
    assert not read_error
    assert len(read_result) == 80000, len(read_result)
    assert "truncated by tool loop" not in read_result

    other_result, other_error = await execute_tool("other_tool", {}, {"other_tool": very_long_result})
    assert not other_error
    assert len(other_result) > 50000 and len(other_result) < 51000, len(other_result)
    assert "truncated by tool loop at 50000 chars" in other_result

    print("ok")


if __name__ == "__main__":
    asyncio.run(_main())
