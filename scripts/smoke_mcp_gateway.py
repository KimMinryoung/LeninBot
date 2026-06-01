#!/usr/bin/env python3
"""Smoke checks for the inbound MCP Gateway policy and stdio protocol."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_gateway.policy import FORBIDDEN_TOOL_NAMES
from mcp_gateway.tools import build_tool_catalog


def _tool_names(profile: str) -> set[str]:
    return {tool["name"] for tool in build_tool_catalog(profile)}


def _assert_policy() -> None:
    inspect = _tool_names("inspect")
    readonly_alias = _tool_names("readonly")
    operator = _tool_names("operator")

    assert "vector_search" in inspect
    assert "knowledge_graph_search" in inspect
    assert "search_dev_docs" in inspect
    assert "corpus_metadata_audit" in inspect
    assert "kg_integrity_check" in inspect
    assert "readonly_query_db" not in inspect
    assert "bounded_query_db" not in inspect
    assert "kg_maintenance_run" not in inspect
    assert "readonly_query_db" in operator
    assert "bounded_query_db" in operator
    assert "kg_maintenance_run" in operator
    assert inspect == readonly_alias
    assert inspect < operator

    forbidden_inspect = sorted(inspect & FORBIDDEN_TOOL_NAMES)
    forbidden_operator = sorted((operator - {"readonly_query_db", "bounded_query_db", "kg_maintenance_run"}) & FORBIDDEN_TOOL_NAMES)
    assert not forbidden_inspect, f"inspect MCP exposes forbidden tools: {forbidden_inspect}"
    assert not forbidden_operator, f"operator MCP exposes forbidden tools: {forbidden_operator}"

    for profile in ("inspect", "operator"):
        for tool in build_tool_catalog(profile):
            assert "inputSchema" in tool, f"{profile}:{tool['name']} missing MCP inputSchema"
            assert "input_schema" not in tool, f"{profile}:{tool['name']} leaked Anthropic input_schema"


def _frame(message: dict) -> bytes:
    payload = json.dumps(message).encode("utf-8")
    return f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload


def _read_framed_messages(data: bytes) -> list[dict]:
    messages = []
    offset = 0
    while offset < len(data):
        header_end = data.find(b"\r\n\r\n", offset)
        assert header_end >= 0, data[offset:]
        headers = data[offset:header_end].decode("ascii")
        length = None
        for line in headers.split("\r\n"):
            key, _, value = line.partition(":")
            if key.lower() == "content-length":
                length = int(value.strip())
                break
        assert length is not None, headers
        payload_start = header_end + 4
        payload_end = payload_start + length
        messages.append(json.loads(data[payload_start:payload_end].decode("utf-8")))
        offset = payload_end
    return messages


def _assert_stdio_protocol() -> None:
    request = b"".join([
        _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        _frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        _frame({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "gateway_status", "arguments": {}},
        }),
    ])
    env = {**os.environ, "MCP_GATEWAY_PROFILE": "inspect"}
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_gateway.server"],
        input=request,
        capture_output=True,
        cwd=str(ROOT),
        env=env,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    lines = _read_framed_messages(proc.stdout)
    assert len(lines) == 3, proc.stdout
    assert lines[0]["result"]["serverInfo"]["name"] == "leninbot-mcp-gateway"
    listed = {tool["name"] for tool in lines[1]["result"]["tools"]}
    assert "gateway_status" in listed
    assert "readonly_query_db" not in listed
    assert "bounded_query_db" not in listed
    assert "kg_maintenance_run" not in listed
    status_text = lines[2]["result"]["content"][0]["text"]
    status = json.loads(status_text)
    assert status["status"] == "ok"
    assert status["profile"] == "inspect"
    assert status["write_tools_exposed"] is False


def _assert_cli_help() -> None:
    proc = subprocess.run(
        [str(ROOT / "scripts" / "mcp-gateway"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Default profile is inspect" in proc.stdout
    assert "--profile" in proc.stdout

    proc = subprocess.run(
        [str(ROOT / "scripts" / "mcp-gateway"), "--list-tools"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "gateway_status" in proc.stdout
    assert "readonly_query_db" not in proc.stdout
    assert "bounded_query_db" not in proc.stdout
    assert "kg_maintenance_run" not in proc.stdout

    proc = subprocess.run(
        [str(ROOT / "scripts" / "mcp-gateway"), "--profile", "operator", "--list-tools"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "readonly_query_db" in proc.stdout
    assert "bounded_query_db" in proc.stdout
    assert "kg_maintenance_run" in proc.stdout


def main() -> int:
    _assert_policy()
    _assert_stdio_protocol()
    _assert_cli_help()
    print("mcp gateway smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
