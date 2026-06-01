"""Minimal stdio MCP server for the inbound LeninBot gateway."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
from typing import Any

from mcp_gateway.policy import INSPECT_PROFILE, OPERATOR_PROFILE, normalize_profile
from mcp_gateway.tools import build_handlers, build_tool_catalog

logger = logging.getLogger(__name__)


JsonRequest = dict[str, Any]


def _result(request_id: Any, value: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _read_message_sync() -> tuple[JsonRequest | None, str]:
    """Read one stdio MCP message.

    Real MCP stdio clients use Content-Length framing. The line-delimited path
    remains for local smoke tests and simple manual probes.
    """
    first = sys.stdin.buffer.readline()
    if not first:
        return None, "headers"
    if first.startswith(b"Content-Length:"):
        headers = [first]
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None, "headers"
            if line in (b"\r\n", b"\n"):
                break
            headers.append(line)
        length = None
        for header in headers:
            key, _, value = header.decode("ascii", errors="replace").partition(":")
            if key.lower() == "content-length":
                length = int(value.strip())
                break
        if length is None:
            raise ValueError("Missing Content-Length header")
        payload = sys.stdin.buffer.read(length)
        return json.loads(payload.decode("utf-8")), "headers"
    return json.loads(first.decode("utf-8")), "line"


def _write(message: dict[str, Any], framing: str) -> None:
    payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
    if framing == "line":
        sys.stdout.buffer.write(payload + b"\n")
    else:
        sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload)
    sys.stdout.buffer.flush()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mcp_gateway.server",
        description=(
            "Run the LeninBot inbound MCP Gateway over stdio. "
            "Default profile is inspect: docs, task/report/corpus status, "
            "and selected runtime search tools. Use operator only for trusted "
            "local sessions that need readonly_query_db via scripts/query-db."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=[INSPECT_PROFILE, OPERATOR_PROFILE, "readonly"],
        default=None,
        help="MCP profile. Default: $MCP_GATEWAY_PROFILE or inspect. readonly is a deprecated alias for inspect.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the selected profile's tool names and exit instead of starting stdio MCP.",
    )
    return parser.parse_args(argv)


def _resolve_profile(args: argparse.Namespace) -> str:
    return normalize_profile(args.profile or os.getenv("MCP_GATEWAY_PROFILE"))


async def _call_tool(name: str, arguments: dict[str, Any], profile: str) -> dict[str, Any]:
    handlers = build_handlers(profile)
    handler = handlers.get(name)
    if handler is None:
        return {
            "content": [{"type": "text", "text": f"Tool is not exposed by MCP profile '{profile}': {name}"}],
            "isError": True,
        }
    try:
        call_args = dict(arguments or {})
        signature = inspect.signature(handler)
        if "profile" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            call_args.setdefault("profile", profile)
        result = handler(**call_args)
        if inspect.isawaitable(result):
            text = await result
        else:
            text = result
        return {"content": [{"type": "text", "text": str(text)}], "isError": False}
    except TypeError as exc:
        return {"content": [{"type": "text", "text": f"Invalid arguments for {name}: {exc}"}], "isError": True}
    except Exception as exc:
        logger.exception("MCP tool call failed: %s", name)
        return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}


async def handle(request: dict[str, Any], profile: str) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}

    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None

    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "leninbot-mcp-gateway", "version": "0.1.0"},
        })
    if method == "tools/list":
        return _result(request_id, {"tools": build_tool_catalog(profile)})
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "tools/call arguments must be an object")
        return _result(request_id, await _call_tool(name, arguments, profile))
    if method == "ping":
        return _result(request_id, {})
    if request_id is None:
        return None
    return _error(request_id, -32601, f"Method not found: {method}")


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=os.getenv("MCP_GATEWAY_LOG_LEVEL", "WARNING"))
    profile = _resolve_profile(args)
    if args.list_tools:
        for tool in build_tool_catalog(profile):
            print(tool["name"])
        return 0
    framing = "headers"
    while True:
        try:
            request, framing = await asyncio.to_thread(_read_message_sync)
            if request is None:
                break
            if not isinstance(request, dict):
                _write(_error(None, -32600, "Invalid JSON-RPC request"), framing)
                continue
            response = await handle(request, profile)
            if response is not None:
                _write(response, framing)
        except json.JSONDecodeError as exc:
            _write(_error(None, -32700, f"Parse error: {exc}"), framing)
        except Exception as exc:
            logger.exception("MCP request failed")
            request_id = request.get("id") if isinstance(locals().get("request"), dict) else None
            _write(_error(request_id, -32603, f"Internal error: {type(exc).__name__}: {exc}"), framing)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
