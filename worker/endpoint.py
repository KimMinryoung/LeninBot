"""Worker MCP endpoint: stateless Streamable HTTP (POST /worker/mcp, one
JSON-RPC message per request), mounted on leninbot-api (the port the frontend
already reaches). It only queues, reads and cancels tasks; worker/service.py
executes them in its own process. dev_docs/agent_worker.md.

Clients authenticate with a bearer token; .env WORKER_MCP_CLIENTS holds only
"name:sha256hex" entries separated by ';' (scripts/worker_token.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from worker import runner, store

logger = logging.getLogger("worker")

PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
TOKEN_RE = re.compile(r"^Bearer ([A-Za-z0-9_-]{32,128})$")
KEY_RE = re.compile(r"^[A-Za-z0-9:_.-]{1,180}$")


def parse_clients(spec: str | None) -> dict[str, bytes]:
    clients = {}
    for entry in filter(None, (part.strip() for part in (spec or "").split(";"))):
        name, _, digest = entry.partition(":")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", name) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"WORKER_MCP_CLIENTS: invalid entry for {name or '?'}")
        clients[name] = bytes.fromhex(digest)
    return clients


def authenticate(clients: dict[str, bytes], header: str | None) -> str | None:
    match = TOKEN_RE.match(header or "")
    if not match:
        return None
    digest = hashlib.sha256(match.group(1).encode()).digest()
    found = None
    for name, expected in clients.items():
        if hmac.compare_digest(digest, expected) and found is None:
            found = name
    return found


def _object(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOLS = [
    {"name": "agent_task_submit",
     "description": "Queue one agent task. request: {instructions, input?, tools[], resultSchema, validator?, sources?, "
                    "tier: author|review, budgetUsd, maxRounds?}. The same idempotencyKey returns the same task.",
     "inputSchema": _object({"idempotencyKey": {"type": "string", "maxLength": 180},
                             "request": {"type": "object"}}, ["idempotencyKey", "request"])},
    {"name": "agent_task_get",
     "description": "Status of a task; when done: result, validation, sources (S1..), usage, rejections.",
     "inputSchema": _object({"taskId": {"type": "string", "maxLength": 20}}, ["taskId"])},
    {"name": "agent_task_cancel",
     "description": "Cancel a queued or running task.",
     "inputSchema": _object({"taskId": {"type": "string", "maxLength": 20}}, ["taskId"])},
]


class Rejected(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _task_id(arguments: dict) -> str:
    task_id = str(arguments.get("taskId") or "")
    if not task_id.isdigit():
        raise Rejected("taskId must be numeric")
    return task_id


async def call(client: str, name: str, arguments: dict) -> dict:
    if not isinstance(arguments, dict):
        raise Rejected("arguments must be an object")
    if name == "agent_task_submit":
        key = arguments.get("idempotencyKey")
        if not isinstance(key, str) or not KEY_RE.match(key):
            raise Rejected("idempotencyKey: 1-180 of A-Z a-z 0-9 : _ . -")
        try:
            request = runner.validate_request(arguments.get("request"))
        except Exception as exc:  # noqa: BLE001 - schema errors are the caller's to fix
            raise Rejected(str(exc)[:500]) from exc
        try:
            return await asyncio.to_thread(store.submit, client, key, request)
        except store.IdempotencyConflict as exc:
            raise Rejected(str(exc), 409) from exc
    if name in ("agent_task_get", "agent_task_cancel"):
        task_id = _task_id(arguments)
        fn = store.get if name == "agent_task_get" else store.cancel
        task = await asyncio.to_thread(fn, client, task_id)
        if task is None:
            raise Rejected(f"task {task_id} not found", 404)
        return task
    raise Rejected(f"unknown tool: {name}", 404)


def worker_router(clients: dict[str, bytes] | None = None) -> APIRouter:
    clients = parse_clients(os.getenv("WORKER_MCP_CLIENTS")) if clients is None else clients
    router = APIRouter()

    def rpc_error(request_id, code, message, status=200):
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
                            status_code=status)

    @router.post("/worker/mcp")
    async def mcp(request: Request):
        client = authenticate(clients, request.headers.get("authorization"))
        if client is None:
            return rpc_error(None, -32001, "unauthorized", 401)
        try:
            message = json.loads(await request.body())
        except ValueError:
            return rpc_error(None, -32700, "parse error", 400)
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return rpc_error(None, -32600, "invalid request", 400)
        request_id, method, params = message.get("id"), message["method"], message.get("params") or {}
        if request_id is None:
            return Response(status_code=202)
        if method == "initialize":
            requested = params.get("protocolVersion")
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "leninbot-worker", "version": "1.0.0"},
                "instructions": "Queue research/writing/review tasks for leninbot's agents and poll their results."}})
        if method == "ping":
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {}})
        if method == "tools/list":
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
        if method != "tools/call":
            return rpc_error(request_id, -32601, f"method not found: {method}")
        name = params.get("name")
        try:
            value = await call(client, name, params.get("arguments") or {})
            logger.info("worker mcp %s %s ok", client, name)
            content = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}], "structuredContent": value}
        except Rejected as exc:
            logger.info("worker mcp %s %s rejected: %s", client, name, exc)
            problem = {"error": str(exc), "status": exc.status}
            content = {"content": [{"type": "text", "text": json.dumps(problem)}], "structuredContent": problem, "isError": True}
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": content})

    return router


def create_app(clients: dict[str, bytes] | None = None) -> FastAPI:
    """Standalone app around the router, for tests."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(worker_router(clients))
    return app
