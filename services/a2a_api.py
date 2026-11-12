"""Public A2A API service entrypoint."""

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from services.api_common import parse_cors_origins, setup_service_logging

setup_service_logging(quiet_neo4j=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_CARD_DIR = PROJECT_ROOT / "research"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _a2a_enabled() -> bool:
    return _env_flag("A2A_ENABLED", default=True)


# Each SendMessage runs a paid tool loop and the endpoint is public, so cap
# both per-client rate and process-wide concurrency.
_A2A_RATE_LIMIT = max(1, int(os.getenv("A2A_RATE_LIMIT", "5") or "5"))
_A2A_RATE_WINDOW_SECONDS = max(60, int(os.getenv("A2A_RATE_WINDOW_SECONDS", "3600") or "3600"))
_A2A_MAX_CONCURRENT = max(1, int(os.getenv("A2A_MAX_CONCURRENT", "2") or "2"))
_a2a_hits: defaultdict[str, deque[float]] = defaultdict(deque)
_a2a_semaphore = asyncio.Semaphore(_A2A_MAX_CONCURRENT)


def _a2a_client_key(request: Request) -> str:
    # Only nginx (via the frontend chat gateway) reaches this port; nginx
    # overwrites X-Real-IP with the connecting address.
    return (request.headers.get("x-real-ip") or "").strip() or (request.client.host if request.client else "unknown")


def _a2a_rate_ok(key: str) -> bool:
    now = time.monotonic()
    hits = _a2a_hits[key]
    while hits and hits[0] < now - _A2A_RATE_WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= _A2A_RATE_LIMIT:
        return False
    hits.append(now)
    if len(_a2a_hits) > 5000:
        for stale in [k for k, v in _a2a_hits.items() if not v][:1000]:
            _a2a_hits.pop(stale, None)
    return True


def _rpc_error(rpc_id, code: int, message: str, status_code: int) -> Response:
    return Response(
        content=json.dumps({"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": rpc_id}, ensure_ascii=False),
        media_type="application/json",
        status_code=status_code,
    )


app = FastAPI(title="Cyber-Lenin A2A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins("A2A_CORS_ORIGINS"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "service": "leninbot-a2a-api"}


async def _serve_agent_card():
    if not _a2a_enabled():
        raise HTTPException(status_code=503, detail="A2A is temporarily disabled")
    filepath = AGENT_CARD_DIR / "cyber_lenin_a2a_agent_card.json"
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Agent card not found")
    return Response(content=filepath.read_text(encoding="utf-8"), media_type="application/json; charset=utf-8")


@app.get("/.well-known/agent-card.json")
async def a2a_agent_card_v1():
    return await _serve_agent_card()


@app.post("/a2a")
async def a2a_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _rpc_error(None, -32700, "Parse error", 400)
    if not isinstance(body, dict):
        return _rpc_error(None, -32600, "Invalid Request: body must be a JSON object", 400)
    rpc_id = body.get("id")
    if not _a2a_enabled():
        return _rpc_error(rpc_id, -32000, "A2A is temporarily disabled", 503)
    if not _a2a_rate_ok(_a2a_client_key(request)):
        return _rpc_error(rpc_id, -32000, "Rate limit exceeded; try again later", 429)
    if _a2a_semaphore.locked():
        return _rpc_error(rpc_id, -32000, "Server busy; try again later", 503)
    from services.a2a_handler import handle_a2a_message

    async with _a2a_semaphore:
        result = await handle_a2a_message(body)
    status_code = 200 if "result" in result else 400
    return Response(content=json.dumps(result, ensure_ascii=False), media_type="application/json", status_code=status_code)


if __name__ == "__main__":
    print("A2A API server starting on 127.0.0.1:8003")
    uvicorn.run(app, host="127.0.0.1", port=8003)
