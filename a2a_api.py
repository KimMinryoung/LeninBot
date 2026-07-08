import json
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger().setLevel(_LOG_LEVEL)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

_DEFAULT_CORS_ORIGINS = "https://cyber-lenin.com,http://localhost:3000"
AGENT_CARD_DIR = Path(__file__).parent / "research"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _a2a_enabled() -> bool:
    return _env_flag("A2A_ENABLED", default=True)


def _parse_cors_origins() -> list[str]:
    raw = (
        os.getenv("A2A_CORS_ORIGINS")
        or os.getenv("WEBCHAT_CORS_ORIGINS")
        or os.getenv("CORS_ALLOW_ORIGINS")
        or _DEFAULT_CORS_ORIGINS
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or [item.strip() for item in _DEFAULT_CORS_ORIGINS.split(",")]


app = FastAPI(title="Cyber-Lenin A2A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
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
        return Response(
            content=json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}),
            media_type="application/json",
            status_code=400,
        )
    if not _a2a_enabled():
        return Response(
            content=json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": "A2A is temporarily disabled"},
                "id": body.get("id"),
            }, ensure_ascii=False),
            media_type="application/json",
            status_code=503,
        )
    from a2a_handler import handle_a2a_message

    result = await handle_a2a_message(body)
    status_code = 200 if "result" in result else 400
    return Response(content=json.dumps(result, ensure_ascii=False), media_type="application/json", status_code=status_code)


if __name__ == "__main__":
    print("A2A API server starting on 127.0.0.1:8003")
    uvicorn.run(app, host="127.0.0.1", port=8003)
