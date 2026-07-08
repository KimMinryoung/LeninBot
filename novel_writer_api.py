import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_routes.writer import router as writer_router

_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger().setLevel(_LOG_LEVEL)

_DEFAULT_CORS_ORIGINS = "https://cyber-lenin.com,http://localhost:3000"


def _parse_cors_origins() -> list[str]:
    raw = (
        os.getenv("WRITER_CORS_ORIGINS")
        or os.getenv("WEBCHAT_CORS_ORIGINS")
        or os.getenv("CORS_ALLOW_ORIGINS")
        or _DEFAULT_CORS_ORIGINS
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or [item.strip() for item in _DEFAULT_CORS_ORIGINS.split(",")]


app = FastAPI(title="Cyber-Lenin Novel Writer API")
app.include_router(writer_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "novel-writer-api"}


if __name__ == "__main__":
    print("Novel writer API server starting on 127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
