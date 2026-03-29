"""embedding_server.py — Standalone BGE-M3 embedding service.

Runs as an independent systemd service so that telegram_bot, api, and
other processes share a single model instance via HTTP.  Model stays
loaded across consumer restarts.

Endpoints:
    POST /embed_query   {"text": "..."} → {"embedding": [...]}
    POST /embed_docs    {"texts": [...]} → {"embeddings": [[...], ...]}
    GET  /health        → {"status": "ok", "model": "BAAI/bge-m3", "dim": 1024}
"""

import logging
import os
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("embedding_server")

# ── Model singleton ──────────────────────────────────────────────────
_embeddings = None


def _load_model():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    t0 = time.time()
    _embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("BGE-M3 loaded in %.1fs", time.time() - t0)
    return _embeddings


# ── FastAPI app ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="LeninBot Embedding Service", lifespan=lifespan)


# ── Request / Response schemas ───────────────────────────────────────
class EmbedQueryRequest(BaseModel):
    text: str


class EmbedQueryResponse(BaseModel):
    embedding: list[float]


class EmbedDocsRequest(BaseModel):
    texts: list[str]


class EmbedDocsResponse(BaseModel):
    embeddings: list[list[float]]


# ── Endpoints ────────────────────────────────────────────────────────
@app.post("/embed_query", response_model=EmbedQueryResponse)
async def embed_query(req: EmbedQueryRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    vec = _embeddings.embed_query(req.text)
    return EmbedQueryResponse(embedding=vec)


@app.post("/embed_docs", response_model=EmbedDocsResponse)
async def embed_docs(req: EmbedDocsRequest):
    if not req.texts:
        raise HTTPException(400, "texts is empty")
    vecs = _embeddings.embed_documents(req.texts)
    return EmbedDocsResponse(embeddings=vecs)


@app.get("/health")
async def health():
    ok = _embeddings is not None
    return {
        "status": "ok" if ok else "loading",
        "model": "BAAI/bge-m3",
        "dim": 1024,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("EMBEDDING_PORT", "8100"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
