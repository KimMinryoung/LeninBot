"""embedding_server.py — Standalone BGE-M3 embedding + reranker service.

Runs as an independent systemd service so that telegram_bot, api, and
other processes share a single model instance via HTTP.  Model stays
loaded across consumer restarts.

Endpoints:
    POST /embed_query   {"text": "..."} → {"embedding": [...]}
    POST /embed_docs    {"texts": [...]} → {"embeddings": [[...], ...]}
    POST /rerank        {"query": "...", "documents": [...], "top_k": 5}
                        → {"results": [{"index": 0, "score": 0.95}, ...]}
    GET  /health        → {"status": "ok", "model": "BAAI/bge-m3", ...}
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

_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
_RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "auto").lower()
_RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", _EMBEDDING_DEVICE).lower()
_OFFLINE = os.environ.get("EMBEDDING_OFFLINE", "1").lower() not in {"0", "false", "no"}
_PRELOAD_RERANKER = os.environ.get("EMBEDDING_PRELOAD_RERANKER", "0").lower() in {
    "1",
    "true",
    "yes",
}

if _OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as e:
        logger.info("CUDA auto-detect unavailable: %s", e)
    return "cpu"


# ── Model singletons ─────────────────────────────────────────────────
_embeddings = None
_reranker = None


def _load_model():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    device = _resolve_device(_EMBEDDING_DEVICE)
    t0 = time.time()
    _embeddings = HuggingFaceEmbeddings(
        model_name=_MODEL_NAME,
        model_kwargs={"device": device, "local_files_only": _OFFLINE},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("%s loaded on %s in %.1fs", _MODEL_NAME, device, time.time() - t0)
    return _embeddings


def _load_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    from sentence_transformers import CrossEncoder

    device = _resolve_device(_RERANKER_DEVICE)
    t0 = time.time()
    _reranker = CrossEncoder(_RERANKER_MODEL, device=device, local_files_only=_OFFLINE)
    logger.info("%s loaded on %s in %.1fs", _RERANKER_MODEL, device, time.time() - t0)
    return _reranker


# ── FastAPI app ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    if _PRELOAD_RERANKER:
        _load_reranker()
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


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 5


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]


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


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if not req.query.strip():
        raise HTTPException(400, "query is empty")
    if not req.documents:
        raise HTTPException(400, "documents is empty")
    try:
        reranker = _load_reranker()
    except Exception as e:
        logger.warning("reranker unavailable: %s", e)
        raise HTTPException(503, "reranker unavailable") from e
    pairs = [(req.query, doc) for doc in req.documents]
    scores = reranker.predict(pairs)
    indexed = [(i, float(s)) for i, s in enumerate(scores)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    top = indexed[: req.top_k]
    return RerankResponse(results=[RerankResult(index=i, score=s) for i, s in top])


@app.get("/health")
async def health():
    ok = _embeddings is not None
    return {
        "status": "ok" if ok else "loading",
        "model": _MODEL_NAME,
        "device": _resolve_device(_EMBEDDING_DEVICE),
        "offline": _OFFLINE,
        "reranker": _RERANKER_MODEL if _reranker is not None else "not_loaded",
        "dim": 1024,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("EMBEDDING_PORT", "8100"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
