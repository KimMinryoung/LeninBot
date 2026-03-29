"""embedding_client.py — HTTP client for the embedding server.

Drop-in replacement for HuggingFaceEmbeddings: exposes embed_query()
and embed_documents() with the same signatures.

Falls back to direct HuggingFaceEmbeddings loading if the server is
unreachable (e.g. during development or server startup race).
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

_EMBEDDING_URL = os.environ.get(
    "EMBEDDING_URL", "http://127.0.0.1:8100"
)
_TIMEOUT = 30  # seconds per request


class EmbeddingClient:
    """HTTP client that mirrors HuggingFaceEmbeddings interface."""

    def __init__(self, base_url: str = _EMBEDDING_URL):
        self._base_url = base_url.rstrip("/")
        self._fallback = None

    def _get_fallback(self):
        if self._fallback is None:
            logger.warning(
                "[EmbeddingClient] server unreachable, loading BGE-M3 locally (slow)"
            )
            from langchain_huggingface import HuggingFaceEmbeddings
            self._fallback = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._fallback

    def embed_query(self, text: str) -> list[float]:
        try:
            resp = requests.post(
                f"{self._base_url}/embed_query",
                json={"text": text},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            logger.debug("[EmbeddingClient] server error: %s", e)
            return self._get_fallback().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = requests.post(
                f"{self._base_url}/embed_docs",
                json={"texts": texts},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except Exception as e:
            logger.debug("[EmbeddingClient] server error: %s", e)
            return self._get_fallback().embed_documents(texts)


# Module-level singleton
_client = None


def get_embedding_client() -> EmbeddingClient:
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client
