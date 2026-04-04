"""embedding_client.py — HTTP client for the embedding server.

Drop-in replacement for HuggingFaceEmbeddings: exposes embed_query()
and embed_documents() with the same signatures.

On transient failures (server restarting, model loading), retries for
up to ~15 seconds before falling back to local model loading.
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

_EMBEDDING_URL = os.environ.get(
    "EMBEDDING_URL", "http://127.0.0.1:8100"
)
_TIMEOUT = 30  # seconds per HTTP request
_RETRY_TOTAL_SEC = 15  # total retry window (covers ~6s model load + margin)
_RETRY_INTERVAL = 2  # seconds between retries


class EmbeddingClient:
    """HTTP client that mirrors HuggingFaceEmbeddings interface."""

    def __init__(self, base_url: str = _EMBEDDING_URL):
        self._base_url = base_url.rstrip("/")
        self._fallback = None

    def _get_fallback(self):
        if self._fallback is None:
            logger.warning(
                "[EmbeddingClient] server unreachable after retries, "
                "loading BGE-M3 locally (slow)"
            )
            from langchain_huggingface import HuggingFaceEmbeddings
            self._fallback = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._fallback

    def _post_with_retry(self, path: str, json_body: dict) -> dict:
        """POST with retry on connection errors (server restarting)."""
        deadline = time.monotonic() + _RETRY_TOTAL_SEC
        last_err = None
        attempt = 0

        while True:
            attempt += 1
            try:
                resp = requests.post(
                    f"{self._base_url}{path}",
                    json=json_body,
                    timeout=_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError as e:
                # Server is down / restarting — retry
                last_err = e
                if time.monotonic() >= deadline:
                    break
                if attempt == 1:
                    logger.info(
                        "[EmbeddingClient] server unavailable, "
                        "retrying for %ds...", _RETRY_TOTAL_SEC,
                    )
                time.sleep(_RETRY_INTERVAL)
            except Exception as e:
                # Non-connection error (bad request, etc.) — don't retry
                raise

        raise last_err

    def embed_query(self, text: str) -> list[float]:
        try:
            data = self._post_with_retry("/embed_query", {"text": text})
            return data["embedding"]
        except Exception as e:
            logger.debug("[EmbeddingClient] giving up: %s", e)
            return self._get_fallback().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            data = self._post_with_retry("/embed_docs", {"texts": texts})
            return data["embeddings"]
        except Exception as e:
            logger.debug("[EmbeddingClient] giving up: %s", e)
            return self._get_fallback().embed_documents(texts)

    def rerank(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query. Returns [(index, score), ...]."""
        try:
            data = self._post_with_retry(
                "/rerank",
                {"query": query, "documents": documents, "top_k": top_k},
            )
            return [(r["index"], r["score"]) for r in data["results"]]
        except Exception:
            logger.warning("[EmbeddingClient] rerank unavailable, returning original order")
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]


# Module-level singleton
_client = None


def get_embedding_client() -> EmbeddingClient:
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client
