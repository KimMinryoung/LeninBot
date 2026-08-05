"""Gateway-aware wrappers for SDK clients owned by third-party libraries."""

from __future__ import annotations

import json
import time

from llm.gateway import check_llm_call, record_llm_call
from tool_loop_common import estimate_text_tokens


def _gemini_usage(response) -> dict[str, int]:
    meta = getattr(response, "usage_metadata", None)
    return {
        "tokens_in": getattr(meta, "prompt_token_count", 0) or 0,
        "tokens_out": (
            (getattr(meta, "candidates_token_count", 0) or 0)
            + (getattr(meta, "thoughts_token_count", 0) or 0)
        ),
        "cache_read": getattr(meta, "cached_content_token_count", 0) or 0,
    }


def _model_arg(args: tuple, kwargs: dict) -> str | None:
    model = kwargs.get("model")
    if not model and args:
        model = args[0]
    return str(model) if model else None


class _AuditedAsyncGeminiModels:
    def __init__(self, models, caller: str):
        self._models = models
        self._caller = caller

    def __getattr__(self, name):
        return getattr(self._models, name)

    async def generate_content(self, *args, **kwargs):
        model = _model_arg(args, kwargs)
        check_llm_call(
            surface="external_sdk", caller=self._caller,
            provider="gemini", model=model,
        )
        started = time.monotonic()
        try:
            response = await self._models.generate_content(*args, **kwargs)
        except Exception as exc:
            record_llm_call(
                surface="external_sdk", caller=self._caller,
                provider="gemini", model=model, label="generate_content",
                status="error", error_excerpt=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
                token_semantics="gemini", estimate_cost=False,
            )
            raise
        record_llm_call(
            surface="external_sdk", caller=self._caller,
            provider="gemini", model=model, label="generate_content",
            latency_ms=int((time.monotonic() - started) * 1000),
            token_semantics="gemini", **_gemini_usage(response),
        )
        return response

    async def embed_content(self, *args, **kwargs):
        model = _model_arg(args, kwargs)
        contents = kwargs.get("contents")
        if contents is None and len(args) > 1:
            contents = args[1]
        # Gemini embeddings do not expose billed token usage in this SDK.
        # Record a clearly labelled conservative text estimate instead of
        # leaving KG embedding spend completely invisible.
        try:
            serialized = json.dumps(contents, ensure_ascii=False, default=str)
        except Exception:
            serialized = str(contents or "")
        tokens_in = estimate_text_tokens(serialized)
        check_llm_call(
            surface="external_sdk", caller=self._caller,
            provider="gemini", model=model,
        )
        started = time.monotonic()
        try:
            response = await self._models.embed_content(*args, **kwargs)
        except Exception as exc:
            record_llm_call(
                surface="external_sdk", caller=self._caller,
                provider="gemini", model=model, label="embed_content:estimated",
                tokens_in=tokens_in, status="error", error_excerpt=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
                token_semantics="gemini", estimate_cost=False,
            )
            raise
        record_llm_call(
            surface="external_sdk", caller=self._caller,
            provider="gemini", model=model, label="embed_content:estimated",
            tokens_in=tokens_in,
            latency_ms=int((time.monotonic() - started) * 1000),
            token_semantics="gemini",
        )
        return response


class _AuditedGeminiAio:
    def __init__(self, aio, caller: str):
        self._aio = aio
        self.models = _AuditedAsyncGeminiModels(aio.models, caller)

    def __getattr__(self, name):
        return getattr(self._aio, name)


class AuditedGenAIClient:
    """Transparent Google GenAI client wrapper for Graphiti async calls."""

    def __init__(self, client, *, caller: str):
        self._client = client
        self.aio = _AuditedGeminiAio(client.aio, caller)

    def __getattr__(self, name):
        return getattr(self._client, name)
