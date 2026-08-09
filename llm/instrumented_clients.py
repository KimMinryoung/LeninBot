"""Gateway-aware wrappers for SDK clients owned by third-party libraries."""

from __future__ import annotations

import json
import time

from llm.gateway import check_llm_call, record_llm_call
from llm.tool_loop_common import estimate_text_tokens


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


def _anthropic_usage(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "tokens_in": getattr(usage, "input_tokens", 0) or 0,
        "tokens_out": getattr(usage, "output_tokens", 0) or 0,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_create": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


class _AuditedAsyncMessages:
    def __init__(self, messages, caller: str, provider: str):
        self._messages = messages
        self._caller = caller
        self._provider = provider

    def __getattr__(self, name):
        return getattr(self._messages, name)

    async def create(self, *args, **kwargs):
        model = _model_arg(args, kwargs)
        # DeepSeek V4 turns thinking on when the request says nothing, and the
        # reasoning shares max_tokens with the reply: a call that only wants
        # text can spend the whole budget deliberating and return an empty text
        # block. Every caller that wants reasoning passes it (web_chat sends
        # disabled, a2a_handler sends _get_deepseek_tool_thinking_params), so
        # silence here means "not asked for" rather than "leave it to the
        # provider".
        kwargs.setdefault("thinking", {"type": "disabled"})
        # The proxy records who called it from this header. Without it every
        # request from a directly-held client lands in the ledger as one
        # anonymous heap.
        headers = dict(kwargs.get("extra_headers") or {})
        headers.setdefault("x-llm-caller", self._caller)
        kwargs["extra_headers"] = headers
        check_llm_call(
            surface="external_sdk", caller=self._caller,
            provider=self._provider, model=model,
        )
        started = time.monotonic()
        try:
            response = await self._messages.create(*args, **kwargs)
        except Exception as exc:
            record_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model, label="messages.create",
                status="error", error_excerpt=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
                estimate_cost=False,
            )
            raise
        record_llm_call(
            surface="external_sdk", caller=self._caller,
            provider=self._provider, model=model, label="messages.create",
            latency_ms=int((time.monotonic() - started) * 1000),
            **_anthropic_usage(response),
        )
        return response


class AuditedAsyncAnthropic:
    """Transparent async Anthropic-compatible client that reports its spend.

    The gateway is applied by call sites, not by the SDK object: llm/call_registry
    and llm/claude_loop wrap every call they make, but `bot_config` handed out the
    raw `anthropic.AsyncAnthropic` under a private-looking name, and anything that
    imported it got an unmetered path. Two maintenance scripts did, and the em dash
    sweep of 2026-08-09 made 576 requests that appear nowhere in the ledger.

    Wrapping the client rather than fixing the two scripts is the point: the next
    script gets metered without knowing the gateway exists.
    """

    def __init__(self, client, *, caller: str, provider: str):
        self._client = client
        self.messages = _AuditedAsyncMessages(client.messages, caller, provider)

    def __getattr__(self, name):
        return getattr(self._client, name)
