"""Gateway-aware wrappers for SDK clients owned by third-party libraries."""

from __future__ import annotations

import json
import time

from llm.gateway import check_llm_call, record_llm_call
from llm.tool_loop_common import estimate_text_tokens


_AUDIT_OWNER_KWARG = "_leninbot_audit_owner"
_AUDIT_OWNER_SENTINEL = object()


def with_audit_owner(client, kwargs: dict, owner: str) -> dict:
    """Mark a request whose usage is recorded by a higher-level seam.

    Only LeninBot's audited wrappers understand the private marker.  Raw SDK
    clients receive the original kwargs unchanged, while a wrapper removes the
    marker before forwarding the request and skips its own policy/audit event.
    Request preparation (caller headers and DeepSeek thinking defaults) still
    runs, so loop-owned calls retain their transport contract without counting
    the same response once as ``external_sdk`` and again as ``loop``.
    """
    if not getattr(client, "_leninbot_audit_wrapper", False):
        return kwargs
    return {**kwargs, _AUDIT_OWNER_KWARG: (_AUDIT_OWNER_SENTINEL, owner)}


def _pop_audit_owner(kwargs: dict) -> str | None:
    marker = kwargs.pop(_AUDIT_OWNER_KWARG, None)
    if (
        isinstance(marker, tuple)
        and len(marker) == 2
        and marker[0] is _AUDIT_OWNER_SENTINEL
    ):
        return str(marker[1])
    return None


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
    def __init__(self, messages, caller: str, provider: str, thinking_off: bool):
        self._messages = messages
        self._caller = caller
        self._provider = provider
        self._thinking_off = thinking_off

    def __getattr__(self, name):
        return getattr(self._messages, name)

    def _prepare(self, kwargs: dict) -> None:
        if self._thinking_off:
            kwargs.setdefault("thinking", {"type": "disabled"})
        # The proxy records who called it from this header. Without it every
        # request from a directly-held client lands in the ledger as one
        # anonymous heap.
        headers = dict(kwargs.get("extra_headers") or {})
        headers.setdefault("x-llm-caller", self._caller)
        kwargs["extra_headers"] = headers

    async def create(self, *args, **kwargs):
        model = _model_arg(args, kwargs)
        audit_owner = _pop_audit_owner(kwargs)
        # DeepSeek V4 turns thinking on when the request says nothing, and the
        # reasoning shares max_tokens with the reply: a call that only wants
        # text can spend the whole budget deliberating and return an empty text
        # block. Every caller that wants reasoning passes it (web_chat sends
        # disabled, a2a_handler sends _get_deepseek_tool_thinking_params), so
        # silence here means "not asked for" rather than "leave it to the
        # provider".
        self._prepare(kwargs)
        if audit_owner is None:
            check_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model,
            )
        started = time.monotonic()
        try:
            response = await self._messages.create(*args, **kwargs)
        except Exception as exc:
            # A failed attempt has no higher-level usage row to duplicate.
            record_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model, label="messages.create",
                status="error", error_excerpt=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
                estimate_cost=False,
            )
            raise
        if audit_owner is None:
            record_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model, label="messages.create",
                latency_ms=int((time.monotonic() - started) * 1000),
                **_anthropic_usage(response),
            )
        return response

    def stream(self, *args, **kwargs):
        model = _model_arg(args, kwargs)
        audit_owner = _pop_audit_owner(kwargs)
        self._prepare(kwargs)
        if audit_owner is None:
            check_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model,
            )
        # Returns the SDK's own async context manager untouched: the caller
        # consumes the stream and records its usage (llm/claude_loop does).
        return self._messages.stream(*args, **kwargs)


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

    def __init__(self, client, *, caller: str, provider: str, thinking_off: bool = False):
        self._leninbot_audit_wrapper = True
        self._client = client
        self.messages = _AuditedAsyncMessages(client.messages, caller, provider, thinking_off)

    def __getattr__(self, name):
        return getattr(self._client, name)


def _openai_usage(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    if details and not cached:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "tokens_in": getattr(usage, "prompt_tokens", 0) or 0,
        "tokens_out": getattr(usage, "completion_tokens", 0) or 0,
        "cache_read": cached,
        "cache_create": getattr(details, "cache_write_tokens", 0) or 0,
    }


class _AuditedOpenAIEndpoint:
    """One `create`-shaped OpenAI endpoint (chat.completions, embeddings, …)."""

    def __init__(self, endpoint, caller: str, provider: str, label: str):
        self._endpoint = endpoint
        self._caller = caller
        self._provider = provider
        self._label = label

    def __getattr__(self, name):
        return getattr(self._endpoint, name)

    async def create(self, *args, **kwargs):
        model = _model_arg(args, kwargs)
        audit_owner = _pop_audit_owner(kwargs)
        headers = dict(kwargs.get("extra_headers") or {})
        headers.setdefault("x-llm-caller", self._caller)
        kwargs["extra_headers"] = headers
        if audit_owner is None:
            check_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model,
            )
        started = time.monotonic()
        try:
            response = await self._endpoint.create(*args, **kwargs)
        except Exception as exc:
            # A failed attempt has no higher-level usage row to duplicate.
            record_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model, label=self._label,
                status="error", error_excerpt=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
                estimate_cost=False,
            )
            raise
        # A streamed response is an iterator with no usage until it is drained;
        # the caller that drains it owns the accounting.
        usage = ({} if kwargs.get("stream") else _openai_usage(response))
        if audit_owner is None:
            record_llm_call(
                surface="external_sdk", caller=self._caller,
                provider=self._provider, model=model, label=self._label,
                latency_ms=int((time.monotonic() - started) * 1000),
                estimate_cost=not kwargs.get("stream"), **usage,
            )
        return response


class _AuditedChat:
    def __init__(self, chat, caller: str, provider: str):
        self._chat = chat
        self.completions = _AuditedOpenAIEndpoint(
            chat.completions, caller, provider, "chat.completions.create")

    def __getattr__(self, name):
        return getattr(self._chat, name)


class AuditedAsyncOpenAI:
    """Transparent async OpenAI-compatible client that names itself.

    Same reason as AuditedAsyncAnthropic: bot_config hands these out by name and
    whatever imports one would otherwise reach the proxy anonymously.
    """

    def __init__(self, client, *, caller: str, provider: str):
        self._leninbot_audit_wrapper = True
        self._client = client
        self.chat = _AuditedChat(client.chat, caller, provider)
        self.embeddings = _AuditedOpenAIEndpoint(
            client.embeddings, caller, provider, "embeddings.create")

    def __getattr__(self, name):
        return getattr(self._client, name)
