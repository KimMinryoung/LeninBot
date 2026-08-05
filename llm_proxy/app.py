"""llm_proxy/app.py — key-injection passthrough proxy for LLM providers.

The enforcement half of the LLM gateway (llm/gateway.py is the observation/
policy half). Provider API keys are loaded ONLY by this service's systemd
credentials; every other service keeps a placeholder key and points its
client base_url at http://127.0.0.1:8110/<provider>. Once the provider keys
are removed from the other services' credential sets, bypassing the gateway
becomes physically impossible — code without a key cannot call a provider.

Deliberately a BYTE passthrough, not a translating router (the reason
LiteLLM proxy was rejected): the request body and the response stream are
forwarded untouched, so provider protocol details the adapters depend on —
SSE streaming, prompt-cache markers, thinking blocks, tool_use shapes —
cannot be altered by this hop. The only mutation is the auth header swap.

Routes:  /{provider}/{path}  →  {upstream}/{path}   (GET/POST)
Auth:    incoming x-api-key / authorization / x-goog-api-key are stripped
         and replaced with the provider's real key from credstore.
Binding: 127.0.0.1 only. Single-tenant VM; no local token layer.

Unit: systemd/leninbot-llm-proxy.service (Restart=always, RestartSec=2).
If this service is down every proxied LLM call fails; the loops' 3-attempt
transient retry absorbs restart blips.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from llm.gateway import evaluate_policy, record_llm_call

logger = logging.getLogger("llm_proxy")


def _credential(name: str) -> str:
    """Read a provider key ONLY from this unit's systemd credentials.

    Deliberately not secrets_loader.get_secret: that helper lets real env
    vars win over credstore, and the services' .env (which this unit also
    loads for DB access) carries placeholder values like
    OPENAI_API_KEY=via-llm-proxy for graphiti's env-built internal client.
    The key custodian must never let an env value shadow its own credstore.
    """
    cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if not cred_dir:
        return ""
    try:
        return (Path(cred_dir) / name.lower()).read_text(encoding="utf-8").strip()
    except OSError:
        return ""

# Route names → the provider names policy config uses (blocked_providers,
# daily_budget_per_provider). Routes not listed map to themselves.
POLICY_PROVIDER = {
    "anthropic": "claude",
    "moonshot": "kimi",
}

# auth styles: "x-api-key" (Anthropic protocol), "bearer" (OpenAI protocol),
# "x-goog-api-key" (Gemini). DeepSeek serves both protocol families from one
# host, so both headers are injected; the endpoint reads whichever it wants.
PROVIDERS: dict[str, dict] = {
    "anthropic": {"upstream": "https://api.anthropic.com", "secrets": ("ANTHROPIC_API_KEY",),
                  "auth": ("x-api-key",)},
    "deepseek": {"upstream": "https://api.deepseek.com", "secrets": ("DEEPSEEK_API_KEY",),
                 "auth": ("x-api-key", "bearer")},
    "moonshot": {"upstream": "https://api.moonshot.ai", "secrets": ("MOONSHOT_API_KEY",),
                 "auth": ("bearer",)},
    "openai": {"upstream": "https://api.openai.com", "secrets": ("OPENAI_API_KEY",),
               "auth": ("bearer",)},
    "gemini": {"upstream": "https://generativelanguage.googleapis.com", "secrets": ("GEMINI_API_KEY",),
               "auth": ("x-goog-api-key",)},
}

# Hop-by-hop plus everything we replace. Content-Length is recomputed by
# httpx for the request and dropped from the response (chunked streaming).
_STRIP_REQUEST = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "authorization", "x-api-key", "x-goog-api-key",
}
_STRIP_RESPONSE = {
    "content-length", "connection", "keep-alive", "transfer-encoding",
}

# Generous read timeout: writer generations stream for many minutes. Stall
# detection is the caller's job (the loops' idle guards), not this hop's.
_TIMEOUT = httpx.Timeout(connect=15.0, read=1200.0, write=120.0, pool=60.0)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_client: httpx.AsyncClient | None = None


def _http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


def model_from_body(body: bytes) -> str | None:
    """Best-effort model ID from a request body. The original bytes are
    forwarded untouched regardless — this is read-only inspection."""
    try:
        model = json.loads(body).get("model")
        return str(model) if model else None
    except Exception:
        return None


def model_from_request(provider: str, path: str, body: bytes) -> str | None:
    """Extract a model from either protocol's canonical location.

    OpenAI/Anthropic put it in JSON.  Gemini puts it in the URL, including
    percent-encoded ``:generateContent`` suffixes emitted by the SDK.  Policy
    must inspect both or Gemini ``blocked_models`` is silently bypassed.
    """
    model = model_from_body(body)
    if model:
        return model
    if provider == "gemini":
        decoded = unquote(path)
        match = re.search(r"(?:^|/)models/([^/:]+)(?::|/|$)", decoded)
        if match:
            return match.group(1)
    return None


def _provider_key(cfg: dict) -> str:
    for name in cfg["secrets"]:
        key = _credential(name)
        if key:
            return key
    return ""


def build_forward_headers(incoming: dict, provider_cfg: dict, key: str) -> dict:
    """Client headers minus hop-by-hop/auth, plus the provider's real auth."""
    headers = {
        k: v for k, v in incoming.items() if k.lower() not in _STRIP_REQUEST
    }
    for style in provider_cfg["auth"]:
        if style == "bearer":
            headers["authorization"] = f"Bearer {key}"
        else:
            headers[style] = key
    return headers


@app.get("/health")
async def health():
    missing = [
        name for name, cfg in PROVIDERS.items() if not _provider_key(cfg)
    ]
    payload = {
        "status": "ok" if not missing else "not_ready",
        "providers_without_key": missing,
    }
    return JSONResponse(payload, status_code=200 if not missing else 503)


@app.api_route("/{provider}/{path:path}", methods=["GET", "POST"])
async def proxy(provider: str, path: str, request: Request):
    cfg = PROVIDERS.get(provider)
    if cfg is None:
        return JSONResponse({"error": f"unknown provider {provider!r}"}, status_code=404)
    key = _provider_key(cfg)
    if not key:
        return JSONResponse(
            {"error": f"no credential for provider {provider!r}"}, status_code=503,
        )

    # Gemini SDKs may carry the key as a query parameter; drop it.
    params = [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
    body = await request.body()

    # Authoritative policy gate. The decision logic is shared with the
    # in-process seam (llm/gateway.evaluate_policy — single source); THIS
    # evaluation is the one a caller cannot skip, because the provider key
    # only exists on the far side of it.
    model = model_from_request(provider, path, body)
    policy_provider = POLICY_PROVIDER.get(provider, provider)
    reason, enforce = evaluate_policy(provider=policy_provider, model=model)
    if reason is not None:
        record_llm_call(
            surface="proxy", caller=request.headers.get("x-llm-caller"),
            provider=policy_provider, model=model,
            status="denied" if enforce else "would_deny", error_excerpt=reason,
        )
        if enforce:
            logger.warning("proxy DENIED %s /%s: %s", provider, path, reason)
            return JSONResponse(
                {"error": f"llm gateway policy: {reason}"}, status_code=403,
            )
        logger.warning("proxy would-deny (shadow) %s /%s: %s", provider, path, reason)

    headers = build_forward_headers(dict(request.headers), cfg, key)

    started = time.monotonic()
    upstream_req = _http_client().build_request(
        request.method, f"{cfg['upstream']}/{path}",
        headers=headers, params=params, content=body,
    )
    try:
        upstream = await _http_client().send(upstream_req, stream=True)
    except httpx.HTTPError as e:
        record_llm_call(
            surface="proxy", caller=request.headers.get("x-llm-caller"),
            provider=policy_provider, model=model, label=path[:200],
            status="error", error_excerpt=str(e),
            latency_ms=int((time.monotonic() - started) * 1000),
            estimate_cost=False,
        )
        logger.warning("proxy %s/%s upstream error: %s", provider, path, e)
        return JSONResponse(
            {"error": f"upstream unreachable: {e.__class__.__name__}"}, status_code=502,
        )

    logger.info(
        "proxy %s /%s → %d (headers in %.0fms, req %dB)",
        provider, path, upstream.status_code,
        (time.monotonic() - started) * 1000, len(body),
    )
    record_llm_call(
        surface="proxy", caller=request.headers.get("x-llm-caller"),
        provider=policy_provider, model=model, label=path[:200],
        status="ok" if 200 <= upstream.status_code < 400 else "error",
        error_excerpt=(None if upstream.status_code < 400
                       else f"upstream HTTP {upstream.status_code}"),
        latency_ms=int((time.monotonic() - started) * 1000),
        estimate_cost=False,
    )
    # aiter_raw: bytes exactly as the provider sent them (no decompression),
    # so the preserved content-encoding header stays truthful.
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers={
            k: v for k, v in upstream.headers.items()
            if k.lower() not in _STRIP_RESPONSE
        },
        background=BackgroundTask(upstream.aclose),
    )


@app.on_event("shutdown")
async def _shutdown():
    if _client is not None:
        await _client.aclose()
