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

import logging
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from secrets_loader import get_secret

logger = logging.getLogger("llm_proxy")

# auth styles: "x-api-key" (Anthropic protocol), "bearer" (OpenAI protocol),
# "x-goog-api-key" (Gemini). DeepSeek serves both protocol families from one
# host, so both headers are injected; the endpoint reads whichever it wants.
PROVIDERS: dict[str, dict] = {
    "anthropic": {"upstream": "https://api.anthropic.com", "secret": "ANTHROPIC_API_KEY",
                  "auth": ("x-api-key",)},
    "deepseek": {"upstream": "https://api.deepseek.com", "secret": "DEEPSEEK_API_KEY",
                 "auth": ("x-api-key", "bearer")},
    "moonshot": {"upstream": "https://api.moonshot.ai", "secret": "MOONSHOT_API_KEY",
                 "auth": ("bearer",)},
    "openai": {"upstream": "https://api.openai.com", "secret": "OPENAI_API_KEY",
               "auth": ("bearer",)},
    "gemini": {"upstream": "https://generativelanguage.googleapis.com", "secret": "GEMINI_API_KEY",
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
        name for name, cfg in PROVIDERS.items()
        if not (get_secret(cfg["secret"], "") or "").strip()
    ]
    return {"status": "ok", "providers_without_key": missing}


@app.api_route("/{provider}/{path:path}", methods=["GET", "POST"])
async def proxy(provider: str, path: str, request: Request):
    cfg = PROVIDERS.get(provider)
    if cfg is None:
        return JSONResponse({"error": f"unknown provider {provider!r}"}, status_code=404)
    key = (get_secret(cfg["secret"], "") or "").strip()
    if not key:
        return JSONResponse(
            {"error": f"no credential for provider {provider!r}"}, status_code=503,
        )

    # Gemini SDKs may carry the key as a query parameter; drop it.
    params = [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
    body = await request.body()
    headers = build_forward_headers(dict(request.headers), cfg, key)

    started = time.monotonic()
    upstream_req = _http_client().build_request(
        request.method, f"{cfg['upstream']}/{path}",
        headers=headers, params=params, content=body,
    )
    try:
        upstream = await _http_client().send(upstream_req, stream=True)
    except httpx.HTTPError as e:
        logger.warning("proxy %s/%s upstream error: %s", provider, path, e)
        return JSONResponse(
            {"error": f"upstream unreachable: {e.__class__.__name__}"}, status_code=502,
        )

    logger.info(
        "proxy %s /%s → %d (headers in %.0fms, req %dB)",
        provider, path, upstream.status_code,
        (time.monotonic() - started) * 1000, len(body),
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
