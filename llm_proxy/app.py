"""llm_proxy/app.py — key-injection passthrough proxy for LLM providers.

The enforcement half of the LLM gateway (llm/gateway.py is the observation/
policy half). Provider API keys are loaded ONLY by this service's systemd
credentials; every other service keeps a placeholder key and points its
client base_url at http://127.0.0.1:8110/<provider>. Once the provider keys
are removed from the other services' credential sets, bypassing the gateway
becomes physically impossible — code without a key cannot call a provider.

The proxy preserves provider protocols and response streams. It rewrites only
the model in text-generation requests to the current ID for that tier, replaces
auth headers, and applies one guarded DeepSeek default: completion requests
with no thinking field get {"type": "disabled"} (see
apply_deepseek_thinking_default). Other request bodies pass through unchanged.

Routes:  /{provider}/{path}  →  {upstream}/{path}   (GET/POST)
         POST /audit/{llm|tool} — audit sink: the ONLY process that inserts
         into llm_audit_log / tool_audit_log (audit_sink.py; localhost-only,
         row/body caps, column whitelist). GET /audit/spend/today feeds the
         budget policy of client processes.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ops import audit_sink
from llm.gateway import evaluate_policy, record_llm_call
from llm.provider_registry import current_text_model

logger = logging.getLogger("llm_proxy")

# This process is the audit sink: its own gateway rows insert directly
# instead of being POSTed back to itself.
audit_sink.set_local_sink(True)


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
    # Jev (System One decisions) routes, 2026-09-19. OpenRouter serves Jev at
    # /api/alpha/decisions while the direct TypeSafe API is waitlisted. Both
    # credentials are mounted only where they exist (drop-in emitted by
    # scripts/migrate_secrets_to_credstore.py), so both are "optional": a
    # missing key makes that route answer 503 and decide() fall back, instead
    # of /health — which gates every consumer's startup — reporting not_ready.
    "openrouter": {"upstream": "https://openrouter.ai", "secret": "OPENROUTER_API_KEY",
                   "auth": ("bearer",), "optional": True},
    "typesafe": {"upstream": "https://api.typesafe.ai", "secret": "TYPESAFE_API_KEY",
                 "auth": ("bearer",), "optional": True},
}

# Billing reads are deliberately NOT added as passthrough providers.  The
# optional admin credentials can read organization-wide financial data and in
# some providers carry broader administrative authority than an inference key.
# They are therefore usable only through the fixed read-only endpoints below;
# no caller can choose an arbitrary upstream admin path.
BILLING_PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "kind": "balance",
        "upstream": "https://api.deepseek.com/user/balance",
        "secret": "DEEPSEEK_API_KEY",
        "auth": "bearer",
    },
    "kimi": {
        "kind": "balance",
        "upstream": "https://api.moonshot.ai/v1/users/me/balance",
        "secret": "MOONSHOT_API_KEY",
        "auth": "bearer",
    },
    "openai": {
        "kind": "cost",
        "upstream": "https://api.openai.com/v1/organization/costs",
        "secret": "OPENAI_ADMIN_KEY",
        "auth": "bearer",
    },
    "claude": {
        "kind": "cost",
        "upstream": "https://api.anthropic.com/v1/organizations/cost_report",
        "secret": "ANTHROPIC_ADMIN_KEY",
        "auth": "anthropic-admin",
    },
}

# Hop-by-hop plus everything we replace. Content-Length is recomputed by
# httpx for the request and dropped from the response (chunked streaming).
_STRIP_REQUEST = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "authorization", "x-api-key", "x-goog-api-key",
    # Internal call-site name, read into the audit row above; providers never
    # need it and it leaked feature names upstream until 2026-09-19.
    "x-llm-caller",
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


_TEXT_BODY_PATHS = {
    "openai": re.compile(r"(?:^|/)(?:chat/completions|responses)$"),
    "anthropic": re.compile(r"(?:^|/)v1/messages$"),
    "deepseek": re.compile(r"(?:^|/)(?:chat/completions|v1/messages)$"),
}
_GEMINI_GENERATE_PATH = re.compile(
    r"(?P<prefix>(?:^|/)models/)(?P<model>.+?)(?P<suffix>:(?:generateContent|streamGenerateContent|batchGenerateContent))$"
)


def normalize_text_model_request(
    provider: str, path: str, body: bytes,
) -> tuple[str, bytes, str | None, str | None]:
    """Resolve tier/known old model at the key-owning proxy boundary.

    Return (path, body, original_model, error). Unrelated endpoints stay byte
    identical. Unknown text models fail closed so newly written scripts cannot
    silently bypass the current-tier catalog with a pinned old ID.
    """
    if provider == "gemini":
        decoded = unquote(path)
        match = _GEMINI_GENERATE_PATH.search(decoded)
        if not match:
            return path, body, None, None
        original = match.group("model")
        current = current_text_model(provider, original)
        if current is None:
            return path, body, original, "unregistered Gemini text model; use tier:high|medium|low"
        if current == original:
            return path, body, original, None
        new_path = (decoded[:match.start("model")] + current + decoded[match.end("model"):])
        return new_path, body, original, None

    pattern = _TEXT_BODY_PATHS.get(provider)
    if pattern is None or not pattern.search(path):
        return path, body, None, None
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return path, body, None, "text generation request must contain a JSON model"
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        return path, body, None, "text generation request must contain a model string"
    original = payload["model"]
    current = current_text_model(provider, original)
    if current is None:
        return path, body, original, "unregistered text model; use tier:frontier|high|medium|low"
    if provider == "openai" and path.endswith("chat/completions"):
        if payload.get("tools") or payload.get("functions"):
            effort = payload.get("reasoning_effort")
            if effort is None:
                reasoning = payload.get("reasoning")
                effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
            if current == "gpt-6-astra" or effort != "none":
                return path, body, original, (
                    "GPT-6 Chat Completions function calls require reasoning_effort=none "
                    "(Sol/Luna); use Responses API for reasoning with tools"
                )
    if provider == "anthropic" and current in {"claude-opus-5-5", "claude-fable-5-1"}:
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            return path, body, original, "current Claude model requires adaptive thinking"
    if current == original:
        return path, body, original, None
    payload["model"] = current
    return path, json.dumps(payload, ensure_ascii=False).encode("utf-8"), original, None


# DeepSeek completion endpoints, both protocol families: the OpenAI-compatible
# path ends in chat/completions, the Anthropic-compatible one in v1/messages.
_DEEPSEEK_COMPLETION_PATH = re.compile(r"(?:^|/)(?:chat/completions|v1/messages)$")


def apply_deepseek_thinking_default(
    provider: str, path: str, body: bytes,
) -> tuple[bytes, bool]:
    """Default DeepSeek completion requests to thinking OFF. → (body, injected)

    DeepSeek V4 turns thinking ON when the request says nothing, and the
    reasoning shares max_tokens with the reply — a call that only wants text
    can burn the whole budget deliberating and return 200 with empty content.
    Every managed path states thinking explicitly (wrappers, loops, registry
    specs), but ad-hoc scripts keep forgetting it, and since the provider key
    lives only here, THIS is the one place such a script cannot bypass. Only a
    JSON object body on a deepseek completion route with no "thinking" key is
    touched; anything else — key present (any value), other providers, other
    endpoints, unparseable/compressed bodies — passes through byte-identical.
    """
    if provider != "deepseek" or not _DEEPSEEK_COMPLETION_PATH.search(path):
        return body, False
    try:
        payload = json.loads(body)
    except Exception:
        return body, False
    if not isinstance(payload, dict) or "thinking" in payload:
        return body, False
    payload["thinking"] = {"type": "disabled"}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), True


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


def billing_request(
    provider: str, days: int, *, now: datetime | None = None,
) -> tuple[dict, dict]:
    """Return fixed upstream request params and non-secret auth metadata.

    Kept pure so the security boundary and provider-specific date formats are
    unit-testable without making a network request.
    """
    cfg = BILLING_PROVIDERS[provider]
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start = now - timedelta(days=days)
    if provider == "openai":
        params = {
            "start_time": int(start.timestamp()),
            "end_time": int(now.timestamp()),
            "bucket_width": "1d",
            "limit": min(days + 1, 180),
        }
    elif provider == "claude":
        params = {
            "starting_at": start.isoformat().replace("+00:00", "Z"),
            "ending_at": now.isoformat().replace("+00:00", "Z"),
            "bucket_width": "1d",
            "limit": min(days + 1, 31),
        }
    else:
        params = {}
    return cfg, params


def billing_headers(auth: str, key: str) -> dict[str, str]:
    if auth == "anthropic-admin":
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "accept": "application/json",
        }
    return {"authorization": f"Bearer {key}", "accept": "application/json"}


def normalize_billing_response(provider: str, payload: dict, days: int) -> dict:
    """Normalize incompatible provider billing payloads for the operator CLI."""
    if provider == "deepseek":
        balances = [
            {
                "currency": str(item.get("currency") or ""),
                "available": float(item.get("total_balance") or 0),
                "granted": float(item.get("granted_balance") or 0),
                "cash": float(item.get("topped_up_balance") or 0),
            }
            for item in payload.get("balance_infos", [])
            if isinstance(item, dict)
        ]
        return {
            "provider": provider,
            "status": "ok",
            "kind": "balance",
            "can_call": bool(payload.get("is_available")),
            "balances": balances,
        }
    if provider == "kimi":
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        return {
            "provider": provider,
            "status": "ok" if payload.get("status") else "error",
            "kind": "balance",
            "can_call": float(data.get("available_balance") or 0) > 0,
            "balances": [{
                "currency": "USD",
                "available": float(data.get("available_balance") or 0),
                "granted": float(data.get("voucher_balance") or 0),
                "cash": float(data.get("cash_balance") or 0),
            }],
        }

    amounts: dict[str, float] = {}
    for bucket in payload.get("data", []):
        if not isinstance(bucket, dict):
            continue
        for result in bucket.get("results", []):
            if not isinstance(result, dict):
                continue
            if provider == "openai":
                amount = result.get("amount") or {}
                value = amount.get("value", 0) if isinstance(amount, dict) else 0
                currency = str(amount.get("currency") or "USD") if isinstance(amount, dict) else "USD"
            else:
                # Anthropic reports fractional cents, unlike OpenAI's dollars.
                value = float(result.get("amount") or 0) / 100
                currency = str(result.get("currency") or "USD")
            amounts[currency.upper()] = amounts.get(currency.upper(), 0.0) + float(value or 0)
    return {
        "provider": provider,
        "status": "ok",
        "kind": "cost",
        "window_days": days,
        "costs": [
            {"currency": currency, "amount": round(amount, 8)}
            for currency, amount in sorted(amounts.items())
        ],
        "has_more": bool(payload.get("has_more")),
    }


async def relay_and_record(
    upstream: httpx.Response, *, caller: str | None, provider: str | None,
    model: str | None, label: str, started: float,
):
    """Forward the upstream bytes; write the transport audit row at stream END.

    The row used to be written when the response HEADERS arrived, so a stream
    that died mid-flight stayed in the ledger as ok and latency_ms measured
    time-to-headers only. Recording after the last byte makes the authoritative
    row truthful: status reflects the actual stream outcome (upstream abort and
    client disconnect are distinguished in error_excerpt) and latency_ms covers
    the whole stream. The in-process seam still writes the billed row; this one
    stays the unbilled transport cross-check (estimate_cost=False).
    """
    bytes_out = 0
    outcome = None
    try:
        async for chunk in upstream.aiter_raw():
            bytes_out += len(chunk)
            yield chunk
    except GeneratorExit:
        outcome = "client disconnected mid-stream"
        raise
    except BaseException as e:  # CancelledError included — the row must be truthful
        detail = str(e)
        outcome = f"stream aborted: {e.__class__.__name__}" + (
            f": {detail}" if detail else ""
        )
        raise
    finally:
        await upstream.aclose()
        if outcome is None and not (200 <= upstream.status_code < 400):
            outcome = f"upstream HTTP {upstream.status_code}"
        latency_ms = int((time.monotonic() - started) * 1000)
        record_llm_call(
            surface="proxy", caller=caller, provider=provider, model=model,
            label=label, status="ok" if outcome is None else "error",
            error_excerpt=outcome, latency_ms=latency_ms, estimate_cost=False,
        )
        logger.info(
            "proxy stream end %s %s → %s (%dB in %dms)",
            provider or "?", label, outcome or "ok", bytes_out, latency_ms,
        )


@app.get("/health")
async def health():
    missing = [
        name for name, cfg in PROVIDERS.items()
        if not cfg.get("optional") and not _credential(cfg["secret"])
    ]
    payload = {
        "status": "ok" if not missing else "not_ready",
        "providers_without_key": missing,
        # Informational only: a DB outage must not take LLM traffic down.
        "audit_sink": audit_sink.sink_health(),
    }
    return JSONResponse(payload, status_code=200 if not missing else 503)


@app.post("/audit/{kind}")
async def audit_ingest(kind: str, request: Request):
    """Append audit rows. 400 for a malformed payload (the client drops it),
    503 when Postgres is unavailable (the client logs and drops it)."""
    if kind not in audit_sink.TABLES:
        return JSONResponse({"error": f"unknown audit ledger {kind!r}"}, status_code=404)
    body = await request.body()
    if len(body) > audit_sink.MAX_BODY_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)
    try:
        rows = audit_sink.normalize_rows(kind, json.loads(body))
    except (ValueError, TypeError) as e:
        logger.warning("audit sink rejected %s payload: %s", kind, e)
        return JSONResponse({"error": str(e)}, status_code=400)
    try:
        n = await _run_blocking(audit_sink.insert_rows, kind, rows)
    except Exception as e:
        logger.warning("audit sink insert failed (%d %s row(s)): %s", len(rows), kind, e)
        return JSONResponse({"error": f"db unavailable: {e.__class__.__name__}"}, status_code=503)
    return JSONResponse({"inserted": n})


@app.get("/audit/spend/today")
async def audit_spend_today():
    try:
        spend = await _run_blocking(audit_sink.today_spend)
    except Exception as e:
        logger.warning("audit sink spend read failed: %s", e)
        return JSONResponse({"error": f"db unavailable: {e.__class__.__name__}"}, status_code=503)
    return JSONResponse({"spend": spend})


async def _run_blocking(fn, *args):
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


@app.get("/billing/{provider}")
async def billing(provider: str, days: int = 30):
    """Read a provider balance/cost report through a fixed, read-only path.

    This endpoint is localhost-only with the rest of the proxy.  Missing
    optional admin credentials are a normal state and return a structured 200
    response so the operator CLI can fall back to the local audit estimate.
    """
    if provider not in BILLING_PROVIDERS:
        return JSONResponse(
            {"provider": provider, "status": "unsupported"}, status_code=404,
        )
    if not 1 <= days <= 30:
        return JSONResponse(
            {"provider": provider, "status": "invalid_days", "allowed": "1..30"},
            status_code=400,
        )
    cfg, params = billing_request(provider, days)
    key = _credential(cfg["secret"])
    if not key:
        return JSONResponse({
            "provider": provider,
            "status": "credential_missing",
            "required_credential": cfg["secret"],
        })

    started = time.monotonic()
    try:
        response = await _http_client().get(
            cfg["upstream"],
            headers=billing_headers(cfg["auth"], key),
            params=params,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("provider returned a non-object JSON payload")
        normalized = normalize_billing_response(provider, payload, days)
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as e:
        logger.warning("billing lookup failed for %s: %s", provider, e)
        return JSONResponse({
            "provider": provider,
            "status": "upstream_error",
            "error": e.__class__.__name__,
        }, status_code=502)

    logger.info(
        "billing lookup %s ok in %dms",
        provider, int((time.monotonic() - started) * 1000),
    )
    return JSONResponse(normalized)


@app.api_route("/{provider}/{path:path}", methods=["GET", "POST"])
async def proxy(provider: str, path: str, request: Request):
    cfg = PROVIDERS.get(provider)
    if cfg is None:
        return JSONResponse({"error": f"unknown provider {provider!r}"}, status_code=404)
    key = _credential(cfg["secret"])
    if not key:
        return JSONResponse(
            {"error": f"no credential for provider {provider!r}"}, status_code=503,
        )

    # Gemini SDKs may carry the key as a query parameter; drop it.
    params = [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
    body = await request.body()
    path, body, requested_model, model_error = normalize_text_model_request(provider, path, body)
    if model_error:
        logger.warning("proxy rejected model selection %s/%s: %s", provider, requested_model, model_error)
        return JSONResponse({"error": model_error, "requested_model": requested_model}, status_code=400)
    body, thinking_injected = apply_deepseek_thinking_default(provider, path, body)
    if thinking_injected:
        logger.info(
            "proxy deepseek /%s: request had no thinking field; "
            "injected {'type': 'disabled'}", path,
        )
    normalized_model = model_from_request(provider, path, body)
    model_rewritten = bool(requested_model and normalized_model != requested_model)
    if model_rewritten:
        logger.info("proxy %s model %s -> %s", provider, requested_model, normalized_model)
    audit_label = (path + (" +model-normalized" if model_rewritten else "")
                   + (" +think-off-default" if thinking_injected else ""))[:200]

    # Authoritative policy gate. The decision logic is shared with the
    # in-process seam (llm/gateway.evaluate_policy — single source); THIS
    # evaluation is the one a caller cannot skip, because the provider key
    # only exists on the far side of it.
    model = normalized_model
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
            provider=policy_provider, model=model, label=audit_label,
            status="error", error_excerpt=f"{e.__class__.__name__}: {e}",
            latency_ms=int((time.monotonic() - started) * 1000),
            estimate_cost=False,
        )
        # httpx timeouts often str() to "" — keep the class name so the
        # journal and audit row say what actually happened.
        logger.warning(
            "proxy %s/%s upstream error: %s: %s",
            provider, path, e.__class__.__name__, e,
        )
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
        relay_and_record(
            upstream, caller=request.headers.get("x-llm-caller"),
            provider=policy_provider, model=model, label=audit_label,
            started=started,
        ),
        status_code=upstream.status_code,
        headers={
            k: v for k, v in upstream.headers.items()
            if k.lower() not in _STRIP_RESPONSE
        },
    )


@app.on_event("shutdown")
async def _shutdown():
    if _client is not None:
        await _client.aclose()
