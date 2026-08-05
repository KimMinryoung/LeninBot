"""llm/call_registry.py — central registry for one-shot LLM call sites.

Agent loops (analyst/scout/diary/...) are governed by config/agent_runtime.json
+ llm/runtime_profile.py. Everything else — small single-request calls scattered
across features (chunk summaries, classifiers, critics, query translation) —
is governed here, by config/llm_call_sites.json.

Each call site is a *feature key* in the JSON. Effective config resolution
order for every field:

  1. legacy env vars listed in the entry's "env" (model only, ops habit compat)
  2. generic env override  LLM_SITE_<FEATURE>_MODEL
  3. the JSON entry
  4. the caller-supplied default

The JSON file is hot-reloaded on mtime change, so edits (or `scripts/
llm_registry_cli.py set`) take effect without a service restart.

Two ways to consume the registry:

  - resolve(feature)              → CallSiteProfile (model-only integration —
                                    KG/graphiti model selection, writer critic)
  - generate(feature, prompt)     → run the call through the shared executor
    generate_sync(feature, prompt)  (gemini / deepseek / openai / claude)

Executor calls pass through the LLM gateway (llm/gateway.py): policy check
before the request, spend/usage audit after. Third-party model-only clients
must use the audited wrappers documented in dev_docs/llm_gateway.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from secrets_loader import get_secret

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_call_sites.json"

# One connection record owns route, credential, and any explicit direct-base
# override. Keeping these together prevents proxy routing and placeholder-key
# resolution from drifting apart.
_PROVIDER_CONNECTIONS = {
    "deepseek": {
        "credential": "DEEPSEEK_API_KEY",
        "direct_base": "https://api.deepseek.com",
        "proxy_path": "deepseek",
    },
    "deepseek_anthropic": {
        "credential": "DEEPSEEK_API_KEY",
        "direct_base": "https://api.deepseek.com/anthropic",
        "direct_base_env": "DEEPSEEK_ANTHROPIC_BASE_URL",
        "proxy_path": "deepseek/anthropic",
    },
    "kimi": {
        "credential": "MOONSHOT_API_KEY",
        "direct_base": "https://api.moonshot.ai/v1",
        "proxy_path": "moonshot/v1",
    },
    "openai": {
        "credential": "OPENAI_API_KEY",
        "direct_base": None,
        "proxy_path": "openai/v1",
    },
    "claude": {
        "credential": "ANTHROPIC_API_KEY",
        "direct_base": None,
        "proxy_path": "anthropic",
    },
    "gemini": {
        "credential": "GEMINI_API_KEY",
        "direct_base": None,
        "proxy_path": "gemini",
    },
}


@dataclass(frozen=True)
class CallSiteProfile:
    feature: str
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 60.0
    json_mode: bool = False
    note: str = ""
    managed: str = "executor"  # executor | model-only | external
    model_env_override: str | None = None  # which env var won, if any
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderConnection:
    """Resolved SDK connection for a registry provider."""

    provider: str
    credential_name: str
    base_url: str | None
    api_key: str


class ProviderConnectionError(RuntimeError):
    """A registry provider cannot be called with the current credentials."""

    def __init__(self, provider: str, credential_name: str):
        self.provider = provider
        self.credential_name = credential_name
        super().__init__(f"{credential_name} not configured for provider {provider}")


def resolve_provider_connection(provider: str) -> ProviderConnection:
    """Resolve base URL and key together, including gateway placeholder mode.

    An explicit direct-base environment override bypasses the proxy and
    therefore still requires the real provider key. Otherwise proxy mode may
    safely use the public placeholder because llm_proxy injects the key.
    """
    try:
        cfg = _PROVIDER_CONNECTIONS[provider]
    except KeyError:
        raise ValueError(f"unknown registry provider: {provider!r}") from None

    credential_name = cfg["credential"]
    real_key = (get_secret(credential_name, "") or "").strip()
    env_name = cfg.get("direct_base_env")
    explicit_base = (os.getenv(env_name) or "").strip() if env_name else ""
    if explicit_base:
        base_url, api_key = explicit_base.rstrip("/"), real_key
    else:
        from llm.gateway import provider_endpoint

        base_url, api_key = provider_endpoint(
            cfg["proxy_path"], cfg["direct_base"], real_key,
        )
    if not api_key:
        raise ProviderConnectionError(provider, credential_name)
    return ProviderConnection(
        provider=provider,
        credential_name=credential_name,
        base_url=base_url,
        api_key=api_key,
    )


# ── Config loading (hot reload) ──────────────────────────────────────

_lock = threading.Lock()
_cache: dict | None = None
_cache_mtime: float | None = None


def _load_config() -> dict:
    global _cache, _cache_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        logger.warning("[llm-registry] config missing: %s", CONFIG_PATH)
        return {}
    with _lock:
        if _cache is not None and _cache_mtime == mtime:
            return _cache
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON must be an object")
        except Exception as e:
            logger.error("[llm-registry] config unreadable (%s); keeping previous", e)
            return _cache or {}
        _cache, _cache_mtime = data, mtime
        return data


def list_features() -> dict[str, dict]:
    """Raw JSON entries, for the CLI."""
    return dict(_load_config())


def resolve(feature: str, **defaults) -> CallSiteProfile:
    """Resolve the effective profile for a feature.

    Unknown features resolve from `defaults` alone (and log once), so a
    missing registry entry degrades to the call site's built-in behavior
    instead of breaking the feature.
    """
    entry = _load_config().get(feature)
    if entry is None:
        logger.warning("[llm-registry] feature %r not registered; using call-site defaults", feature)
        entry = {}

    model = str(entry.get("model") or defaults.get("model") or "")
    override_src = None
    generic_env = f"LLM_SITE_{feature.upper()}_MODEL"
    env_names = list(entry.get("env") or []) + [generic_env]
    for name in env_names:
        val = (os.getenv(name) or "").strip()
        if val:
            model, override_src = val, name
            break

    def _pick(key, fallback):
        return entry.get(key, defaults.get(key, fallback))

    return CallSiteProfile(
        feature=feature,
        provider=str(_pick("provider", "gemini")),
        model=model,
        temperature=float(_pick("temperature", 0.0)),
        max_tokens=int(_pick("max_tokens", 1024)),
        timeout=float(_pick("timeout", 60.0)),
        json_mode=bool(_pick("json_mode", False)),
        note=str(entry.get("note", "")),
        managed=str(entry.get("managed", "executor")),
        model_env_override=override_src,
        extra={k: v for k, v in entry.items() if k not in (
            "provider", "model", "temperature", "max_tokens", "timeout",
            "json_mode", "note", "managed", "env",
        )},
    )


# ── Shared executor ──────────────────────────────────────────────────


def _gemini_usage(response) -> dict:
    meta = getattr(response, "usage_metadata", None)
    return {
        "tokens_in": getattr(meta, "prompt_token_count", 0) or 0,
        "tokens_out": (
            (getattr(meta, "candidates_token_count", 0) or 0)
            + (getattr(meta, "thoughts_token_count", 0) or 0)
        ),
        "cache_read": getattr(meta, "cached_content_token_count", 0) or 0,
    }


def _openai_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details and not cached:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "tokens_in": getattr(usage, "prompt_tokens", 0) or 0,
        "tokens_out": getattr(usage, "completion_tokens", 0) or 0,
        "cache_read": cached,
    }


def _anthropic_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    return {
        "tokens_in": getattr(usage, "input_tokens", 0) or 0,
        "tokens_out": getattr(usage, "output_tokens", 0) or 0,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_create": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _generate_gemini(p: CallSiteProfile, prompt: str, system: str | None) -> tuple[str, dict]:
    from google import genai
    from google.genai.types import GenerateContentConfig

    connection = resolve_provider_connection("gemini")
    client = genai.Client(
        api_key=connection.api_key,
        **({"http_options": {"base_url": connection.base_url}}
           if connection.base_url else {}),
    )
    config = GenerateContentConfig(
        temperature=p.temperature,
        max_output_tokens=p.max_tokens,
        system_instruction=system or None,
        response_mime_type="application/json" if p.json_mode else None,
    )
    response = client.models.generate_content(
        model=p.model, contents=prompt, config=config,
    )
    return (response.text or "").strip(), _gemini_usage(response)


def _generate_openai_compat(p: CallSiteProfile, prompt: str, system: str | None) -> tuple[str, dict]:
    from openai import OpenAI

    connection = resolve_provider_connection(p.provider)
    client = OpenAI(
        api_key=connection.api_key,
        base_url=connection.base_url,
        timeout=p.timeout,
    )
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    kwargs: dict = {}
    if p.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if p.provider == "openai":
        # GPT-5.6은 추론 모델 — max_tokens가 아니라 max_completion_tokens를 받고,
        # temperature는 기본값만 허용한다 (openai_tool_loop._call_sdk와 같은 계약).
        # max_completion_tokens는 추론 토큰까지 포함하므로 reasoning_effort로 억제하지
        # 않으면 짧은 상한에서 본문이 비어 돌아온다.
        kwargs["max_completion_tokens"] = p.max_tokens
        effort = p.extra.get("reasoning_effort")
        if effort:
            kwargs["reasoning_effort"] = str(effort)
    else:
        kwargs["max_tokens"] = p.max_tokens
        # Kimi K3는 temperature=1만 허용 (그 외 400) — 파라미터 자체를 생략한다.
        if p.provider != "kimi":
            kwargs["temperature"] = p.temperature
        # DeepSeek V4는 thinking이 기본 ON이라, 끄지 않으면 추론이 max_tokens를
        # 다 쓰고 본문 없이 200이 돌아온다. 토글의 정식 자리는 Anthropic 호환
        # 엔드포인트(provider="deepseek_anthropic")지만, json_mode처럼 그쪽에
        # 대응이 없는 호출부를 위해 여기서도 넘길 수 있게 열어둔다. 선언한
        # 호출부에만 적용되므로 기존 동작은 그대로다.
        if p.provider == "deepseek" and isinstance(p.extra.get("thinking"), dict):
            kwargs["extra_body"] = {"thinking": p.extra["thinking"]}
    response = client.chat.completions.create(
        model=p.model,
        messages=messages,
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip(), _openai_usage(response)


def _generate_claude(p: CallSiteProfile, prompt: str, system: str | None) -> tuple[str, dict]:
    import anthropic

    connection = resolve_provider_connection("claude")
    client = anthropic.Anthropic(
        api_key=connection.api_key, timeout=p.timeout,
        **({"base_url": connection.base_url} if connection.base_url else {}),
    )
    kwargs: dict = {"system": system} if system else {}
    response = client.messages.create(
        model=p.model,
        max_tokens=p.max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    text = " ".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    ).strip()
    return text, _anthropic_usage(response)


def _generate_deepseek_anthropic(p: CallSiteProfile, prompt: str, system: str | None) -> tuple[str, dict]:
    """DeepSeek over its Anthropic-compatible endpoint, thinking off by default.

    DeepSeek V4 defaults to thinking mode, and the toggle only exists on this
    endpoint (see bot_config._get_deepseek_thinking_params). On the plain
    OpenAI-compatible path a think-heavy response can spend the whole
    max_tokens budget before emitting any visible text, so the call returns
    200 with empty content and generate_sync reports it as a failure with no
    cause — the empty-reply mode bot_config documents for autonomous ticks
    216/217. One-shot generation call sites want text, not deliberation, so
    thinking is disabled unless the call site asks for it.
    """
    import anthropic

    connection = resolve_provider_connection("deepseek_anthropic")
    client = anthropic.Anthropic(
        api_key=connection.api_key,
        base_url=connection.base_url,
        timeout=p.timeout,
    )
    kwargs: dict = {"system": system} if system else {}
    thinking = p.extra.get("thinking")
    if isinstance(thinking, dict):
        kwargs["thinking"] = thinking
        effort = p.extra.get("output_config")
        if isinstance(effort, dict):
            kwargs["output_config"] = effort
    else:
        kwargs["thinking"] = {"type": "disabled"}

    response = client.messages.create(
        model=p.model,
        max_tokens=p.max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    text = " ".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    ).strip()
    return text, _anthropic_usage(response)


_EXECUTORS = {
    "gemini": _generate_gemini,
    "deepseek": _generate_openai_compat,
    "deepseek_anthropic": _generate_deepseek_anthropic,
    "kimi": _generate_openai_compat,
    "openai": _generate_openai_compat,
    "claude": _generate_claude,
}


def generate_sync(feature: str, prompt: str, *, system: str | None = None, **defaults) -> str | None:
    """Run a one-shot generation for a registered feature (blocking).

    Returns None on any failure — call sites keep their own fallbacks
    (extractive summary, skip, default label) instead of blocking.
    """
    from llm.gateway import LLMGatewayDenied, check_llm_call, record_llm_call

    profile = resolve(feature, **defaults)
    executor = _EXECUTORS.get(profile.provider)
    if executor is None:
        logger.error("[llm-registry] %s: unknown provider %r", feature, profile.provider)
        return None
    if not profile.model:
        logger.error("[llm-registry] %s: no model configured", feature)
        return None
    # deepseek_anthropic is a protocol variant, not a distinct provider.
    provider = "deepseek" if profile.provider == "deepseek_anthropic" else profile.provider
    token_semantics = {
        "deepseek_anthropic": "anthropic",
        "claude": "anthropic",
        "gemini": "gemini",
    }.get(profile.provider, "openai")
    try:
        check_llm_call(
            surface="oneshot", caller=feature,
            provider=provider, model=profile.model,
        )
    except LLMGatewayDenied as e:
        # Denial behaves like any other failure: the call site's own
        # fallback (extractive summary, default label, skip) takes over.
        logger.warning("[llm-registry] %s denied by llm gateway: %s", feature, e)
        return None
    started = time.monotonic()
    try:
        text, usage = executor(profile, prompt, system)
    except Exception as e:
        record_llm_call(
            surface="oneshot", caller=feature, provider=provider,
            model=profile.model, status="error", error_excerpt=str(e),
            latency_ms=int((time.monotonic() - started) * 1000),
            token_semantics=token_semantics, estimate_cost=False,
        )
        logger.warning("[llm-registry] %s (%s/%s) failed: %s",
                       feature, profile.provider, profile.model, e)
        return None
    record_llm_call(
        surface="oneshot", caller=feature, provider=provider,
        model=profile.model, latency_ms=int((time.monotonic() - started) * 1000),
        token_semantics=token_semantics,
        **usage,
    )
    return text or None


async def generate(feature: str, prompt: str, *, system: str | None = None, **defaults) -> str | None:
    """Async wrapper around generate_sync with the profile timeout enforced."""
    profile = resolve(feature, **defaults)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(generate_sync, feature, prompt, system=system, **defaults),
            timeout=profile.timeout + 5,  # executor-level timeout is primary; this is the backstop
        )
    except asyncio.TimeoutError:
        logger.warning("[llm-registry] %s timed out after %.0fs", feature, profile.timeout + 5)
        return None
