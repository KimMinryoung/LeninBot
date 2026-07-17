"""llm/call_registry.py — central registry for one-shot LLM call sites.

Agent loops (analyst/scout/diary/...) are governed by config/agent_runtime.json
+ runtime_profile.py. Everything else — small single-request calls scattered
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
                                    KG/graphiti, razvedchik, writer critic)
  - generate(feature, prompt)     → run the call through the shared executor
    generate_sync(feature, prompt)  (gemini / deepseek / openai / claude)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from secrets_loader import get_secret

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_call_sites.json"

_PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "kimi": "https://api.moonshot.ai/v1",
    "openai": None,
}
_PROVIDER_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
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

def _api_key(provider: str) -> str:
    name = _PROVIDER_KEYS.get(provider)
    key = (get_secret(name, "") or "").strip() if name else ""
    if not key:
        raise RuntimeError(f"{name or provider} not configured")
    return key


def _generate_gemini(p: CallSiteProfile, prompt: str, system: str | None) -> str:
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=_api_key("gemini"))
    config = GenerateContentConfig(
        temperature=p.temperature,
        max_output_tokens=p.max_tokens,
        system_instruction=system or None,
        response_mime_type="application/json" if p.json_mode else None,
    )
    response = client.models.generate_content(
        model=p.model, contents=prompt, config=config,
    )
    return (response.text or "").strip()


def _generate_openai_compat(p: CallSiteProfile, prompt: str, system: str | None) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=_api_key(p.provider),
        base_url=_PROVIDER_BASE_URLS.get(p.provider),
        timeout=p.timeout,
    )
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    kwargs: dict = {}
    if p.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # Kimi K3는 temperature=1만 허용 (그 외 400) — 파라미터 자체를 생략한다.
    if p.provider != "kimi":
        kwargs["temperature"] = p.temperature
    response = client.chat.completions.create(
        model=p.model,
        messages=messages,
        max_tokens=p.max_tokens,
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def _generate_claude(p: CallSiteProfile, prompt: str, system: str | None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key("claude"), timeout=p.timeout)
    kwargs: dict = {"system": system} if system else {}
    response = client.messages.create(
        model=p.model,
        max_tokens=p.max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return " ".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    ).strip()


_EXECUTORS = {
    "gemini": _generate_gemini,
    "deepseek": _generate_openai_compat,
    "kimi": _generate_openai_compat,
    "openai": _generate_openai_compat,
    "claude": _generate_claude,
}


def generate_sync(feature: str, prompt: str, *, system: str | None = None, **defaults) -> str | None:
    """Run a one-shot generation for a registered feature (blocking).

    Returns None on any failure — call sites keep their own fallbacks
    (extractive summary, skip, default label) instead of blocking.
    """
    profile = resolve(feature, **defaults)
    executor = _EXECUTORS.get(profile.provider)
    if executor is None:
        logger.error("[llm-registry] %s: unknown provider %r", feature, profile.provider)
        return None
    if not profile.model:
        logger.error("[llm-registry] %s: no model configured", feature)
        return None
    try:
        return executor(profile, prompt, system) or None
    except Exception as e:
        logger.warning("[llm-registry] %s (%s/%s) failed: %s",
                       feature, profile.provider, profile.model, e)
        return None


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
