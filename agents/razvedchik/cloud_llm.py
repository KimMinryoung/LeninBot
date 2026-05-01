"""DeepSeek-backed LLM helpers for Razvedchik.

Razvedchik must not depend on the MOON PC/local llama-server path. This module
pins scout generation to the DeepSeek cloud API and rejects localhost-style
base URLs so a misconfigured environment fails loudly instead of silently
returning to a single home-PC dependency.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from secrets_loader import get_secret

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
TIMEOUT = float(os.getenv("RAZVEDCHIK_LLM_TIMEOUT", "120"))
MAX_TOKENS = int(os.getenv("RAZVEDCHIK_LLM_MAX_TOKENS", "2048"))

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "moon"}


def _model() -> str:
    return os.getenv("RAZVEDCHIK_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _base_url() -> str:
    base = os.getenv("RAZVEDCHIK_LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
    base = base.strip().rstrip("/")
    parsed = urlparse(base)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host in _LOCAL_HOSTS or host.endswith(".local"):
        raise RuntimeError(f"Razvedchik refuses non-cloud LLM base URL: {base}")
    if "ollama" in host or "llama" in host:
        raise RuntimeError(f"Razvedchik refuses local LLM-looking base URL: {base}")
    return base


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = get_secret("DEEPSEEK_API_KEY", "") or ""
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for Razvedchik LLM calls")
    return OpenAI(api_key=key, base_url=_base_url(), timeout=TIMEOUT)


def ask_chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    selected_model = model or _model()
    logger.info("[razvedchik-llm] DeepSeek cloud call model=%s base=%s", selected_model, _base_url())
    response = _client().chat.completions.create(
        model=selected_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or MAX_TOKENS,
        stream=False,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


def ask(
    prompt: str,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    timeout: int | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return ask_chat(messages, model=model, temperature=temperature)


def ask_with_system(
    user_prompt: str,
    system_prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    return ask(user_prompt, model=model, system_prompt=system_prompt, temperature=temperature)


def check_llm() -> dict:
    try:
        _client()
        return {"status": "ok", "provider": "deepseek", "model": _model(), "url": _base_url()}
    except Exception as e:
        return {"status": "error", "provider": "deepseek", "model": _model(), "reason": str(e)}
