"""
llm_client.py — 프로젝트 공용 LLM 클라이언트 (MOON PC llama-server)

MOON PC llama-server (qwen3.5-9b Q8_0, Tailscale magic DNS http://moon:8080)
OpenAI 호환 API (/v1/chat/completions).
.env: MOON_LLM_BASE_URL, MOON_LLM_MODEL
"""

import asyncio
import os
import time
import logging
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── 설정 ─────────────────────────────────────────────────────────────────────
MOON_BASE       = os.getenv("MOON_LLM_BASE_URL", "http://moon:8080")
MOON_MODEL      = os.getenv("MOON_LLM_MODEL", "qwen3.5-9b")

# Concurrency limit: llama-server runs with 1 slot (no --parallel flag).
# Serialize requests to avoid API errors under concurrent task load.
LOCAL_SEMAPHORE = asyncio.Semaphore(1)  # --parallel 1 on MOON PC

TIMEOUT        = 300
HEALTH_TIMEOUT = 3
LOCAL_CONTEXT_LIMIT = int(os.getenv("MOON_LLM_CONTEXT", "131072"))

# Generation-side defaults for local (llama-server) dispatch.
# Tool calls fail silently when the thinking block consumes the entire
# completion budget before a <tool_call> can be emitted. 8192 gives room
# for both reasoning and the tool-call payload on Q4 quantizations; scout
# agent already runs with 8192 for the same reason.
LOCAL_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "8192"))

# Qwen3 thinking mode often hurts tool-call reliability on Q4 quantized
# weights — the <think> block eats the token budget and the tool_call
# token is never emitted. Env-overridable so a healthier model (e.g.
# Qwen3.6) can re-enable it without a code change.
LOCAL_ENABLE_THINKING = os.getenv("LOCAL_LLM_ENABLE_THINKING", "true").strip().lower() != "false"

# ── 백엔드 정의 ──────────────────────────────────────────────────────────────
_BACKENDS = [
    {"name": "moon",  "base": MOON_BASE,  "model": MOON_MODEL},
]

_backend_cache: dict = {}  # {"name", "base", "model", "checked_at"}


def _health_ok(base_url: str) -> bool:
    """llama-server /health 헬스체크."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=HEALTH_TIMEOUT)
        return resp.status_code == 200
    except Exception:
        return False


def _resolve_backend(force_refresh: bool = False) -> dict:
    """사용 가능한 백엔드 반환. 5초 캐시."""
    now = time.time()
    if (
        not force_refresh
        and _backend_cache
        and now - _backend_cache.get("checked_at", 0) < 5
    ):
        return _backend_cache

    for b in _BACKENDS:
        if _health_ok(b["base"]):
            result = {**b, "checked_at": now}
            _backend_cache.update(result)
            logger.info("[llm] %s 연결 → %s (%s)", b["name"], b["base"], b["model"])
            return result

    raise ConnectionError("MOON PC llama-server 응답 없음")


# ── OpenAI 호환 호출 (공통) ───────────────────────────────────────────────────

def _extract_answer(msg: dict) -> str:
    """Return the model's response with <think> reasoning preserved.

    Qwopus3.5 is distilled from Claude Opus reasoning; per the model card, its
    <think> blocks are meant to be visible ("transparently follow the AI's
    internal logic"). When tokens run out mid-thinking, showing the reasoning
    is better than returning empty — the user still sees the train of thought.
    """
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()

    if reasoning and content:
        return f"💭 {reasoning}\n\n{content}"
    if reasoning:
        return f"💭 {reasoning}"
    return content


def _call(base: str, model: str, messages: list[dict],
          temperature: float, timeout: int) -> str:
    resp = httpx.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": LOCAL_MAX_TOKENS,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return _extract_answer(resp.json()["choices"][0]["message"])


def _call_llm(messages: list[dict], temperature: float = 0.7,
              timeout: int = TIMEOUT) -> str:
    """백엔드 선택 후 호출."""
    b = _resolve_backend()
    return _call(b["base"], b["model"], messages, temperature, timeout)


# ── 공개 API ─────────────────────────────────────────────────────────────────

def ask(
    prompt: str,
    model: str | None = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 20,
    timeout: int = TIMEOUT,
) -> str:
    """단일 프롬프트 호출."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return _call_llm(messages, temperature, timeout)


def ask_with_system(
    user_prompt: str,
    system_prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """시스템 프롬프트 포함 호출."""
    return ask(prompt=user_prompt, system_prompt=system_prompt, temperature=temperature)


def ask_chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """다중 턴 대화용 호출."""
    return _call_llm(messages, temperature)


def check_llm() -> dict:
    """서버 상태 확인 (하위호환)."""
    try:
        b = _resolve_backend(force_refresh=True)
        return {"status": "ok", "backend": b["name"], "model": b["model"], "url": b["base"]}
    except Exception as e:
        return {"status": "error", "reason": str(e)}



# ── 빠른 테스트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== LLM 클라이언트 자가 테스트 ===\n")

    status = check_llm()
    print(f"백엔드 상태: {status}")

    if status["status"] != "ok":
        print("LLM 서버 응답 없음. 테스트 중단.")
        exit(1)

    print(f"\n[1] ask() 호출 ({status['backend']}: {status['model']})...")
    result = ask("한 문장으로: 변증법적 유물론이란?")
    print(f"응답: {result[:300]}")

    print("\n=== 테스트 완료 ===")
