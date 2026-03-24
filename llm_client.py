"""
llm_client.py — 프로젝트 공용 LLM 클라이언트 (MOON PC 우선, 로컬 llama-server 폴백)

우선순위:
    1차: MOON PC llama-server (qwen3.5-9b Q8_0, SSH 터널 127.0.0.1:8080)
    2차: 로컬 llama-server (qwen3.5-4b Q4_K_M, localhost:11435)

양쪽 모두 OpenAI 호환 API (/v1/chat/completions).

사용 예:
    from llm_client import ask, ask_with_system, ask_chat

    result = ask("마르크스의 잉여가치론을 설명하라")
    result = ask_with_system("질문", system_prompt="너는 시인이다")
"""

import os
import time
import logging
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── 설정 ─────────────────────────────────────────────────────────────────────
MOON_BASE       = os.getenv("MOON_LLM_BASE_URL", "http://127.0.0.1:8080")
MOON_MODEL      = os.getenv("MOON_LLM_MODEL", "qwen3.5-9b")

LOCAL_BASE      = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11435")
LOCAL_MODEL     = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-4b")

TIMEOUT        = 300
HEALTH_TIMEOUT = 3

# ── 백엔드 정의 ──────────────────────────────────────────────────────────────
_BACKENDS = [
    {"name": "moon",  "base": MOON_BASE,  "model": MOON_MODEL},
    {"name": "local", "base": LOCAL_BASE, "model": LOCAL_MODEL},
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

    raise ConnectionError("MOON PC와 로컬 llama-server 모두 응답 없음")


# ── OpenAI 호환 호출 (공통) ───────────────────────────────────────────────────
def _call(base: str, model: str, messages: list[dict],
          temperature: float, timeout: int) -> str:
    resp = httpx.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_llm(messages: list[dict], temperature: float = 0.7,
              timeout: int = TIMEOUT) -> str:
    """백엔드 자동 선택 후 호출. 1차 실패 시 폴백."""
    b = _resolve_backend()
    try:
        return _call(b["base"], b["model"], messages, temperature, timeout)
    except Exception as e:
        logger.warning("[llm] %s 호출 실패: %s → 폴백 시도", b["name"], e)
        _backend_cache.clear()
        b2 = _resolve_backend(force_refresh=True)
        if b2["name"] == b["name"]:
            raise
        return _call(b2["base"], b2["model"], messages, temperature, timeout)


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


DEFAULT_MODEL = MOON_MODEL


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
