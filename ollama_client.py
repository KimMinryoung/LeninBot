"""
ollama_client.py — 프로젝트 공용 LLM 클라이언트 (MOON PC 우선, 로컬 Ollama 폴백)

우선순위:
    1차: MOON PC llama-server (OpenAI 호환 API, qwen3.5-9b Q8_0)
    2차: 로컬 Ollama (qwen3.5:4b)

사용 예:
    from ollama_client import ask, ask_with_system, ask_chat

    result = ask("마르크스의 잉여가치론을 설명하라")
    result = ask_with_system("질문", system_prompt="너는 시인이다")
"""

import os
import logging
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── 설정 ─────────────────────────────────────────────────────────────────────
MOON_BASE       = os.getenv("MOON_LLM_BASE_URL", "http://127.0.0.1:8080")
MOON_URL        = f"{MOON_BASE}/v1/chat/completions"
MOON_HEALTH_URL = f"{MOON_BASE}/health"
MOON_MODEL      = os.getenv("MOON_LLM_MODEL", "qwen3.5-9b")

LOCAL_URL      = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
LOCAL_API      = f"{LOCAL_URL}/api/chat"
LOCAL_MODEL    = os.getenv("LOCAL_LLM_MODEL", "qwen3.5:4b")

TIMEOUT        = 300
HEALTH_TIMEOUT = 3

# ── 내부: 백엔드 상태 캐시 ───────────────────────────────────────────────────
_backend_cache: dict = {}  # {"backend": "moon"|"local", "checked_at": float}

import time

def _check_moon() -> bool:
    """MOON PC llama-server 헬스체크."""
    try:
        resp = httpx.get(MOON_HEALTH_URL, timeout=HEALTH_TIMEOUT)
        return resp.status_code == 200
    except Exception:
        return False


def _check_local() -> bool:
    """로컬 Ollama 헬스체크."""
    try:
        resp = httpx.get(f"{LOCAL_URL}/api/tags", timeout=HEALTH_TIMEOUT)
        return resp.status_code == 200
    except Exception:
        return False


def _resolve_backend(force_refresh: bool = False) -> tuple[str, str, str]:
    """
    사용 가능한 백엔드를 결정. (backend_name, api_url, model) 반환.
    5초간 캐시하여 매 요청마다 헬스체크하지 않음.
    """
    now = time.time()
    if (
        not force_refresh
        and _backend_cache
        and now - _backend_cache.get("checked_at", 0) < 5
    ):
        b = _backend_cache["backend"]
        if b == "moon":
            return ("moon", MOON_URL, MOON_MODEL)
        return ("local", LOCAL_API, LOCAL_MODEL)

    if _check_moon():
        _backend_cache.update(backend="moon", checked_at=now)
        logger.info("[llm] MOON PC 연결 성공 → %s (%s)", MOON_URL, MOON_MODEL)
        return ("moon", MOON_URL, MOON_MODEL)

    if _check_local():
        _backend_cache.update(backend="local", checked_at=now)
        logger.info("[llm] MOON 실패, 로컬 Ollama 폴백 → %s (%s)", LOCAL_API, LOCAL_MODEL)
        return ("local", LOCAL_API, LOCAL_MODEL)

    raise ConnectionError("MOON PC와 로컬 Ollama 모두 응답 없음")


# ── OpenAI 호환 요청 (MOON llama-server) ─────────────────────────────────────
def _call_openai_compat(url: str, model: str, messages: list[dict],
                        temperature: float, timeout: int) -> str:
    resp = httpx.post(
        url,
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


# ── Ollama 네이티브 요청 (로컬 폴백) ─────────────────────────────────────────
def _call_ollama(url: str, model: str, messages: list[dict],
                 temperature: float, timeout: int) -> str:
    resp = httpx.post(
        url,
        json={
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "keep_alive": -1,
            "think":      False,
            "options": {
                "temperature": temperature,
                "top_k":       20,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── 통합 호출 ────────────────────────────────────────────────────────────────
def _call_llm(messages: list[dict], temperature: float = 0.7,
              timeout: int = TIMEOUT) -> str:
    """백엔드 자동 선택 후 호출. 1차 실패 시 폴백 시도."""
    backend, url, model = _resolve_backend()

    try:
        if backend == "moon":
            return _call_openai_compat(url, model, messages, temperature, timeout)
        else:
            return _call_ollama(url, model, messages, temperature, timeout)
    except Exception as e:
        logger.warning("[llm] %s 호출 실패: %s → 폴백 시도", backend, e)
        # 캐시 무효화 후 반대쪽 시도
        _backend_cache.clear()
        backend2, url2, model2 = _resolve_backend(force_refresh=True)
        if backend2 == backend:
            raise  # 같은 백엔드만 살아있으면 재시도 무의미
        if backend2 == "moon":
            return _call_openai_compat(url2, model2, messages, temperature, timeout)
        else:
            return _call_ollama(url2, model2, messages, temperature, timeout)


# ── 공개 API (기존 인터페이스 유지) ───────────────────────────────────────────

def ask(
    prompt: str,
    model: str | None = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 20,
    timeout: int = TIMEOUT,
) -> str:
    """단일 프롬프트 호출. model 파라미터는 하위호환용 (무시됨, 자동 선택)."""
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
    return ask(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
    )


def ask_chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """다중 턴 대화용 호출."""
    return _call_llm(messages, temperature)


def check_ollama() -> dict:
    """서버 상태 확인 (하위호환)."""
    try:
        backend, url, model = _resolve_backend(force_refresh=True)
        return {
            "status": "ok",
            "backend": backend,
            "model": model,
            "url": url,
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


DEFAULT_MODEL = MOON_MODEL  # 하위호환용


# ── 빠른 테스트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== LLM 클라이언트 자가 테스트 ===\n")

    status = check_ollama()
    print(f"백엔드 상태: {status}")

    if status["status"] != "ok":
        print("LLM 서버 응답 없음. 테스트 중단.")
        exit(1)

    print(f"\n[1] ask() 호출 ({status['backend']}: {status['model']})...")
    result = ask("한 문장으로: 변증법적 유물론이란?")
    print(f"응답: {result[:300]}")

    print("\n=== 테스트 완료 ===")
