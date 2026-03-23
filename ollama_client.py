"""
ollama_client.py — 프로젝트 공용 Ollama 클라이언트

설정이 검증된 표준 호출 함수 모음.
어디서든 import해서 사용하면 됨.

기본 설정:
    - think=False       : thinking 토큰 비활성화 (qwen3 계열, ~40% 속도 향상)
    - stream=False      : 스트리밍 비활성화 (안정적 단일 응답)
    - keep_alive=-1     : 모델 메모리 유지 (후속 요청 ~4초대 응답)
    - timeout=300       : 충분한 타임아웃 (복잡한 프롬프트 대비)

사용 예:
    from ollama_client import ask, ask_with_system, DEFAULT_MODEL

    result = ask("마르크스의 잉여가치론을 설명하라")
    result = ask("분석하라", model="qwen3.5:9b", temperature=0.9)
    result = ask_with_system("사용자 질문", system_prompt="너는 시인이다")
"""

import httpx
from typing import Optional

# ── 기본 설정 ─────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
TIMEOUT       = 300        # seconds — 복잡한 프롬프트도 여유있게
KEEP_ALIVE    = -1         # 모델 메모리 영구 유지
THINK         = False      # thinking 토큰 비활성화 (속도 최적화)
STREAM        = False      # 단일 응답 (스트리밍 없음)


# ── 핵심 함수 ─────────────────────────────────────────────────────────────────

def ask(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 20,
    timeout: int = TIMEOUT,
) -> str:
    """
    가장 간단한 Ollama 호출. 단일 문자열 반환.

    Args:
        prompt       : 사용자 프롬프트
        model        : 사용할 모델 (기본: qwen3.5:4b)
        system_prompt: 시스템 프롬프트 (선택)
        temperature  : 생성 다양성 (0.0~1.0, 기본 0.7)
        top_k        : 토큰 샘플링 범위 (기본 20)
        timeout      : 요청 타임아웃 초 (기본 300)

    Returns:
        LLM 응답 문자열

    Raises:
        httpx.HTTPStatusError : 서버 오류
        httpx.TimeoutException: 타임아웃
        KeyError              : 응답 형식 이상
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model":      model,
            "messages":   messages,
            "stream":     STREAM,
            "keep_alive": KEEP_ALIVE,
            "think":      THINK,
            "options": {
                "temperature": temperature,
                "top_k":       top_k,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ask_with_system(
    user_prompt: str,
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    시스템 프롬프트 포함 호출. ask()의 편의 래퍼.

    Args:
        user_prompt  : 사용자 입력
        system_prompt: 역할/행동 지침
        model        : 사용할 모델
        temperature  : 생성 다양성

    Returns:
        LLM 응답 문자열
    """
    return ask(
        prompt=user_prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
    )


def ask_chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    다중 턴 대화용 호출. messages 리스트를 직접 전달.

    Args:
        messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
        model   : 사용할 모델
        temperature: 생성 다양성

    Returns:
        LLM 응답 문자열

    Example:
        result = ask_chat([
            {"role": "system",    "content": "너는 레닌이다"},
            {"role": "user",      "content": "제국주의란?"},
            {"role": "assistant", "content": "자본주의의 최고 단계다."},
            {"role": "user",      "content": "구체적으로 설명하라"},
        ])
    """
    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model":      model,
            "messages":   messages,
            "stream":     STREAM,
            "keep_alive": KEEP_ALIVE,
            "think":      THINK,
            "options": {
                "temperature": temperature,
                "top_k":       20,
            },
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def check_ollama() -> dict:
    """
    Ollama 서버 상태 확인.

    Returns:
        {"status": "ok", "models": [...]} 또는 {"status": "error", "reason": "..."}
    """
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "ok", "models": models}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── 빠른 테스트 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Ollama 클라이언트 자가 테스트 ===\n")

    # 1) 서버 상태
    status = check_ollama()
    print(f"서버 상태: {status}")

    if status["status"] != "ok":
        print("Ollama 서버 응답 없음. 테스트 중단.")
        exit(1)

    # 2) 기본 호출
    print("\n[1] ask() 기본 호출...")
    result = ask("한 문장으로: 변증법적 유물론이란?")
    print(f"응답: {result[:200]}")

    # 3) 시스템 프롬프트
    print("\n[2] ask_with_system() 호출...")
    result2 = ask_with_system(
        user_prompt="제국주의란 무엇인가?",
        system_prompt="너는 레닌이다. 간결하고 단호하게 답하라.",
    )
    print(f"응답: {result2[:200]}")

    print("\n=== 테스트 완료 ===")
