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
import re as _re
_THINK_RE = _re.compile(r"<think>.*?</think>\s*", _re.DOTALL)
# Qwen3.5 outputs "Thinking Process:" as plain text when --reasoning off
_THINKING_PROCESS_RE = _re.compile(
    r"^(Thinking Process:?|Thought Process:?)\s*\n",
    _re.IGNORECASE,
)
# Markers that indicate the model's final draft within its thinking
_DRAFT_MARKERS = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Final|Revised|Refined|Polished|Selected)\s*(?:Check|Draft|Version|Answer|Response|Output|Polish)?:?"
    r"|Let'?s?\s+finalize"
    r"|(?:Drafting\s*-?\s*Attempt\s*\d+)"
    r")\s*\n",
    _re.IGNORECASE,
)


def _extract_answer(msg: dict) -> str:
    """Extract the actual answer from an LLM response message.

    Handles three cases:
    1. Normal response — content has the answer directly.
    2. <think> tags — llama-server splits into reasoning_content + content.
       If content is empty (tokens exhausted on thinking), extract from reasoning.
    3. Plain-text thinking — model outputs "Thinking Process:" as regular text
       (qwen3.5 with --reasoning off). Parse out the final draft.
    """
    content = (msg.get("content") or "").strip()

    # Strip <think> tags if leaked into content
    if content:
        stripped = _THINK_RE.sub("", content).strip()
        if stripped:
            content = stripped

    # Strip plain-text thinking prefix
    if content and _THINKING_PROCESS_RE.match(content):
        content = _extract_from_thinking(content)

    if content:
        return content

    # content is empty — try reasoning_content (llama-server separated it)
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not reasoning:
        return ""

    return _extract_from_thinking(reasoning)


def _extract_from_thinking(text: str) -> str:
    """Extract the final answer draft from a thinking/reasoning block."""
    # Find the last draft marker and take text after it
    markers = list(_DRAFT_MARKERS.finditer(text))
    if markers:
        last = markers[-1]
        candidate = text[last.end():].strip()
        # Clean up: take first paragraph (the draft), skip further analysis
        lines = []
        for line in candidate.splitlines():
            line = line.strip()
            if not line:
                if lines:
                    break
                continue
            # Stop if model starts analyzing again
            if _re.match(r"^(Count|Char|Length|Check|Wait|Oops|Let'?s|Note:|\d+\.\s)", line, _re.I):
                break
            # Strip surrounding quotes
            if len(line) > 2 and line[0] == '"' and line[-1] == '"':
                line = line[1:-1]
            lines.append(line)
        result = " ".join(lines).strip()
        if len(result) > 20:
            return result

    # Fallback: find quoted text that looks like a final answer
    quotes = _re.findall(r'"([^"]{20,})"', text)
    if quotes:
        return quotes[-1].strip()

    # Last resort: take last substantive line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if not _re.match(r"^(\d+[\.\):]|\*\*|[-*•]|#{1,3}\s|Count|Char|Wait|Oops|Let)", line) and len(line) > 20:
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            return line
    return ""


def _call(base: str, model: str, messages: list[dict],
          temperature: float, timeout: int) -> str:
    resp = httpx.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 8192,
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
