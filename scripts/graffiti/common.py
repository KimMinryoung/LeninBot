"""Common utilities for graffiti scripts."""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

MOON_LLM_URL = os.getenv("MOON_LLM_BASE_URL", "http://127.0.0.1:8080")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11435")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-9b")
API_URL = os.getenv("LENINBOT_API_URL", "https://leninbot.duckdns.org")
API_KEY = os.getenv("GRAFFITI_API_KEY", "")


def ask_local(prompt: str, max_tokens: int = 4096, temperature: float = 0.8) -> str:
    """MOON PC 우선, 로컬 llama-server 폴백. 양쪽 다 OpenAI 호환 API."""
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    with httpx.Client(timeout=300.0) as client:
        for base_url in [MOON_LLM_URL, LOCAL_LLM_URL]:
            try:
                resp = client.post(f"{base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                return (msg.get("content") or "").strip()
            except Exception:
                continue
        raise ConnectionError("MOON PC와 로컬 llama-server 모두 응답 없음")


def post_graffiti(endpoint: str, data: dict) -> dict:
    """Post to the graffiti API."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{API_URL}/graffiti/{endpoint}",
            json=data,
            headers={"X-Graffiti-Key": API_KEY},
        )
        resp.raise_for_status()
        return resp.json()
