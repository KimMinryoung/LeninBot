"""Common utilities for graffiti scripts."""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

MOON_IP = os.getenv("MOON_IPv4", "37.19.205.183")
MOON_LLM_URL = os.getenv("MOON_LLM_BASE_URL", f"http://{MOON_IP}:8080")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-9b")
API_URL = os.getenv("LENINBOT_API_URL", "https://leninbot.duckdns.org")
API_KEY = os.getenv("GRAFFITI_API_KEY", "")


def ask_local(prompt: str, max_tokens: int = 4096, temperature: float = 0.8) -> str:
    """MOON PC 우선, 로컬 Ollama 폴백."""
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    with httpx.Client(timeout=300.0) as client:
        # 1차: MOON PC (OpenAI 호환)
        try:
            resp = client.post(f"{MOON_LLM_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            return (msg.get("content") or "").strip()
        except Exception:
            pass
        # 2차: 로컬 Ollama
        resp = client.post(
            f"{LOCAL_LLM_URL}/api/chat",
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "keep_alive": -1,
                "think": False,
                "options": {"temperature": temperature},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


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
