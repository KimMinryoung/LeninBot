"""Common utilities for graffiti scripts."""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-9b")
API_URL = os.getenv("LENINBOT_API_URL", "https://leninbot.duckdns.org")
API_KEY = os.getenv("GRAFFITI_API_KEY", "")


def ask_local(prompt: str, max_tokens: int = 4096, temperature: float = 0.8) -> str:
    """Synchronous call to local LLM."""
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            f"{LOCAL_LLM_URL}/v1/chat/completions",
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return (msg.get("content") or "").strip()


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
