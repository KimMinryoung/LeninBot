#!/usr/bin/env python
"""smoke_kimi_k3.py — Kimi K3 실호출 스모크 (키·결제 확인용, ~수십 토큰).

root에서 credstore 자격으로 실행:
  systemd-run --wait --pipe --collect \
    -p LoadCredentialEncrypted=moonshot_api_key:/etc/credstore.encrypted/moonshot_api_key.cred \
    /home/grass/leninbot/venv/bin/python /home/grass/leninbot/scripts/smoke_kimi_k3.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_key() -> str:
    cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if cred_dir:
        path = Path(cred_dir) / "moonshot_api_key"
        if path.exists():
            return path.read_text().strip()
    key = (os.environ.get("MOONSHOT_API_KEY") or "").strip()
    if key:
        return key
    print("FAIL: moonshot_api_key credential not found (run via systemd-run, see docstring)")
    sys.exit(1)


def main() -> None:
    from openai import OpenAI

    key = _load_key()
    print(f"key loaded: {key[:6]}...{key[-4:]} ({len(key)} chars)")

    client = OpenAI(api_key=key, base_url="https://api.moonshot.ai/v1", timeout=60)
    response = client.chat.completions.create(
        model="kimi-k3",
        messages=[{"role": "user", "content": "Reply with exactly: KIMI_K3_SMOKE_OK"}],
        # K3는 temperature=1만 허용 (그 외 400), 추론 모델이라 max_tokens에 추론분 여유 필요
        max_tokens=512,
    )
    msg = response.choices[0].message
    text = (msg.content or "").strip()
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    usage = response.usage
    print(f"model: {response.model}")
    print(f"reply: {text!r}")
    if reasoning:
        print(f"reasoning ({len(reasoning)} chars): {reasoning[:120]!r}...")
    print(f"usage: in={usage.prompt_tokens} out={usage.completion_tokens}")
    if "KIMI_K3_SMOKE_OK" in text:
        print("RESULT: PASS — 키·결제·K3 호출 정상")
    else:
        print("RESULT: PARTIAL — 호출은 성공했으나 응답 형식이 예상과 다름 (기능상 문제 없음)")


if __name__ == "__main__":
    main()
