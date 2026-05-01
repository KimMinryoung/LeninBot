#!/usr/bin/env python3
"""Register the configured Mersoom account and persist credentials to .env."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
import string
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
BASE_URL = "https://www.mersoom.com/api"
DEFAULT_AUTH_ID = "razvedchik"
PASSWORD_LENGTH = 16
POW_RETRIES = 3
SPECIALS = "!@#$%^&*_-+="


class MersoomError(RuntimeError):
    pass


class MersoomHTTPError(MersoomError):
    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self.payload = payload
        super().__init__(f"HTTP {status}: {json.dumps(payload, ensure_ascii=False)}")


def generate_password(length: int = PASSWORD_LENGTH) -> str:
    if length < 10 or length > 20:
        raise ValueError("Mersoom password length must be 10-20 characters")

    required = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(SPECIALS),
    ]
    alphabet = string.ascii_letters + string.digits + SPECIALS
    rest = [secrets.choice(alphabet) for _ in range(length - len(required))]
    chars = required + rest
    random.SystemRandom().shuffle(chars)
    return "".join(chars)


def request_json(method: str, path: str, *, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    url = BASE_URL + path
    data = None
    request_headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    req = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"text": raw[:1000]}
        raise MersoomHTTPError(exc.code, payload) from exc
    except URLError as exc:
        raise MersoomError(f"Network error: {exc}") from exc


def extract_challenge(data: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(data, dict):
        raise MersoomError(f"Unexpected challenge response: {data!r}")

    token = data.get("token")
    challenge = data.get("challenge")
    if isinstance(challenge, dict) and token:
        return challenge, str(token)

    if token and ("seed" in data or "type" in data):
        return data, str(token)

    raise MersoomError(f"Unexpected challenge response: {json.dumps(data, ensure_ascii=False)[:500]}")


def solve_pow(challenge: dict[str, Any]) -> str:
    seed = str(challenge["seed"])
    target_prefix = str(challenge.get("target_prefix", "0000"))
    limit_ms = int(challenge.get("limit_ms", 15000))
    deadline = time.monotonic() + max(limit_ms / 1000.0, 1.0)

    nonce = 0
    while time.monotonic() < deadline:
        digest = hashlib.sha256(f"{seed}{nonce}".encode("utf-8")).hexdigest()
        if digest.startswith(target_prefix):
            return str(nonce)
        nonce += 1
    raise TimeoutError(f"PoW nonce not found within {limit_ms}ms")


def solve_documented_ai_puzzle(challenge: dict[str, Any]) -> str:
    """Solve the word-index puzzle documented in Mersoom skills.md.

    Unknown puzzle formats fail loudly. The caller can request a fresh challenge.
    """

    text = "\n".join(
        str(challenge.get(key, ""))
        for key in ("question", "prompt", "puzzle", "description", "text")
        if challenge.get(key)
    )
    words_match = re.search(r"(?:words?|단어)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if words_match:
        words_part = words_match.group(1).splitlines()[0]
        words = re.findall(r"[A-Za-z]+", words_part)
    else:
        candidates = re.findall(r"\b[A-Za-z]{2,}\b", text)
        words = candidates[-12:] if len(candidates) >= 3 else candidates

    korean_ordinals = {
        "첫": 1,
        "두": 2,
        "세": 3,
        "네": 4,
        "다섯": 5,
        "여섯": 6,
        "일곱": 7,
        "여덟": 8,
        "아홉": 9,
        "열": 10,
    }
    groups = re.findall(r"\[([^\]]+)\]", text)
    index_groups: list[list[int]] = []
    for group in groups:
        values = [int(n) for n in re.findall(r"(\d+)\s*번째", group)]
        if not values:
            values = [korean_ordinals[w] for w in korean_ordinals if f"{w}번째" in group]
        if values:
            index_groups.append(values)

    if len(index_groups) < 2 or not words:
        raise MersoomError(f"Unsupported AI Puzzle format: {text[:300]}")

    word_indexes, char_indexes = index_groups[0], index_groups[1]
    if len(word_indexes) != len(char_indexes):
        raise MersoomError(f"Unsupported AI Puzzle index shape: {text[:300]}")

    answer_chars: list[str] = []
    for word_index, char_index in zip(word_indexes, char_indexes):
        word = words[word_index - 1]
        answer_chars.append(word[char_index - 1])

    answer = "".join(answer_chars)
    if "역순" in text or "reverse" in text.lower():
        answer = answer[::-1]
    if "소문자" in text or "lower" in text.lower():
        answer = answer.lower()
    if "대문자" in text or "upper" in text.lower():
        answer = answer.upper()
    return answer


def solve_challenge(challenge: dict[str, Any]) -> str:
    challenge_type = str(challenge.get("type", "PoW")).strip().lower()
    if challenge_type in {"pow", "proof_of_work", "proof-of-work", ""} or "seed" in challenge:
        return solve_pow(challenge)
    if "puzzle" in challenge_type or "ai" in challenge_type:
        return solve_documented_ai_puzzle(challenge)
    raise MersoomError(f"Unsupported challenge type: {challenge.get('type')!r}")


def challenge_headers() -> tuple[dict[str, str], dict[str, Any], str]:
    data = request_json("POST", "/challenge", body={})
    challenge, token = extract_challenge(data)
    proof = solve_challenge(challenge)
    return {
        "X-Mersoom-Token": token,
        "X-Mersoom-Proof": proof,
    }, challenge, proof


def dotenv_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def update_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    keys_left = dict(values)
    out: list[str] = []
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")

    for line in lines:
        match = pattern.match(line)
        if match and match.group(1) in keys_left:
            key = match.group(1)
            out.append(f"{key}={dotenv_quote(keys_left.pop(key))}")
        else:
            out.append(line)

    if keys_left and out and out[-1].strip():
        out.append("")
    for key, value in keys_left.items():
        out.append(f"{key}={dotenv_quote(value)}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def register(auth_id: str, password: str, max_attempts: int = POW_RETRIES) -> dict[str, Any]:
    payload = {"auth_id": auth_id, "password": password}
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            headers, challenge, proof = challenge_headers()
            response = request_json("POST", "/auth/register", body=payload, headers=headers)
            return {
                "registered": True,
                "status": 200,
                "auth_id": auth_id,
                "response": response,
                "challenge_type": challenge.get("type", "PoW"),
                "proof": proof,
                "attempt": attempt,
            }
        except TimeoutError as exc:
            last_error = exc
            continue
        except MersoomHTTPError as exc:
            if exc.status == 409:
                return {
                    "registered": False,
                    "status": 409,
                    "auth_id": auth_id,
                    "response": exc.payload,
                    "attempt": attempt,
                }
            raise
        except MersoomError as exc:
            last_error = exc
            continue

    raise MersoomError(f"Registration challenge failed after {max_attempts} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-id", default=DEFAULT_AUTH_ID)
    parser.add_argument("--password", default=os.environ.get("MERSOOM_PASSWORD") or generate_password())
    parser.add_argument("--env", type=Path, default=ENV_PATH)
    args = parser.parse_args()

    if args.auth_id != DEFAULT_AUTH_ID:
        raise SystemExit("This mission requires auth_id exactly 'razvedchik'")
    if len(args.auth_id) != 10:
        raise SystemExit("auth_id must be exactly 10 characters")

    result = register(args.auth_id, args.password)
    if result.get("registered"):
        update_env(args.env, {"MERSOOM_AUTH_ID": args.auth_id, "MERSOOM_PASSWORD": args.password})

    public_result = dict(result)
    public_result["password_saved"] = bool(result.get("registered"))
    public_result.pop("proof", None)
    print(json.dumps(public_result, ensure_ascii=False, indent=2))
    return 0 if result.get("registered") or result.get("status") == 409 else 1


if __name__ == "__main__":
    sys.exit(main())
