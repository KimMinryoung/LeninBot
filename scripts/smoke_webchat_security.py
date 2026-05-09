#!/usr/bin/env python3
"""Smoke checks for public web-chat request boundaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["WEBCHAT_PROXY_SECRET"] = "smoke-secret"


class _Req:
    def __init__(self, headers: dict[str, str], host: str = "127.0.0.1"):
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.client = type("Client", (), {"host": host})()


def main() -> int:
    import api

    direct = _Req({"x-user-fingerprints": "fp-a,fp-b"})
    assert api._parse_user_fingerprints(direct) == []

    trusted = _Req({
        "x-webchat-proxy-secret": "smoke-secret",
        "x-user-fingerprints": "fp-a, fp-b",
    })
    assert api._parse_user_fingerprints(trusted) == ["fp-a", "fp-b"]

    wrong = _Req({
        "x-webchat-proxy-secret": "wrong",
        "x-user-fingerprints": "fp-a",
        "x-forwarded-for": "203.0.113.7",
    })
    assert api._parse_user_fingerprints(wrong) == []
    assert api._client_ip(wrong) == "127.0.0.1"

    forwarded = _Req({
        "x-webchat-proxy-secret": "smoke-secret",
        "x-forwarded-for": "203.0.113.9, 10.0.0.1",
    })
    assert api._client_ip(forwarded) == "203.0.113.9"

    print("webchat security smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
