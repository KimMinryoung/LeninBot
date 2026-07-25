#!/usr/bin/env python3
"""Pure in-process regression checks for outbound URL SSRF guards."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_fetch.url_security import (
    UnsafeUrlError,
    safe_requests_get,
    validate_public_http_url,
)
from content_fetch.urls import diagnose_url_fetch_failure


def _resolver(*addresses: str):
    def resolve(_host, port, **_kwargs):
        rows = []
        for address in addresses:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
            rows.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return rows
    return resolve


def _blocked(url: str, resolver=None) -> None:
    try:
        kwargs = {"resolver": resolver} if resolver else {}
        validate_public_http_url(url, **kwargs)
    except UnsafeUrlError:
        return
    raise AssertionError(f"unsafe URL was allowed: {url}")


class _Response:
    def __init__(self, status_code: int, location: str = ""):
        self.status_code = status_code
        self.headers = {"Location": location} if location else {}
        self.closed = False

    def close(self):
        self.closed = True


def main() -> int:
    public = _resolver("8.8.8.8")
    assert validate_public_http_url("https://example.com/path", resolver=public)
    assert validate_public_http_url("http://example.com:8080/path", resolver=public)

    for url in (
        "file:///etc/passwd",
        "gopher://example.com/",
        "http://user:pass@example.com/",
        "http://localhost/",
        "http://localhost./",
        "http://127.0.0.1/",
        "http://10.1.2.3/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "https://example.com:22/",
    ):
        _blocked(url, resolver=public)

    _blocked("https://mixed.example/", resolver=_resolver("8.8.8.8", "10.0.0.5"))
    assert "blocked_url" in diagnose_url_fetch_failure("http://127.0.0.1:8000/health")

    calls = []
    first = _Response(302, "http://127.0.0.1/admin")

    def private_redirect(url, **_kwargs):
        calls.append(url)
        return first

    try:
        safe_requests_get(
            "https://example.com/start",
            request_get=private_redirect,
            resolver=public,
        )
    except UnsafeUrlError:
        pass
    else:
        raise AssertionError("public-to-private redirect was allowed")
    assert calls == ["https://example.com/start"]
    assert first.closed

    calls.clear()
    responses = [
        _Response(302, "https://other.example/final"),
        _Response(200),
    ]

    def public_redirect(url, **_kwargs):
        calls.append(url)
        return responses.pop(0)

    response = safe_requests_get(
        "https://example.com/start",
        request_get=public_redirect,
        resolver=public,
    )
    assert response.status_code == 200
    assert calls == ["https://example.com/start", "https://other.example/final"]

    print("url security smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
