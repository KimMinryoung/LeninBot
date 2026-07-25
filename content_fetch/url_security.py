"""SSRF-safe URL validation and redirect handling for outbound fetch tools."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlsplit

SAFE_HTTP_SCHEMES = frozenset({"http", "https"})
SAFE_WEB_PORTS = frozenset({80, 443, 8080, 8443})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
MAX_URL_LENGTH = 4096
MAX_REDIRECTS = 5


class UnsafeUrlError(ValueError):
    """Raised when an outbound URL can reach a non-public destination."""


def _public_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    if not address.is_global:
        raise UnsafeUrlError(f"destination IP is not public: {address.compressed}")
    return address


def validate_public_http_url(
    url: str,
    *,
    resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
) -> str:
    """Return a stripped public HTTP(S) URL or raise :class:`UnsafeUrlError`.

    Every DNS answer must be globally routable. Rejecting mixed public/private
    answers prevents a hostname from smuggling a private fallback address into
    Playwright or requests. Redirect targets must be passed through this function
    again before they are followed.
    """
    value = str(url or "").strip()
    if not value:
        raise UnsafeUrlError("URL is required")
    if len(value) > MAX_URL_LENGTH:
        raise UnsafeUrlError(f"URL exceeds {MAX_URL_LENGTH} characters")
    if any(ord(ch) < 0x20 or ch.isspace() for ch in value):
        raise UnsafeUrlError("URL contains whitespace or control characters")

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError(f"invalid URL: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in SAFE_HTTP_SCHEMES:
        raise UnsafeUrlError(f"URL scheme is not allowed: {scheme or '(missing)'}")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("URL userinfo is not allowed")

    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeUrlError("URL hostname is required")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise UnsafeUrlError(f"local hostname is not allowed: {host}")

    effective_port = port or (443 if scheme == "https" else 80)
    if effective_port not in SAFE_WEB_PORTS:
        raise UnsafeUrlError(f"URL port is not allowed: {effective_port}")

    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        pass
    else:
        _public_ip(host)
        return value

    try:
        infos = resolver(host, effective_port, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise UnsafeUrlError(f"URL hostname could not be resolved: {host}") from exc
    if not infos:
        raise UnsafeUrlError(f"URL hostname has no addresses: {host}")

    addresses: set[str] = set()
    for info in infos:
        try:
            raw = str(info[4][0])
            addresses.add(_public_ip(raw).compressed)
        except (IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, UnsafeUrlError):
                raise
            raise UnsafeUrlError(f"invalid DNS answer for {host}") from exc
    if not addresses:
        raise UnsafeUrlError(f"URL hostname has no usable addresses: {host}")
    return value


def safe_requests_get(
    url: str,
    *,
    request_get: Callable[..., Any] | None = None,
    resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
    max_redirects: int = MAX_REDIRECTS,
    **kwargs: Any,
):
    """Run ``requests.get`` while validating every redirect before following it."""
    if request_get is None:
        import requests

        request_get = requests.get

    kwargs.pop("allow_redirects", None)
    current = validate_public_http_url(url, resolver=resolver)
    for redirect_count in range(max_redirects + 1):
        response = request_get(current, allow_redirects=False, **kwargs)
        if response.status_code not in REDIRECT_STATUS_CODES:
            return response

        location = str(response.headers.get("Location") or "").strip()
        if not location:
            return response
        next_url = urljoin(current, location)
        try:
            next_url = validate_public_http_url(next_url, resolver=resolver)
        except Exception:
            response.close()
            raise
        response.close()
        if redirect_count >= max_redirects:
            raise UnsafeUrlError(f"redirect limit exceeded ({max_redirects})")
        current = next_url

    raise UnsafeUrlError(f"redirect limit exceeded ({max_redirects})")
