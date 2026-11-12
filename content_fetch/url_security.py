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
    """Return a stripped public HTTP(S) URL or raise :class:`UnsafeUrlError`."""
    return resolve_public_http_url(url, resolver=resolver)[0]


def resolve_public_http_url(
    url: str,
    *,
    resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
) -> tuple[str, str, list[str]]:
    """Validate ``url`` and return ``(url, host, public_addresses)``.

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
        return value, host, [_public_ip(host).compressed]

    try:
        infos = resolver(host, effective_port, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise UnsafeUrlError(f"URL hostname could not be resolved: {host}") from exc
    if not infos:
        raise UnsafeUrlError(f"URL hostname has no addresses: {host}")

    addresses: list[str] = []
    for info in infos:
        try:
            raw = str(info[4][0])
            address = _public_ip(raw).compressed
            if address not in addresses:
                addresses.append(address)
        except (IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, UnsafeUrlError):
                raise
            raise UnsafeUrlError(f"invalid DNS answer for {host}") from exc
    if not addresses:
        raise UnsafeUrlError(f"URL hostname has no usable addresses: {host}")
    return value, host, addresses


class _PinnedConnectionMixin:
    """urllib3 connection that dials only the addresses validated for its host.

    ``self.host`` keeps the hostname, so the Host header, SNI and certificate
    verification are unchanged; only the TCP destination is fixed. Without this,
    requests resolves the name a second time and a rebinding DNS server can
    answer that lookup with a private address.
    """

    _pins: dict[str, list[str]] = {}

    def _new_conn(self):  # type: ignore[override]
        from socket import timeout as socket_timeout

        from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
        from urllib3.util import connection as util_connection

        addresses = self._pins.get(self.host.lower())
        if not addresses:
            raise NewConnectionError(self, f"Refusing to connect to unvalidated host {self.host}")
        error: Exception | None = None
        for address in addresses:
            try:
                return util_connection.create_connection(
                    (address, self.port), self.timeout,
                    source_address=self.source_address, socket_options=self.socket_options,
                )
            except socket_timeout:
                error = ConnectTimeoutError(
                    self, f"Connection to {self.host} timed out. (connect timeout={self.timeout})")
            except OSError as exc:
                error = NewConnectionError(self, f"Failed to establish a new connection: {exc}")
        raise error  # type: ignore[misc]


def _pinned_session(host: str, addresses: list[str]):
    """A requests session whose connections may reach only ``host`` at ``addresses``."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.connection import HTTPConnection, HTTPSConnection
    from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

    pins = {host.lower(): list(addresses)}

    class PinnedAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            pools = {}
            for scheme, pool_cls, conn_cls in (("http", HTTPConnectionPool, HTTPConnection),
                                               ("https", HTTPSConnectionPool, HTTPSConnection)):
                pinned_conn = type(f"Pinned{conn_cls.__name__}", (_PinnedConnectionMixin, conn_cls),
                                   {"_pins": pins})
                pools[scheme] = type(f"Pinned{pool_cls.__name__}", (pool_cls,), {"ConnectionCls": pinned_conn})
            self.poolmanager.pool_classes_by_scheme = pools

    session = requests.Session()
    # Environment proxies would move DNS resolution to the proxy and bypass the pin.
    session.trust_env = False
    adapter = PinnedAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _pinned_get(url: str, host: str, addresses: list[str], **kwargs: Any):
    session = _pinned_session(host, addresses)
    try:
        response = session.get(url, **kwargs)
    except BaseException:
        session.close()
        raise
    # A streamed body still needs the pool; close the session with the response.
    close_response = response.close

    def close() -> None:
        try:
            close_response()
        finally:
            session.close()

    response.close = close  # type: ignore[method-assign]
    return response


def safe_requests_get(
    url: str,
    *,
    request_get: Callable[..., Any] | None = None,
    resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
    max_redirects: int = MAX_REDIRECTS,
    **kwargs: Any,
):
    """GET ``url`` following redirects, validating and pinning every hop.

    Each hop's hostname is resolved once; all answers must be public, and the
    connection dials exactly those addresses (no second lookup to rebind).
    ``request_get`` replaces the pinned transport (tests, special callers).
    """
    kwargs.pop("allow_redirects", None)
    current, host, addresses = resolve_public_http_url(url, resolver=resolver)
    for redirect_count in range(max_redirects + 1):
        if request_get is None:
            response = _pinned_get(current, host, addresses, allow_redirects=False, **kwargs)
        else:
            response = request_get(current, allow_redirects=False, **kwargs)
        if response.status_code not in REDIRECT_STATUS_CODES:
            return response

        location = str(response.headers.get("Location") or "").strip()
        if not location:
            return response
        next_url = urljoin(current, location)
        try:
            next_url, host, addresses = resolve_public_http_url(next_url, resolver=resolver)
        except Exception:
            response.close()
            raise
        response.close()
        if redirect_count >= max_redirects:
            raise UnsafeUrlError(f"redirect limit exceeded ({max_redirects})")
        current = next_url

    raise UnsafeUrlError(f"redirect limit exceeded ({max_redirects})")
