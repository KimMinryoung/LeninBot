"""URL extraction, fetch diagnostics, and URL content fallback fetching."""

import asyncio
import logging
import re as _re
import socket
from typing import Optional as _Optional
from urllib.parse import urlparse as _urlparse

from secrets_loader import get_secret
from content_fetch.url_security import (
    UnsafeUrlError,
    safe_requests_get,
    validate_public_http_url,
)
from content_fetch.browser_pool import (
    _is_low_quality,
    _playwright_fetch,
    _playwright_fetch_async,
    _pw_submit,
)

logger = logging.getLogger(__name__)

_URL_PATTERN = _re.compile(r'https?://[^\s<>\"\')]+')


def extract_urls(text: str) -> list[str]:
    """Extract HTTP/HTTPS URLs from text."""
    return _URL_PATTERN.findall(text)


def _classify_url_error_text(error_text: str) -> str | None:
    """Return a compact fetch/browser failure category from exception text."""
    text = (error_text or "").lower()
    if not text:
        return None
    if any(token in text for token in ("name or service not known", "temporary failure in name resolution", "dns")):
        return "dns_resolution_failed"
    if any(token in text for token in ("connecttimeout", "timed out", "timeout", "page.goto: timeout", "page.navigate() timed out")):
        return "origin_connection_timeout"
    if any(token in text for token in ("connection refused", "err_connection_refused")):
        return "origin_connection_refused"
    if any(token in text for token in ("err_connection_reset", "connection reset", "net::err_aborted")):
        return "origin_connection_reset"
    if any(token in text for token in ("ssl", "certificate", "cert_authority_invalid", "tls")):
        return "tls_or_certificate_error"
    if any(token in text for token in ("403", "forbidden", "access denied")):
        return "http_forbidden"
    if any(token in text for token in ("429", "too many requests", "rate limit")):
        return "http_rate_limited"
    if any(token in text for token in ("451", "unavailable for legal reasons")):
        return "http_legal_or_geo_block"
    if any(token in text for token in ("captcha", "challenge", "cloudflare")):
        return "anti_bot_challenge"
    return None


def diagnose_url_fetch_failure(url: str, observed_errors: list[str] | None = None, timeout: float = 8.0) -> str:
    """Diagnose a failed URL fetch with actionable categories.

    This is deliberately advisory. It should help agents choose the next step,
    not decide that a source is permanently unreachable.
    """
    try:
        url = validate_public_http_url(url)
    except UnsafeUrlError as exc:
        return f"Fetch diagnosis: blocked_url - {exc}."

    parsed = _urlparse(url)
    host = parsed.hostname
    scheme = parsed.scheme
    if not host:
        return "Fetch diagnosis: invalid_url - the URL has no hostname."

    categories = [_classify_url_error_text(e) for e in (observed_errors or [])]
    categories = [c for c in categories if c]
    if categories:
        category = categories[0]
        if category == "origin_connection_timeout":
            return (
                "Fetch diagnosis: origin_connection_timeout - the browser/fetcher timed out before "
                "the origin responded. If TCP checks also time out, this is usually server-network, "
                "datacenter, geography, or firewall filtering rather than a page parsing issue. "
                "Use another network/proxy, search for mirrors, or ask for user-supplied page content."
            )
        if category == "http_forbidden":
            return "Fetch diagnosis: http_forbidden - the site returned or implied HTTP 403/access denied. Try browser automation, cookies/login, or an alternate source."
        if category == "http_rate_limited":
            return "Fetch diagnosis: http_rate_limited - the site appears to be rate limiting. Wait, reduce retries, or use an authenticated/alternate route."
        if category == "http_legal_or_geo_block":
            return "Fetch diagnosis: http_legal_or_geo_block - the site appears unavailable for legal/geographic reasons from this server."
        if category == "anti_bot_challenge":
            return "Fetch diagnosis: anti_bot_challenge - the site likely presented a CAPTCHA or anti-bot challenge. Use authenticated/manual access or another source."

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        validate_public_http_url(url, resolver=lambda *_args, **_kwargs: infos)
    except UnsafeUrlError as e:
        return f"Fetch diagnosis: blocked_url - {e}."
    except socket.gaierror as e:
        return f"Fetch diagnosis: dns_resolution_failed - this server cannot resolve {host}: {e}."
    except Exception as e:
        return f"Fetch diagnosis: dns_check_failed - DNS check for {host} failed unexpectedly: {e}."

    addresses = []
    for family, socktype, proto, _canonname, sockaddr in infos:
        ip = sockaddr[0]
        if ip not in addresses:
            addresses.append(ip)
    first_error = None
    for family, socktype, proto, _canonname, sockaddr in infos[:4]:
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
            sock.close()
            break
        except TimeoutError as e:
            first_error = e
            continue
        except ConnectionRefusedError as e:
            first_error = e
            return (
                f"Fetch diagnosis: origin_connection_refused - {host}:{port} resolves to "
                f"{', '.join(addresses[:4])}, but the TCP connection was refused from this server."
            )
        except OSError as e:
            first_error = e
            continue
        finally:
            try:
                sock.close()
            except Exception:
                pass
    else:
        if isinstance(first_error, TimeoutError):
            return (
                f"Fetch diagnosis: origin_tcp_timeout - {host}:{port} resolves to "
                f"{', '.join(addresses[:4])}, but TCP connection attempts time out from this server. "
                "This usually points to server IP/geography/datacenter filtering or upstream firewall drops. "
                "Use a reachable proxy/VPN/relay, search for mirrors, or ask the user to provide the content."
            )
        return (
            f"Fetch diagnosis: origin_tcp_unreachable - {host}:{port} resolves to "
            f"{', '.join(addresses[:4])}, but this server could not establish TCP "
            f"({first_error})."
        )

    try:
        import requests as _req
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"}
        resp = safe_requests_get(url, request_get=_req.get, headers=headers, timeout=timeout, stream=True)
        status = resp.status_code
        resp.close()
        if status in (401, 403):
            return f"Fetch diagnosis: http_forbidden - TCP works, but the site returned HTTP {status}. Try browser cookies/login or another source."
        if status == 429:
            return "Fetch diagnosis: http_rate_limited - TCP works, but the site returned HTTP 429. Wait or use a lower-rate/alternate route."
        if status == 451:
            return "Fetch diagnosis: http_legal_or_geo_block - TCP works, but the site returned HTTP 451."
        if 500 <= status <= 599:
            return f"Fetch diagnosis: origin_server_error - TCP works, but the origin returned HTTP {status}."
        return (
            f"Fetch diagnosis: extraction_or_dynamic_page - TCP/HTTP reached the site (HTTP {status}), "
            "but usable text was not extracted. Try browse_web, a site-specific extractor, or manual content."
        )
    except Exception as e:
        category = _classify_url_error_text(str(e))
        if category == "tls_or_certificate_error":
            return f"Fetch diagnosis: tls_or_certificate_error - TCP works, but TLS/certificate validation failed: {e}."
        return f"Fetch diagnosis: http_probe_failed - TCP works, but the HTTP probe failed: {e}."


def _fetch_url_fallbacks(url: str, max_chars: int = 10000) -> _Optional[str]:
    """SSRF-safe fallback chain after Playwright: Tavily → requests+BeautifulSoup.

    Crawl4AI is deliberately not used here because its internal redirect and
    subresource fetches cannot share the common destination validator.
    """
    def _clean_text(raw: str) -> str:
        """Strip boilerplate noise and keep only substantive paragraphs."""
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        cleaned = []
        for line in lines:
            if len(line) < 4 and not any(c.isalpha() for c in line):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    url = validate_public_http_url(url)

    # 1) Tavily Extract (remote extraction; skip if key/quota unavailable)
    best_fallback = None
    tavily_key = get_secret("TAVILY_API_KEY", "") or ""
    if tavily_key:
        try:
            from langchain_tavily import TavilyExtract
            extractor = TavilyExtract(tavily_api_key=tavily_key)
            result = extractor.invoke({"urls": [url]})
            if isinstance(result, dict) and result.get("error"):
                raise ValueError(result["error"])
            items = []
            if isinstance(result, dict) and result.get("results"):
                items = result["results"]
            elif isinstance(result, list):
                items = result
            if items:
                item = items[0] if isinstance(items[0], dict) else {"content": str(items[0])}
                content = item.get("raw_content", "") or item.get("content", "")
                if content and len(content) > 50:
                    cleaned = _clean_text(content)[:max_chars]
                    if not _is_low_quality(cleaned):
                        return cleaned
                    best_fallback = cleaned
        except Exception as e:
            logger.info("[URL] Tavily Extract 실패 (%s): %s", url[:60], e)

    # 2) requests + BeautifulSoup
    try:
        import requests as _req
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        }
        resp = safe_requests_get(url, request_get=_req.get, headers=headers, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        from bs4 import BeautifulSoup, Comment
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                         "iframe", "noscript", "form", "button", "svg", "img"]):
            tag.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        _BOILERPLATE = _re.compile(
            r"(sidebar|widget|advert|banner|cookie|popup|modal|share|social|"
            r"related|recommend|comment|reply|breadcrumb|pagination|menu|"
            r"toolbar|tooltip|disclaimer|copyright|footer|signup|login|"
            r"newsletter|promo)", _re.I
        )
        for el in soup.find_all(attrs={"class": _BOILERPLATE}):
            el.decompose()
        for el in soup.find_all(attrs={"id": _BOILERPLATE}):
            el.decompose()

        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=_re.compile(r"(article[_-]?body|post[_-]?(body|content)|entry[_-]?content|se-main-container)", _re.I))
            or soup.find(class_=_re.compile(r"(content|article|post|entry)", _re.I))
            or soup.find(id=_re.compile(r"(content|article|post|main)", _re.I))
        )
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)

        text = _clean_text(text)

        if len(text) > 50:
            if not _is_low_quality(text):
                return text[:max_chars]
            if best_fallback is None or len(text) > len(best_fallback):
                best_fallback = text[:max_chars]
    except Exception as e:
        logger.warning("[URL] requests fallback도 실패 (%s): %s", url[:60], e)

    # Return best fallback result even if low-quality (better than nothing)
    return best_fallback


def fetch_url_content(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Sync entry point with public-network validation on every local fetch."""
    url = validate_public_http_url(url)
    try:
        result = _playwright_fetch(url, max_chars)
        if result and not _is_low_quality(result):
            return result
    except Exception as e:
        logger.info("[URL] Playwright 실패 (%s): %s", url[:60], e)
    return _fetch_url_fallbacks(url, max_chars)


async def fetch_url_content_async(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Async entry point with public-network validation on every local fetch."""
    url = await asyncio.to_thread(validate_public_http_url, url)
    try:
        fut = _pw_submit(_playwright_fetch_async(url, max_chars))
        result = await asyncio.wrap_future(fut)
        if result and not _is_low_quality(result):
            return result
    except Exception as e:
        logger.info("[URL] Playwright 실패 (%s): %s", url[:60], e)
    return await asyncio.to_thread(_fetch_url_fallbacks, url, max_chars)


def fetch_urls_as_documents(urls: list[str], logs: list | None = None) -> list:
    """Fetch content from multiple URLs and return as Document-compatible dicts.

    Returns list of dicts with 'page_content' and 'metadata' keys.
    If langchain Document class is available, returns Document objects.
    Accepts optional logs list to append progress messages.
    """
    if logs is None:
        logs = []
    results = []
    for url in urls[:3]:  # Limit to 3 URLs max
        logs.append(f"🔗 [URL] 웹 페이지 내용 확인 중: {url[:80]}...")
        content = fetch_url_content(url)
        if content:
            domain = _urlparse(url).netloc
            try:
                from langchain_core.documents import Document
                doc = Document(
                    page_content=content,
                    metadata={"source": url, "title": f"[{domain}] URL 직접 참조"},
                )
            except ImportError:
                doc = {
                    "page_content": content,
                    "metadata": {"source": url, "title": f"[{domain}] URL 직접 참조"},
                }
            results.append(doc)
            logs.append(f"   ✅ {len(content)}자의 본문 내용을 확보했습니다.")
        else:
            logs.append(f"   ⚠️ 페이지 내용을 가져올 수 없습니다.")
            # Create a failure document so the LLM knows the fetch failed
            # (prevents hallucination from URL text alone)
            fail_msg = (
                f"[FETCH FAILED] URL: {url}\n"
                "이 URL의 본문을 가져오는 데 실패했습니다. "
                "URL 텍스트만으로 내용을 추측하거나 환각하지 마세요. "
                "사용자에게 페이지 접근에 실패했음을 알리고, "
                "직접 내용을 복사해서 붙여넣거나 다른 URL을 제공하도록 안내하세요."
            )
            try:
                from langchain_core.documents import Document
                doc = Document(
                    page_content=fail_msg,
                    metadata={"source": url, "title": "[FETCH FAILED]", "fetch_failed": True},
                )
            except ImportError:
                doc = {
                    "page_content": fail_msg,
                    "metadata": {"source": url, "title": "[FETCH FAILED]", "fetch_failed": True},
                }
            results.append(doc)
    return results

